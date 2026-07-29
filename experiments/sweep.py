"""Run the hyperparameter grids across processes.

    uv run experiments/sweep.py --all --workers 6
    uv run experiments/sweep.py --algo ppo --workers 4
    uv run experiments/sweep.py --algo dqn --only dqn-04 dqn-09

Roughly 35 to 50 minutes for all 40 runs at 6 workers on an 18 core machine.
Results land in logs/<run-id>/ and are collected into experiments/results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / "configs"
RESULTS_CSV = Path(__file__).parent / "results.csv"
REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_TIMEOUT_S = 3600

# Flags the training parsers expose as store_true rather than as a value
STORE_TRUE = {"normalise_advantage"}
# use_rms_prop defaults on, and is turned off with --adam instead of a value
NEGATED = {"use_rms_prop": "--adam"}
SKIP_KEYS = {"id", "note"}


def load_grid(algo: str) -> dict:
    """Read a config YAML and merge defaults into every run."""
    cfg = yaml.safe_load((CONFIG_DIR / f"{algo}.yaml").read_text())
    defaults = cfg.get("defaults", {}) or {}
    runs = [{**defaults, **run} for run in cfg["runs"]]
    return {"algo": cfg["algo"], "script": cfg["script"], "runs": runs}


def build_cmd(algo: str, script: str, run_cfg: dict) -> list[str]:
    """Turn one grid row into a training CLI invocation."""
    cmd = [sys.executable, script]
    if algo != "dqn":
        cmd += ["--algo", algo]
    cmd += ["--run-name", str(run_cfg["id"])]

    for key, value in run_cfg.items():
        if key in SKIP_KEYS:
            continue
        flag = "--" + key.replace("_", "-")
        if key in NEGATED:
            if not value:
                cmd.append(NEGATED[key])
            continue
        if key in STORE_TRUE:
            if value:
                cmd.append(flag)
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            cmd += [flag] + [str(v) for v in value]
            continue
        cmd += [flag, str(value)]
    return cmd


def run_one(algo: str, script: str, run_cfg: dict) -> dict:
    """Launch one training run as a subprocess and return its eval results.

    Deliberately a subprocess and not an import: one run that segfaults or runs
    out of memory must not take the other 39 down with it.
    """
    run_id = str(run_cfg["id"])
    cmd = build_cmd(algo, script, run_cfg)
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    started = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=RUN_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return {"id": run_id, "algo": algo, "status": "timeout", "note": run_cfg.get("note", "")}

    elapsed = round(time.time() - started, 1)
    if proc.returncode != 0:
        return {
            "id": run_id,
            "algo": algo,
            "status": "failed",
            "note": run_cfg.get("note", ""),
            "error": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown",
            "wall_s": elapsed,
        }

    config_path = REPO_ROOT / "logs" / run_id / "config.json"
    if not config_path.exists():
        return {"id": run_id, "algo": algo, "status": "no_config", "note": run_cfg.get("note", "")}

    result = json.loads(config_path.read_text())
    result.update(status="ok", note=run_cfg.get("note", ""), wall_s=elapsed)
    return result


def collect(results: list[dict]) -> None:
    """Write results.csv, the single source for analysis/tables.py."""
    if not results:
        print("nothing to collect")
        return
    ordered = sorted(results, key=lambda r: str(r.get("id", "")))
    fields: list[str] = []
    for row in ordered:
        for key in row:
            if key not in fields:
                fields.append(key)
    # Put the columns the report tables read first
    lead = ["id", "algo", "status", "note", "train_mean_return", "heldout_mean_return"]
    fields = [f for f in lead if f in fields] + [f for f in fields if f not in lead]

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    print(f"wrote {RESULTS_CSV} ({len(ordered)} rows)")


def run_sweep(algos: list[str], workers: int, only: list[str] | None, dry_run: bool) -> None:
    """Fan the selected runs across a ProcessPoolExecutor, then collect."""
    jobs: list[tuple[str, str, dict]] = []
    for algo in algos:
        grid = load_grid(algo)
        for run_cfg in grid["runs"]:
            if only and str(run_cfg["id"]) not in only:
                continue
            jobs.append((grid["algo"], grid["script"], run_cfg))

    if not jobs:
        print("no runs matched")
        return

    if dry_run:
        for algo, script, run_cfg in jobs:
            print(" ".join(build_cmd(algo, script, run_cfg)))
        print(f"\n{len(jobs)} runs, {workers} workers")
        return

    print(f"launching {len(jobs)} runs across {workers} workers")
    started = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, *job): job[2]["id"] for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            mark = "ok " if result.get("status") == "ok" else "FAIL"
            ret = result.get("heldout_mean_return")
            ret_s = f"{ret:7.2f}" if isinstance(ret, (int, float)) else "      -"
            mins = (time.time() - started) / 60
            print(f"[{done:2d}/{len(jobs)}] {mark} {result['id']:12s} heldout {ret_s}  ({mins:.1f} min)")

    collect(results)
    failed = [r["id"] for r in results if r.get("status") != "ok"]
    print(f"done in {(time.time() - started) / 60:.1f} min, {len(failed)} failed")
    if failed:
        print("failed runs:", ", ".join(failed))


def main() -> None:
    p = argparse.ArgumentParser(description="Run hyperparameter sweeps")
    p.add_argument("--algo", choices=("dqn", "reinforce", "ppo", "a2c"))
    p.add_argument("--all", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--only", nargs="+", help="run ids to run, defaults to all in the grid")
    p.add_argument("--dry-run", action="store_true", help="print commands and exit")
    args = p.parse_args()

    if not args.all and args.algo is None:
        p.error("pass --algo or --all")

    algos = ["dqn", "reinforce", "ppo", "a2c"] if args.all else [args.algo]
    run_sweep(algos, args.workers, args.only, args.dry_run)


if __name__ == "__main__":
    main()

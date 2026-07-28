"""Run the hyperparameter grids across processes.

    uv run experiments/sweep.py --all --workers 6
    uv run experiments/sweep.py --algo ppo --workers 4
    uv run experiments/sweep.py --algo dqn --only dqn-04 dqn-09

Roughly 35 to 50 minutes for all 40 runs at 6 workers on an 18 core machine.
Results land in logs/<run-id>/ and are collected into experiments/results.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "configs"
RESULTS_CSV = Path(__file__).parent / "results.csv"


def load_grid(algo: str) -> dict:
    """Read a config YAML and merge defaults into every run."""
    # TODO: yaml.safe_load, then {**defaults, **run} per entry
    raise NotImplementedError


def run_one(algo: str, run_cfg: dict) -> dict:
    """Launch one training run as a subprocess and return its eval results."""
    # TODO: build the CLI from run_cfg, subprocess.run, then read back
    # logs/<id>/config.json. Do not import the training modules in-process,
    # a crashed run must not take the sweep down with it.
    raise NotImplementedError


def collect(results: list[dict]) -> None:
    """Write results.csv, the single source for analysis/tables.py."""
    raise NotImplementedError


def run_sweep(algos: list[str], workers: int, only: list[str] | None, dry_run: bool) -> None:
    """Fan the selected runs across a ProcessPoolExecutor, then collect."""
    # TODO: expand grids, filter by `only`, print and exit when dry_run,
    # otherwise submit run_one across `workers` processes and collect results.
    raise NotImplementedError


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

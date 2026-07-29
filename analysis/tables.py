"""Build the four hyperparameter tables from disk.

Read from experiments/results.csv, never retyped by hand. Rows stay in grid
order so the reader can follow the experimental logic. Sorting by result hides
which variable was manipulated.

    uv run analysis/tables.py --out assets/tables
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

COLUMNS = {
    "dqn": [
        "lr", "gamma", "buffer_size", "batch_size", "exploration_fraction",
        "exploration_final_eps", "target_update_interval", "net_arch",
    ],
    "reinforce": [
        "lr", "gamma", "baseline", "ent_coef", "episodes_per_update",
        "normalise_advantage",
    ],
    "ppo": [
        "lr", "gamma", "n_steps", "batch_size", "n_epochs", "clip_range",
        "gae_lambda", "ent_coef",
    ],
    "a2c": [
        "lr", "gamma", "n_steps", "ent_coef", "vf_coef", "use_rms_prop",
        "max_grad_norm",
    ],
}

HEADERS = {
    "lr": "Learning rate",
    "gamma": r"$\gamma$",
    "buffer_size": "Replay buffer",
    "batch_size": "Batch size",
    "exploration_fraction": "Exploration frac",
    "exploration_final_eps": "Final " r"$\epsilon$",
    "target_update_interval": "Target update",
    "net_arch": "Network",
    "baseline": "Baseline",
    "ent_coef": "Entropy coef",
    "episodes_per_update": "Episodes/update",
    "normalise_advantage": "Norm. advantage",
    "n_steps": "Rollout steps",
    "n_epochs": "Epochs",
    "clip_range": "Clip range",
    "gae_lambda": r"GAE $\lambda$",
    "vf_coef": "Value coef",
    "use_rms_prop": "Optimiser",
    "max_grad_norm": "Grad clip",
}

ALGO_TITLES = {
    "dqn": "DQN",
    "reinforce": "REINFORCE",
    "ppo": "PPO",
    "a2c": "A2C",
}


def _fmt(val: str) -> str:
    """Format a cell value for display."""
    if val == "" or val == "None":
        return "---"
    try:
        f = float(val)
        if f == 0.0:
            return "0"
        if abs(f) < 0.001 or abs(f) >= 10000:
            return f"{f:.2e}"
        if abs(f) >= 1:
            return f"{f:.4g}"
        return f"{f:.5g}"
    except ValueError:
        pass
    if val.lower() in ("true", "false"):
        return val
    if val.startswith("[") and val.endswith("]"):
        return val.replace(" ", "")
    return val


def _optimiser_label(use_rms: str) -> str:
    if use_rms.lower() == "true":
        return "RMSProp"
    if use_rms.lower() == "false":
        return "Adam"
    return use_rms


def build_table(algo: str, results_csv: Path) -> str:
    rows = []
    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["algo"] == algo:
                rows.append(row)

    cols = COLUMNS[algo]
    header_cells = [HEADERS.get(c, c) for c in cols] + ["Mean reward (heldout)", "Notes"]
    header = "| " + " | ".join(header_cells) + " |"
    sep = "|" + "|".join("---" for _ in header_cells) + "|"

    lines = [header, sep]
    for row in rows:
        cells = []
        for c in cols:
            val = row.get(c, "")
            if c == "use_rms_prop":
                val = _optimiser_label(val)
            elif c == "net_arch":
                raw = row.get(c, "")
                if raw.startswith("["):
                    parts = raw.strip("[]").split(",")
                    val = "[" + ", ".join(p.strip() for p in parts) + "]"
                else:
                    val = raw
            else:
                val = _fmt(val)
            cells.append(val)
        heldout = f"{float(row['heldout_mean_return']):.2f}"
        cells.append(heldout)
        cells.append(row.get("note", ""))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate report tables")
    p.add_argument("--out", type=Path, default=Path("assets/tables"))
    p.add_argument("--results", type=Path, default=Path("experiments/results.csv"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for algo in ["dqn", "reinforce", "ppo", "a2c"]:
        md = build_table(algo, args.results)
        out_path = args.out / f"table_{algo}.md"
        out_path.write_text(md + "\n")
        print(f"wrote {out_path}")

    combined_path = args.out / "tables.md"
    parts = []
    for algo in ["dqn", "reinforce", "ppo", "a2c"]:
        parts.append(f"## {ALGO_TITLES[algo]}\n")
        parts.append(build_table(algo, args.results))
        parts.append("")
    combined_path.write_text("\n".join(parts) + "\n")
    print(f"wrote {combined_path}")


if __name__ == "__main__":
    main()

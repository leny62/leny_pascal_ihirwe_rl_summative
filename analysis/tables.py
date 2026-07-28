"""Build the four hyperparameter tables from disk.

Read from logs/*/config.json and experiments/results.csv, never retyped by hand.
Rows stay in grid order so the reader can follow the experimental logic. Sorting
by result hides which variable was manipulated.

    uv run analysis/tables.py --out assets/tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Column order per algorithm, matching experiments/configs/*.yaml and the report
# template. "Mean reward" and "Notes" are appended to each.
COLUMNS = {
    "dqn": ["lr", "gamma", "buffer_size", "batch_size", "exploration", "target_update_interval", "net_arch"],
    "reinforce": ["lr", "gamma", "baseline", "ent_coef", "episodes_per_update", "normalise_advantage"],
    "ppo": ["lr", "gamma", "n_steps", "batch_size", "n_epochs", "clip_range", "gae_lambda", "ent_coef"],
    "a2c": ["lr", "gamma", "n_steps", "ent_coef", "vf_coef", "optimiser", "max_grad_norm"],
}

HEADERS = {
    "lr": "Learning rate",
    "gamma": "Gamma",
    "buffer_size": "Replay buffer",
    "batch_size": "Batch size",
    "exploration": "Exploration",
    "target_update_interval": "Target update",
    "net_arch": "Network",
    "baseline": "Baseline",
    "ent_coef": "Entropy coef",
    "episodes_per_update": "Episodes/update",
    "normalise_advantage": "Norm. advantage",
    "n_steps": "n_steps",
    "n_epochs": "Epochs",
    "clip_range": "Clip range",
    "gae_lambda": "GAE lambda",
    "vf_coef": "Value coef",
    "optimiser": "Optimiser",
    "max_grad_norm": "Grad clip",
}


def build_table(algo: str, results_csv: Path) -> str:
    """Return one markdown table, ten rows, in grid order."""
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser(description="Generate report tables")
    p.add_argument("--out", type=Path, default=Path("assets/tables"))
    p.add_argument("--results", type=Path, default=Path("experiments/results.csv"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError


if __name__ == "__main__":
    main()

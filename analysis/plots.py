"""Generate every figure the report needs.

Figures 1 to 5 are all required by the report, so run --all before writing up.

    uv run analysis/plots.py --all --out assets/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Fixed across every figure. Inconsistent formatting between figures is called
# out in the rubric.
ALGO_COLOURS = {
    "dqn": "#1f77b4",
    "reinforce": "#d62728",
    "ppo": "#2ca02c",
    "a2c": "#ff7f0e",
}
ALGO_LABELS = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
ROLLING_WINDOW = 20
DPI = 150
UNIFORM_ENTROPY = 2.890  # ln(18), a uniform policy over the action space


def load_monitor(run_dir: Path):
    """Read a Monitor CSV into a DataFrame. Same schema for all four algorithms."""
    raise NotImplementedError


def fig1_cumulative_reward(out: Path) -> None:
    """2x2 subplots, one per algorithm, shared y limits.

    Overlay the random and scripted agronomist means as horizontal lines. Those
    turn the figure from "curves went up" into a claim worth making.
    """
    raise NotImplementedError


def fig2_dqn_objective(out: Path) -> None:
    """TD loss (log y), mean predicted Q against realised return, epsilon."""
    raise NotImplementedError


def fig3_pg_entropy(out: Path) -> None:
    """Policy entropy for the three PG methods, with the uniform line marked."""
    raise NotImplementedError


def fig4_convergence(out: Path) -> None:
    """Episodes to threshold as bars, plus return against step for all four.

    State the threshold definition in the caption. An unstated one is worthless.
    """
    raise NotImplementedError


def fig5_generalisation(out: Path) -> None:
    """Held-out seeds 10000 to 10099, boxplot against training distribution."""
    raise NotImplementedError


def fig6_policy_behaviour(out: Path) -> None:
    """One episode under the best policy. Not required, most persuasive figure."""
    raise NotImplementedError


def write_captions(out: Path) -> None:
    """Captions written alongside the figures so none reaches the report bare."""
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser(description="Generate report figures")
    p.add_argument("--out", type=Path, default=Path("assets/figures"))
    p.add_argument("--all", action="store_true")
    p.add_argument("--only", nargs="+", choices=[f"fig{i}" for i in range(1, 7)])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError


if __name__ == "__main__":
    main()

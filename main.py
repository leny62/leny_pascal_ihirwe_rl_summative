"""Entry point. Runs the best available policy in the 3D view.

    uv sync
    uv run main.py

Falls back to the scripted agronomist if no trained model is present, so a fresh
clone always shows something rather than a stack trace.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODEL_SEARCH_ORDER = [
    Path("models/pg/ppo_best.zip"),
    Path("models/pg/a2c_best.zip"),
    Path("models/pg/reinforce_best.pt"),
    Path("models/dqn/dqn_best.zip"),
]


def find_model() -> Path | None:
    return next((p for p in MODEL_SEARCH_ORDER if p.exists()), None)


def main() -> None:
    p = argparse.ArgumentParser(description="Run a season under the best trained policy")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()

    from play import run_episodes  # keeps main.py importable without a GL context

    model_path = find_model()
    if model_path is None:
        print("No trained model found, falling back to the scripted agronomist.")
        print("Train one with: uv run training/pg_training.py --algo ppo --run-name ppo-01")

    run_episodes(
        model_path=model_path,
        episodes=args.episodes,
        seed=args.seed,
        render=not args.no_render,
        verbose=True,
    )


if __name__ == "__main__":
    main()

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
    Path("models/ppo/ppo-06.zip"),
    Path("models/ppo/ppo-05.zip"),
    Path("models/ppo/ppo-01.zip"),
    Path("models/reinforce/rf-09.pt"),
    Path("models/dqn/dqn-07.zip"),
    Path("models/a2c/a2c-06.zip"),
]


def find_model() -> Path | None:
    return next((p for p in MODEL_SEARCH_ORDER if p.exists()), None)


def main() -> None:
    p = argparse.ArgumentParser(description="Run a season under the best trained policy")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-render", action="store_true")
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="simulated days per second when rendering; lower is easier to narrate",
    )
    args = p.parse_args()

    from play import DEFAULT_RENDER_FPS, run_episodes

    common = {
        "episodes": args.episodes,
        "seed": args.seed,
        "render": not args.no_render,
        "verbose": True,
        "fps": args.fps if args.fps is not None else DEFAULT_RENDER_FPS,
    }

    model_path = find_model()
    if model_path is None:
        print("No trained model found, falling back to the scripted agronomist.")
        print("Train one with: uv run training/pg_training.py --algo ppo --run-name ppo-01")
        run_episodes(baseline="scripted", **common)
    else:
        run_episodes(model_path=model_path, **common)


if __name__ == "__main__":
    main()

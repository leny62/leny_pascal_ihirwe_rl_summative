"""Watch a policy work a season.

    uv run play.py --model models/pg/ppo_best.zip --episodes 2 --verbose
    uv run play.py --baseline scripted --episodes 50 --no-render

--verbose prints one line per day. The video needs visible terminal output
alongside the GUI, so keep it on when recording.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_policy(model_path: Path | None, baseline: str | None):
    """Load an SB3 zip, a REINFORCE checkpoint, or a scripted baseline."""
    # TODO: dispatch on suffix and on the baseline flag. REINFORCE saves a .pt
    # state dict, the SB3 algorithms save .zip.
    raise NotImplementedError


def run_episodes(
    model_path: Path | None = None,
    baseline: str | None = None,
    episodes: int = 1,
    seed: int | None = None,
    render: bool = True,
    verbose: bool = False,
) -> list[float]:
    """Roll out and return the episodic returns."""
    # TODO: make the env with render_mode="human" when render, step the policy,
    # print the per-day line when verbose, print the summary and the termination
    # cause at the end of each episode.
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser(description="Run a trained policy on Umurima-v0")
    p.add_argument("--model", type=Path)
    p.add_argument("--baseline", choices=("random", "scripted"))
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int)
    p.add_argument("--no-render", dest="render", action="store_false")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.model is None and args.baseline is None:
        p.error("pass --model or --baseline")

    returns = run_episodes(
        model_path=args.model,
        baseline=args.baseline,
        episodes=args.episodes,
        seed=args.seed,
        render=args.render,
        verbose=args.verbose,
    )
    print(f"mean return over {len(returns)} episodes: {sum(returns) / len(returns):.2f}")


if __name__ == "__main__":
    main()

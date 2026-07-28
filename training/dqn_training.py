"""DQN training. Grid: experiments/configs/dqn.yaml.

Every hyperparameter is a CLI flag so the sweep and a manual run go through the
same code path.

    uv run training/dqn_training.py --run-name dqn-01 --lr 1e-4 --gamma 0.99
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DQN on Umurima-v0")
    p.add_argument("--run-name", required=True)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--target-update-interval", type=int, default=1_000)
    p.add_argument("--exploration-fraction", type=float, default=0.20)
    p.add_argument("--exploration-final-eps", type=float, default=0.05)
    p.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    p.add_argument("--timesteps", type=int, default=400_000)
    p.add_argument("--seed", type=int, default=0)
    return p


def train(args: argparse.Namespace) -> dict:
    """Train, evaluate on train and held-out seeds, save the model and config."""
    # TODO: build env, construct DQN with MlpPolicy and args.net_arch, attach the
    # eval callback, learn, save to models/dqn/<run-name>.zip, dump config.json
    # with the eval results merged in.
    raise NotImplementedError


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()

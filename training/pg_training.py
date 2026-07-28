"""Policy gradient training: REINFORCE, PPO and A2C.

PPO and A2C come from Stable-Baselines3. REINFORCE is in training/reinforce.py
because SB3 does not ship one. Grids: experiments/configs/.

    uv run training/pg_training.py --algo ppo --run-name ppo-01 --lr 3e-4
"""

from __future__ import annotations

import argparse

ALGOS = ("reinforce", "ppo", "a2c")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a policy gradient method on Umurima-v0")
    p.add_argument("--algo", choices=ALGOS, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    p.add_argument("--timesteps", type=int, default=400_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=8)

    # PPO and A2C
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--use-rms-prop", action="store_true", default=True)
    p.add_argument("--adam", dest="use_rms_prop", action="store_false")

    # REINFORCE
    p.add_argument("--baseline", choices=("none", "value"), default="value")
    p.add_argument("--episodes-per-update", type=int, default=1)
    p.add_argument("--normalise-advantage", action="store_true")
    return p


def train(args: argparse.Namespace) -> dict:
    """Dispatch to the right trainer, then share the eval and logging path."""
    # TODO: reinforce -> training.reinforce.train_reinforce, otherwise construct
    # the SB3 model. All three write the same Monitor CSV schema and config.json.
    raise NotImplementedError


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()

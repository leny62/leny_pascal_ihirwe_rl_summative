"""DQN training. Grid: experiments/configs/dqn.yaml.

Every hyperparameter is a CLI flag so the sweep and a manual run go through the
same code path.

    uv run training/dqn_training.py --run-name dqn-01 --lr 1e-4 --gamma 0.99
"""

from __future__ import annotations

import argparse

from stable_baselines3 import DQN

from training.common import (
    LOG_ROOT,
    MODEL_ROOT,
    dump_config,
    evaluate_both,
    limit_threads,
    make_vec_env,
)


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
    limit_threads()
    run_dir = LOG_ROOT / args.run_name
    # One env: DQN is off-policy and replays from the buffer, so parallel
    # collection buys little and changes the gradient-steps-per-sample ratio.
    env = make_vec_env(1, args.seed, str(run_dir))

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs={"net_arch": list(args.net_arch)},
        seed=args.seed,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    model_path = MODEL_ROOT / "dqn" / f"{args.run_name}.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    env.close()

    config = {
        "id": args.run_name,
        "algo": "dqn",
        "lr": args.lr,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "learning_starts": args.learning_starts,
        "train_freq": args.train_freq,
        "target_update_interval": args.target_update_interval,
        "exploration_fraction": args.exploration_fraction,
        "exploration_final_eps": args.exploration_final_eps,
        "net_arch": list(args.net_arch),
        "timesteps": args.timesteps,
        "seed": args.seed,
        **evaluate_both(model),
    }
    dump_config(run_dir, config)
    return config


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()

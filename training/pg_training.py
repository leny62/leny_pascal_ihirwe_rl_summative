"""Policy gradient training: REINFORCE, PPO and A2C.

PPO and A2C come from Stable-Baselines3. REINFORCE is in training/reinforce.py
because SB3 does not ship one. Grids: experiments/configs/.

    uv run training/pg_training.py --algo ppo --run-name ppo-01 --lr 3e-4
"""

from __future__ import annotations

import argparse

from stable_baselines3 import A2C, PPO

from training.common import (
    LOG_ROOT,
    MODEL_ROOT,
    dump_config,
    evaluate_both,
    limit_threads,
    make_env,
    make_vec_env,
)
from training.reinforce import ReinforceConfig, train_reinforce

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


def _build_sb3(args: argparse.Namespace, env):
    """PPO and A2C share a policy and differ only in the update."""
    policy_kwargs = {"net_arch": list(args.net_arch)}
    if args.algo == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            gamma=args.gamma,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            clip_range=args.clip_range,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            verbose=0,
            device="cpu",
        )
    return A2C(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        gamma=args.gamma,
        n_steps=args.n_steps,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        use_rms_prop=args.use_rms_prop,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        verbose=0,
        device="cpu",
    )


def train(args: argparse.Namespace) -> dict:
    """Dispatch to the right trainer, then share the eval and logging path."""
    limit_threads()
    run_dir = LOG_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config: dict = {
        "id": args.run_name,
        "algo": args.algo,
        "lr": args.lr,
        "gamma": args.gamma,
        "ent_coef": args.ent_coef,
        "net_arch": list(args.net_arch),
        "timesteps": args.timesteps,
        "seed": args.seed,
    }

    if args.algo == "reinforce":
        # Monte Carlo updates need whole episodes, so this one runs unvectorised.
        env = make_env(seed=args.seed, monitor_dir=str(run_dir))()
        cfg = ReinforceConfig(
            lr=args.lr,
            gamma=args.gamma,
            baseline=args.baseline,
            ent_coef=args.ent_coef,
            episodes_per_update=args.episodes_per_update,
            normalise_advantage=args.normalise_advantage,
            net_arch=tuple(args.net_arch),
            seed=args.seed,
        )
        model = train_reinforce(env, cfg, args.timesteps, run_dir)
        model_path = MODEL_ROOT / "reinforce" / f"{args.run_name}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        config.update(
            baseline=args.baseline,
            episodes_per_update=args.episodes_per_update,
            normalise_advantage=args.normalise_advantage,
            n_envs=1,
        )
    else:
        env = make_vec_env(args.n_envs, args.seed, str(run_dir))
        model = _build_sb3(args, env)
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        model_path = MODEL_ROOT / args.algo / f"{args.run_name}.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        config.update(
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            gae_lambda=args.gae_lambda,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
        )
        if args.algo == "ppo":
            config.update(
                batch_size=args.batch_size, n_epochs=args.n_epochs, clip_range=args.clip_range
            )
        else:
            config.update(use_rms_prop=args.use_rms_prop)

    config.update(evaluate_both(model))
    env.close()
    dump_config(run_dir, config)
    return config


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()

"""Shared training plumbing.

Every algorithm goes through here so the comparison stays fair: same env, same
network shape, same logging schema, same eval protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import environment  # noqa: F401  registers Umurima-v0

ENV_ID = "Umurima-v0"
NET_ARCH = [256, 256]
TOTAL_TIMESTEPS = 400_000

TRAIN_SEED_RANGE = (0, 1000)
HELDOUT_SEED_RANGE = (10_000, 10_100)
EVAL_EPISODES = 30
GENERALISATION_EPISODES = 100

LOG_ROOT = Path("logs")
MODEL_ROOT = Path("models")


def limit_threads() -> None:
    """One torch thread per process.

    The sweep already saturates the machine with worker processes. Letting each
    one also fan out across cores turns the sweep into cache thrashing.
    """
    torch.set_num_threads(1)


def make_env(seed: int | None = None, render_mode: str | None = None, monitor_dir: str | None = None):
    """Factory used by every training script and by the sweep."""

    def _init() -> gym.Env:
        env = gym.make(ENV_ID, render_mode=render_mode)
        if monitor_dir is not None:
            env = Monitor(env, filename=str(Path(monitor_dir) / "monitor"))
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _init


def make_vec_env(n_envs: int, seed: int, monitor_dir: str):
    """Always DummyVecEnv, never SubprocVecEnv. Never pass a render_mode here.

    A step of this env is a few hundred numpy operations, so worker IPC costs
    more than it saves. The sweep gets its parallelism from running whole runs
    side by side instead, which also keeps one crashed run from killing others.
    """
    Path(monitor_dir).mkdir(parents=True, exist_ok=True)
    fns = [
        make_env(seed=seed + i, monitor_dir=str(Path(monitor_dir) / str(i)))
        for i in range(n_envs)
    ]
    return DummyVecEnv(fns)


def dump_config(run_dir: Path, config: dict[str, Any]) -> None:
    """Write config.json beside the logs. analysis/tables.py reads these."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))


def train_eval_seeds(n: int = EVAL_EPISODES) -> range:
    """Seeds drawn from the training distribution."""
    return range(TRAIN_SEED_RANGE[0], TRAIN_SEED_RANGE[0] + n)


def heldout_seeds(n: int = GENERALISATION_EPISODES) -> range:
    """Seeds never touched during training, so the gap means something."""
    return range(HELDOUT_SEED_RANGE[0], HELDOUT_SEED_RANGE[0] + n)


def evaluate(model, seeds: range, deterministic: bool = True) -> dict[str, float]:
    """Roll the policy over fixed seeds. Used for both table and held-out numbers."""
    env = gym.make(ENV_ID)
    returns: list[float] = []
    lengths: list[int] = []
    causes: dict[str, int] = {}

    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        total, steps = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += float(reward)
            steps += 1
            if terminated or truncated:
                cause = info.get("termination_cause") or "truncated"
                causes[cause] = causes.get(cause, 0) + 1
                break
        returns.append(total)
        lengths.append(steps)
    env.close()

    arr = np.asarray(returns, dtype=np.float64)
    n = max(len(returns), 1)
    out = {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
        "mean_length": float(np.mean(lengths)),
        "n_episodes": float(len(returns)),
    }
    for cause in ("harvested", "crop_failure", "insolvency", "truncated"):
        out[f"frac_{cause}"] = causes.get(cause, 0) / n
    return out


def evaluate_both(model) -> dict[str, Any]:
    """Train-distribution and held-out numbers in the schema results.csv expects."""
    train = evaluate(model, train_eval_seeds())
    heldout = evaluate(model, heldout_seeds())
    return {
        **{f"train_{k}": v for k, v in train.items()},
        **{f"heldout_{k}": v for k, v in heldout.items()},
        "generalisation_gap": train["mean_return"] - heldout["mean_return"],
    }

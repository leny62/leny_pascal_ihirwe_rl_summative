"""Shared training plumbing.

Every algorithm goes through here so the comparison stays fair: same env, same
network shape, same logging schema, same eval protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

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
    """SubprocVecEnv above 1, DummyVecEnv at 1. Never pass a render_mode here."""
    # TODO: build the vec env, seed each worker from seed + index
    raise NotImplementedError


def dump_config(run_dir: Path, config: dict[str, Any]) -> None:
    """Write config.json beside the logs. analysis/tables.py reads these."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))


def evaluate(model, seeds: range, deterministic: bool = True) -> dict[str, float]:
    """Roll the policy over fixed seeds. Used for both table and held-out numbers."""
    # TODO: return mean, std, min, max return plus mean episode length and the
    # distribution of termination causes
    raise NotImplementedError

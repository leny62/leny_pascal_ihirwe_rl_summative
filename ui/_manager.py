"""Episode lifecycle management for the Streamlit dashboard.

Single-responsibility: owns one UmurimaEnv and its policy, exposes a narrow
interface for the UI layer to consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym

import environment  # noqa: F401  registers Umurima-v0


@dataclass(frozen=True)
class StepRecord:
    day: int
    action: str
    reward: float
    cash: float


class EpisodeManager:
    """Wraps an Umurima episode so the UI never touches the gym API directly."""

    def __init__(self, seed: int = 0, policy_spec: str = "scripted") -> None:
        self._env = gym.make("Umurima-v0")
        self._policy = _build_policy(policy_spec)
        self._obs, _ = self._env.reset(seed=seed)
        self._done = False
        self._history: list[StepRecord] = []

    # ------------------------------------------------------------------
    # commands
    # ------------------------------------------------------------------

    def step(self, days: int = 1) -> list[StepRecord]:
        records: list[StepRecord] = []
        for _ in range(days):
            if self._done:
                break
            action, _ = self._policy.predict(self._obs, deterministic=True)
            self._obs, reward, terminated, truncated, info = self._env.step(int(action))
            self._done = terminated or truncated
            record = StepRecord(
                day=info.get("day", 0),
                action=_action_name(int(action)),
                reward=round(float(reward), 4),
                cash=round(float(info.get("cash_krwf", 0)), 2),
            )
            self._history.append(record)
            records.append(record)
        return records

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        return self._env.unwrapped._render_state()

    @property
    def ledger(self) -> dict[str, Any]:
        return json.loads(self._env.unwrapped.ledger_json())

    @property
    def done(self) -> bool:
        return self._done

    @property
    def termination_cause(self) -> str | None:
        if not self._done:
            return None
        return self.state.get("action", "truncated") if self._done else None

    @property
    def history(self) -> list[StepRecord]:
        return list(self._history)

    @property
    def ledger_verified(self) -> bool:
        return self._env.unwrapped._ledger.verify()


# ---------------------------------------------------------------------------
# policy dispatch
# ---------------------------------------------------------------------------


def _build_policy(spec: str):
    if spec == "random":
        from training.baselines import RandomPolicy

        return RandomPolicy(gym.make("Umurima-v0").action_space)
    if spec == "scripted":
        from training.baselines import ScriptedAgronomist

        return ScriptedAgronomist()

    path = Path(spec)
    suffix = path.suffix.lower()
    if suffix == ".zip":
        algo = path.parent.name
        if algo == "dqn":
            from stable_baselines3 import DQN

            return DQN.load(str(path))
        if algo == "ppo":
            from stable_baselines3 import PPO

            return PPO.load(str(path))
        if algo == "a2c":
            from stable_baselines3 import A2C

            return A2C.load(str(path))
        raise ValueError(f"unknown algorithm for {path}")
    if suffix == ".pt":
        from training.reinforce import ReinforceAgent

        return ReinforceAgent.load(str(path))
    raise ValueError(f"unknown policy: {spec}")


def _action_name(action: int) -> str:
    from environment.custom_env import Action

    try:
        return Action(action).name
    except ValueError:
        return f"UNKNOWN({action})"

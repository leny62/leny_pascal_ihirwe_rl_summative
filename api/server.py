"""Read-only JSON API over a running episode.

Shows the environment state and the block ledger as something a web or mobile
frontend could consume. Same serialisation the renderer uses.

    uv run --extra api uvicorn api.server:app --reload
"""

from __future__ import annotations

import uuid
from typing import Any

import gymnasium as gym

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:
    raise SystemExit("Install the api extra: uv sync --extra api") from exc

import environment  # noqa: F401  registers Umurima-v0

app = FastAPI(title="Umurima block API", version="0.1.0")

_episodes: dict[str, _Episode] = {}

DEFAULT_POLICY = "scripted"
DEFAULT_SEED = 0


class _Episode:
    __slots__ = ("env", "policy", "obs", "done")

    def __init__(self, env: gym.Env, policy: Any, obs: Any) -> None:
        self.env = env
        self.policy = policy
        self.obs = obs
        self.done = False


def _load_policy(policy_spec: str):
    from play import load_policy

    if policy_spec in ("random", "scripted"):
        return load_policy(None, policy_spec)
    from pathlib import Path

    path = Path(policy_spec)
    if not path.exists():
        raise HTTPException(400, f"model not found: {policy_spec}")
    return load_policy(path, None)


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/episode")
def create_episode(seed: int = DEFAULT_SEED, policy: str = DEFAULT_POLICY) -> dict[str, Any]:
    episode_id = uuid.uuid4().hex[:12]
    env = gym.make("Umurima-v0")
    obs, _ = env.reset(seed=seed)
    _episodes[episode_id] = _Episode(env, _load_policy(policy), obs)
    return {
        "episode_id": episode_id,
        "seed": seed,
        "policy": policy,
        "state": _state_dict(env, episode_id),
    }


@app.get("/episode/{episode_id}/state")
def get_state(episode_id: str) -> dict[str, Any]:
    ep = _episodes.get(episode_id)
    if ep is None:
        raise HTTPException(404, f"episode not found: {episode_id}")
    return _state_dict(ep.env, episode_id)


@app.post("/episode/{episode_id}/advance")
def advance(episode_id: str, days: int = 1) -> dict[str, Any]:
    if days < 1:
        raise HTTPException(400, "days must be >= 1")
    ep = _episodes.get(episode_id)
    if ep is None:
        raise HTTPException(404, f"episode not found: {episode_id}")
    if ep.done:
        raise HTTPException(400, "episode already finished")

    steps: list[dict[str, Any]] = []
    for _ in range(days):
        if ep.done:
            break
        action, _ = ep.policy.predict(ep.obs, deterministic=True)
        ep.obs, reward, terminated, truncated, info = ep.env.step(int(action))
        ep.done = terminated or truncated
        steps.append(
            {
                "day": info.get("day", 0),
                "action": int(action),
                "reward": round(float(reward), 4),
                "cash_krwf": round(float(info.get("cash_krwf", 0)), 2),
            }
        )

    return {
        "episode_id": episode_id,
        "steps": steps,
        "done": ep.done,
        "termination_cause": info.get("termination_cause") if ep.done else None,
        "state": _state_dict(ep.env, episode_id),
    }


@app.get("/episode/{episode_id}/ledger")
def get_ledger(episode_id: str) -> dict[str, Any]:
    ep = _episodes.get(episode_id)
    if ep is None:
        raise HTTPException(404, f"episode not found: {episode_id}")
    import json

    return json.loads(ep.env.unwrapped.ledger_json())


@app.get("/episode/{episode_id}/ledger/verify")
def verify_ledger(episode_id: str) -> dict[str, Any]:
    ep = _episodes.get(episode_id)
    if ep is None:
        raise HTTPException(404, f"episode not found: {episode_id}")
    env = ep.env.unwrapped
    verified = env._ledger.verify()
    return {"episode_id": episode_id, "verified": verified, "head": env._ledger.head}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _state_dict(env: gym.Env, episode_id: str) -> dict[str, Any]:
    unwrapped = env.unwrapped
    return {
        "episode_id": episode_id,
        **unwrapped._render_state(),
    }

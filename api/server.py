"""Read-only JSON API over a running episode.

Shows the environment state and the block ledger as something a web or mobile
frontend could consume. Same serialisation the renderer uses.

    uv run --extra api uvicorn api.server:app --reload
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI
except ImportError as exc:  # optional extra
    raise SystemExit("Install the api extra: uv sync --extra api") from exc

app = FastAPI(title="Umurima block API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/episode/{episode_id}/state")
def get_state(episode_id: str) -> dict[str, Any]:
    """Current block state: per-zone soil, canopy, nitrogen, pest and weather."""
    raise NotImplementedError


@app.post("/episode")
def create_episode(seed: int | None = None, policy: str = "ppo_best") -> dict[str, Any]:
    """Start an episode under a saved policy and return its id."""
    raise NotImplementedError


@app.post("/episode/{episode_id}/advance")
def advance(episode_id: str, days: int = 1) -> dict[str, Any]:
    """Step the episode forward and return the new state plus actions taken."""
    raise NotImplementedError


@app.get("/episode/{episode_id}/ledger")
def get_ledger(episode_id: str) -> dict[str, Any]:
    """Hash-chained field event record. The audit view."""
    raise NotImplementedError


@app.get("/episode/{episode_id}/ledger/verify")
def verify_ledger(episode_id: str) -> dict[str, bool]:
    """Recompute the chain. False means a record was altered after the fact."""
    raise NotImplementedError

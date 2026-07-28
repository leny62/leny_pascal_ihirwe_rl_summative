"""Append-only event ledger for a production block.

Records every field operation with the hash of the record before it, so an
altered entry breaks verification for itself and everything after it. This is
the record a GlobalG.A.P. audit asks for and that most blocks still keep on
paper.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

GENESIS = "0" * 64


def _digest(payload: dict[str, Any]) -> str:
    """Hash a record. Sorted keys and no whitespace so it is reproducible."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class FieldEvent:
    day: int
    action: str
    zone: int | None = None
    quantity: float | None = None
    unit: str | None = None
    note: str | None = None


@dataclass
class BlockLedger:
    """Hash-chained log of one block for one season."""

    block_id: str
    season: str
    crop: str = "french_bean"
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def head(self) -> str:
        return self.events[-1]["hash"] if self.events else GENESIS

    def append(self, event: FieldEvent) -> str:
        """Chain an event onto the ledger and return its hash."""
        payload = {k: v for k, v in asdict(event).items() if v is not None}
        payload["prev_hash"] = self.head
        entry = dict(payload)
        entry["hash"] = _digest(payload)
        self.events.append(entry)
        return entry["hash"]

    def verify(self) -> bool:
        """Walk the chain and recompute every hash."""
        prev = GENESIS
        for entry in self.events:
            payload = {k: v for k, v in entry.items() if k != "hash"}
            if payload.get("prev_hash") != prev or _digest(payload) != entry["hash"]:
                return False
            prev = entry["hash"]
        return True

    def totals(self) -> dict[str, float]:
        """Season aggregates for the audit record."""
        totals = {
            "water_mm": 0.0,
            "nitrogen_kg_ha": 0.0,
            "potash_kg_ha": 0.0,
            "sprays": 0,
            "weeding_events": 0,
            "harvest_revenue_krwf": 0.0,
        }
        for event in self.events:
            action, qty = event.get("action"), event.get("quantity", 0.0) or 0.0
            if action == "irrigate":
                totals["water_mm"] += qty
            elif action == "apply_n":
                totals["nitrogen_kg_ha"] += qty
            elif action == "apply_k":
                totals["potash_kg_ha"] += qty
            elif action == "spray_biopesticide":
                totals["sprays"] += 1
            elif action == "hire_weeding_crew":
                totals["weeding_events"] += 1
            elif action == "harvest":
                totals["harvest_revenue_krwf"] += qty
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "season": self.season,
            "crop": self.crop,
            "events": self.events,
            "totals": self.totals(),
            "head": self.head,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

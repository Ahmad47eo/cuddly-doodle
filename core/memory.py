"""Bounded memory system for the platform.

Memory tiers:

* SHORT_TERM — recent observations/events
* EPISODIC — previous simulated episodes (summarized)
* SKILL — per-skill performance statistics
* STRATEGY — historical strategy results

All tiers have hard limits (entry count + byte budget) with LRU-style
eviction. Memory is serializable for optional persistence. LOW_RAM_MODE
uses much smaller limits.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from core.errors import MemoryError
from core.types import (
    EntityState,
    EpisodeResult,
    MemoryEntry,
    MemoryTier,
    Observation,
)


# ---------------------------------------------------------------------------
# Default tier limits
# ---------------------------------------------------------------------------

DEFAULT_LIMITS: dict[str, dict[str, int | float]] = {
    "short_term": {"max_entries": 64, "max_bytes": 2_000_000, "evict_fraction": 0.25},
    "episodic": {"max_entries": 128, "max_bytes": 8_000_000, "evict_fraction": 0.20},
    "skill": {"max_entries": 256, "max_bytes": 4_000_000, "evict_fraction": 0.20},
    "strategy": {"max_entries": 256, "max_bytes": 4_000_000, "evict_fraction": 0.20},
}

DEFAULT_LOW_RAM_LIMITS: dict[str, dict[str, int | float]] = {
    "short_term": {"max_entries": 16, "max_bytes": 512_000, "evict_fraction": 0.50},
    "episodic": {"max_entries": 32, "max_bytes": 1_000_000, "evict_fraction": 0.40},
    "skill": {"max_entries": 48, "max_bytes": 1_000_000, "evict_fraction": 0.40},
    "strategy": {"max_entries": 48, "max_bytes": 1_000_000, "evict_fraction": 0.40},
}


# ---------------------------------------------------------------------------
# Tier state
# ---------------------------------------------------------------------------

@dataclass
class TierState:
    """Internal state for one memory tier."""

    name: str
    max_entries: int
    max_bytes: int
    entries: list[MemoryEntry] = field(default_factory=list)
    current_bytes: int = 0
    access_order: list[str] = field(default_factory=list)
    entry_index: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Memory manager
# ---------------------------------------------------------------------------

class MemoryManager:
    """Bounded in-memory store for the memory tiers."""

    @classmethod
    def with_defaults(cls) -> "MemoryManager":
        return cls(limits=None)

    @classmethod
    def with_low_ram_limits(cls) -> "MemoryManager":
        return cls(limits=DEFAULT_LOW_RAM_LIMITS)

    def __init__(self, *, limits: Optional[dict[str, dict[str, int | float]]] = None) -> None:
        self._tiers: dict[str, TierState] = {}
        self._limits = limits or {}
        source = limits if limits is not None else DEFAULT_LIMITS
        for tier_name, tier_limits in source.items():
            self._create_tier_internal(tier_name, dict(tier_limits))

    def create_tier(self, name: str, *, max_entries: int = 64, max_bytes: int = 2_000_000) -> None:
        if name in self._tiers:
            raise MemoryError(f"tier '{name}' already exists", context={"tier": name})
        self._create_tier_internal(name, {"max_entries": max_entries, "max_bytes": max_bytes})

    def _create_tier_internal(self, name: str, limits: dict[str, int | float]) -> None:
        self._tiers[name] = TierState(
            name=name,
            max_entries=int(limits.get("max_entries", 64)),
            max_bytes=int(limits.get("max_bytes", 2_000_000)),
        )

    def tier(self, name: str) -> TierState:
        if name not in self._tiers:
            raise MemoryError(f"tier '{name}' does not exist", context={"tier": name})
        return self._tiers[name]

    def list_tiers(self) -> list[str]:
        return list(self._tiers.keys())

    def limits(self) -> dict[str, dict[str, int | float]]:
        out: dict[str, dict[str, int | float]] = {}
        for name, ts in self._tiers.items():
            # Find the evict_fraction from the limits that created this tier.
            evict = 0.25
            for lname, lcfg in self._limits.items():
                if lname == name and "evict_fraction" in lcfg:
                    evict = float(lcfg["evict_fraction"])
                    break
            out[name] = {
                "max_entries": ts.max_entries,
                "max_bytes": ts.max_bytes,
                "evict_fraction": evict,
                "current_entries": len(ts.entries),
                "current_bytes": ts.current_bytes,
            }
        return out

    def add(
        self,
        tier_name: str,
        key: str,
        payload: Mapping[str, Any],
        *,
        priority: float = 0.0,
    ) -> MemoryEntry:
        ts = self.tier(tier_name)
        entry = MemoryEntry(
            tier=MemoryTier[tier_name.upper()],
            key=key,
            payload=dict(payload),
            priority=priority,
        )
        self._insert_entry(ts, entry)
        self._enforce_limits(ts)
        return entry

    def update(self, tier_name: str, key: str, payload: Mapping[str, Any]) -> None:
        ts = self.tier(tier_name)
        idx = ts.entry_index.get(key)
        if idx is None:
            raise MemoryError(f"entry '{key}' not found in tier '{tier_name}'", context={"tier": tier_name, "key": key})
        entry = ts.entries[idx]
        entry.payload = dict(payload)
        self._touch_entry(ts, entry)

    def get(
        self, tier_name: str, key: str
    ) -> tuple[Optional[MemoryEntry], Optional[dict[str, Any]]]:
        if tier_name not in self._tiers:
            return None, None
        ts = self._tiers[tier_name]
        idx = ts.entry_index.get(key)
        if idx is None:
            return None, None
        entry = ts.entries[idx]
        meta = {"age": entry.age, "priority": entry.priority}
        self._touch_entry(ts, entry)
        return entry, meta

    def list(self, tier_name: str) -> list[MemoryEntry]:
        ts = self.tier(tier_name)
        return list(ts.entries)

    def delete(self, tier_name: str, key: str) -> bool:
        ts = self.tier(tier_name)
        idx = ts.entry_index.get(key)
        if idx is None:
            return False
        entry = ts.entries.pop(idx)
        del ts.entry_index[key]
        ts.access_order.remove(key)
        ts.current_bytes -= entry.size_bytes
        return True

    def clear(self, tier_name: str) -> int:
        ts = self.tier(tier_name)
        count = len(ts.entries)
        ts.entries.clear()
        ts.current_bytes = 0
        ts.access_order.clear()
        ts.entry_index.clear()
        return count

    def tick(self, *, age_delta: float = 1.0) -> None:
        for ts in self._tiers.values():
            for entry in ts.entries:
                entry.age += age_delta

    def stats(self, tier_name: str) -> dict[str, Any]:
        ts = self.tier(tier_name)
        return {
            "tier": tier_name,
            "count": len(ts.entries),
            "bytes_used": ts.current_bytes,
            "max_entries": ts.max_entries,
            "max_bytes": ts.max_bytes,
        }

    def _enforce_limits(self, ts: TierState) -> None:
        max_bytes = ts.max_bytes
        max_entries = ts.max_entries
        # Evict until under entry limit first.
        while len(ts.entries) >= max_entries:
            evicted = self._evict_one(ts)
            if evicted is None:
                break
        # Then enforce byte budget.
        while ts.current_bytes > max_bytes and ts.entries:
            evicted = self._evict_one(ts)
            if evicted is None:
                break

    def _evict_one(self, ts: TierState) -> Optional[MemoryEntry]:
        # LRU-ish: evict least-accessed entries first, weighted by priority.
        if not ts.entries:
            return None
        # Choose the least-recently-accessed entry with lowest priority.
        ordered = sorted(
            ts.entries,
            key=lambda e: (ts.access_order.index(e.key) if e.key in ts.access_order else len(ts.access_order), e.priority),
        )
        victim = ordered[0]
        self.delete(ts.name, victim.key)
        return victim

    def _insert_entry(self, ts: TierState, entry: MemoryEntry) -> None:
        ts.entries.append(entry)
        ts.entry_index[entry.key] = len(ts.entries) - 1
        ts.access_order.append(entry.key)
        ts.current_bytes += entry.size_bytes

    def _touch_entry(self, ts: TierState, entry: MemoryEntry) -> None:
        if entry.key in ts.access_order:
            ts.access_order.remove(entry.key)
        ts.access_order.append(entry.key)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def serialize_entry(entry: Optional[MemoryEntry]) -> Optional[dict[str, Any]]:
    if entry is None:
        return None
    return {
        "tier": entry.tier.value,
        "key": entry.key,
        "payload": entry.payload,
        "age": entry.age,
        "priority": entry.priority,
    }


def deserialize_entry(data: Optional[dict[str, Any]]) -> Optional[MemoryEntry]:
    if data is None:
        return None
    try:
        return MemoryEntry(
            tier=MemoryTier(data["tier"]),
            key=data["key"],
            payload=data.get("payload", {}),
            age=float(data.get("age", 0.0)),
            priority=float(data.get("priority", 0.0)),
        )
    except Exception as e:
        raise MemoryError(f"Failed to deserialize memory entry: {e}", context={"data_keys": list(data.keys())})


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def short_term_entry(key: str, payload: Mapping[str, Any], *, priority: float = 0.0) -> MemoryEntry:
    return MemoryEntry(
        tier=MemoryTier.SHORT_TERM,
        key=key,
        payload=dict(payload),
        priority=priority,
    )


class MemoryStats:
    """Statistics for a single memory tier (TypedDict-like)."""

    tier: str
    count: int
    bytes_used: int
    max_entries: int
    max_bytes: int

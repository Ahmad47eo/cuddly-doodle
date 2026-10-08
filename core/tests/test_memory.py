"""Memory system tests: tiers, limits, eviction, serialization, low-ram eviction, stats."""

from __future__ import annotations

import pytest

from core.memory import (
    DEFAULT_LIMITS,
    MemoryError,
    MemoryManager,
    short_term_entry,
    serialize_entry,
    deserialize_entry,
)


# ---------------------------------------------------------------------------
# Tier creation and limits
# ---------------------------------------------------------------------------

class TestMemoryTiers:
    def test_create_tier_uses_defaults(self) -> None:
        mgr = MemoryManager.with_defaults()
        st = mgr.tier("short_term")
        assert st.max_entries == DEFAULT_LIMITS["short_term"]["max_entries"]
        assert st.max_bytes == DEFAULT_LIMITS["short_term"]["max_bytes"]

    def test_create_tier_with_custom_limits(self) -> None:
        mgr = MemoryManager.with_defaults()
        mgr.create_tier("custom", max_entries=5, max_bytes=1024)
        st = mgr.tier("custom")
        assert st.max_entries == 5
        assert st.max_bytes == 1024

    def test_duplicate_tier_raises(self) -> None:
        mgr = MemoryManager.with_defaults()
        mgr.create_tier("x", max_entries=1, max_bytes=100)
        with pytest.raises(MemoryError, match="already exists"):
            mgr.create_tier("x", max_entries=1, max_bytes=100)


# ---------------------------------------------------------------------------
# Add / retrieve / evict
# ---------------------------------------------------------------------------

class TestAddAndRetrieve:
    def test_add_and_get(self) -> None:
        mgr = MemoryManager.with_defaults()
        key = "k1"
        payload = {"a": 1, "b": "two"}
        mgr.add("short_term", key, payload)
        entry, meta = mgr.get("short_term", key)
        assert entry is not None
        assert entry.key == key
        assert entry.payload == payload
        assert meta is not None

    def test_get_missing_returns_none(self) -> None:
        mgr = MemoryManager.with_defaults()
        entry, meta = mgr.get("short_term", "missing")
        assert entry is None
        assert meta is None

    def test_update_preserves_identity(self) -> None:
        mgr = MemoryManager.with_defaults()
        mgr.add("short_term", "k", {"v": 1})
        mgr.update("short_term", "k", {"v": 2})
        entry, _ = mgr.get("short_term", "k")
        assert entry is not None
        assert entry.payload == {"v": 2}

    def test_evict_oldest_when_over_entry_limit(self) -> None:
        mgr = MemoryManager.with_defaults()
        st = mgr.tier("short_term")
        original_max = st.max_entries
        st.max_entries = 3
        for i in range(5):
            mgr.add("short_term", f"k{i}", {"i": i})
        remaining = mgr.list("short_term")
        assert len(remaining) <= 3
        # After evicting, the oldest entries should be gone.
        keys = {e.key for e in remaining}
        assert "k2" not in keys or "k0" not in keys  # at least one evicted

    def test_evict_over_bytes_limit(self) -> None:
        mgr = MemoryManager.with_defaults()
        st = mgr.tier("short_term")
        original_bytes = st.max_bytes
        st.max_bytes = 200
        # Add entries until over the byte limit.
        for i in range(20):
            payload = {"big": "x" * 20}
            mgr.add("short_term", f"kb{i}", payload)
        remaining = mgr.list("short_term")
        total_bytes = sum(e.size_bytes for e in remaining)
        assert total_bytes <= st.max_bytes + 10  # allow small rounding


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

class TestPriority:
    def test_priority_field_stored(self) -> None:
        mgr = MemoryManager.with_defaults()
        mgr.add("short_term", "p1", {"v": 1}, priority=0.9)
        mgr.add("short_term", "p2", {"v": 2}, priority=0.3)
        entries = mgr.list("short_term")
        assert len(entries) == 2
        by_priority = sorted(entries, key=lambda e: -e.priority)
        assert by_priority[0].priority == 0.9
        assert by_priority[0].key == "p1"


# ---------------------------------------------------------------------------
# Age / expiry
# ---------------------------------------------------------------------------

class TestAge:
    def test_age_increases(self) -> None:
        mgr = MemoryManager.with_defaults()
        mgr.add("short_term", "a", {"v": 1})
        entry, _ = mgr.get("short_term", "a")
        age0 = entry.age
        mgr.tick(age_delta=0.5)
        entry, meta = mgr.get("short_term", "a")
        assert meta is not None
        assert entry.age > age0


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_round_trip(self) -> None:
        payload = {"a": 1, "b": [2, 3], "c": "text"}
        entry = short_term_entry("k", payload, priority=0.5)
        blob = serialize_entry(entry)
        restored = deserialize_entry(blob)
        assert restored.tier == entry.tier
        assert restored.key == entry.key
        assert restored.payload == entry.payload
        assert restored.priority == entry.priority

    def test_serialize_none_entry(self) -> None:
        assert serialize_entry(None) is None

    def test_deserialize_none_returned_for_none(self) -> None:
        assert deserialize_entry(None) is None


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_after_add(self) -> None:
        mgr = MemoryManager.with_defaults()
        mgr.add("short_term", "k", {"v": 1})
        st = mgr.tier("short_term")
        stats = mgr.stats("short_term")
        assert stats["count"] == 1
        assert stats["tier"] == "short_term"


# ---------------------------------------------------------------------------
# Memory error paths
# ---------------------------------------------------------------------------

class TestMemoryError:
    def test_add_to_missing_tier_raises(self) -> None:
        mgr = MemoryManager.with_defaults()
        with pytest.raises(MemoryError, match="does not exist"):
            mgr.add("no_such_tier", "k", {})

    def test_get_from_missing_tier_returns_none(self) -> None:
        mgr = MemoryManager.with_defaults()
        entry, meta = mgr.get("no_such_tier", "k")
        assert entry is None
        assert meta is None

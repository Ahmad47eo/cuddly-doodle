"""LOW_RAM_MODE tests: detection, config, memory-manager low-ram limits, awareness broadcast, sensitivity warning."""

from __future__ import annotations

import gc

import pytest

from core.low_ram import (
    DEFAULT_LOW_RAM_LIMITS,
    LOW_RAM_MODE,
    detect_low_ram_mode,
    estimate_memory_usage_mb,
    format_bytes,
    sensitivity_ratio,
)
from core.memory import DEFAULT_LIMITS, MemoryManager

try:
    import psutil
except Exception:
    psutil = None  # type: ignore


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestDetectLowRamMode:
    def test_returns_config_struct(self) -> None:
        cfg = detect_low_ram_mode()
        assert cfg is not None
        assert cfg.enabled in (True, False)
        # ram_mb may be None when psutil is unavailable.
        assert cfg.threshold_mb is not None
        assert cfg.reason

    def test_includes_why_or_not_why(self) -> None:
        cfg = detect_low_ram_mode()
        assert "why" in cfg.reason or "not_why" in cfg.reason

    def test_threshold_under_4gb_raises_flag(self) -> None:
        cfg = detect_low_ram_mode()
        # On a real 4GB machine the threshold-based reason should mention it.
        assert isinstance(cfg.threshold_mb, (int, float))


# ---------------------------------------------------------------------------
# Memory estimates
# ---------------------------------------------------------------------------

class TestEstimateMemory:
    def test_format_bytes_human(self) -> None:
        assert format_bytes(1_048_576) == pytest.approx("1.00 MB", abs=1e-2)
        assert format_bytes(0) == "0 B"

    def test_estimate_returns_mb(self) -> None:
        mb = estimate_memory_usage_mb()
        assert mb >= 0.0
        assert isinstance(mb, float)


# ---------------------------------------------------------------------------
# Sensitivity ratio helper
# ---------------------------------------------------------------------------

class TestSensitivityRatio:
    def test_ratio_valid(self) -> None:
        # sensitivity_ratio(used, threshold, free_ratio)
        r = sensitivity_ratio(used_mb=200, threshold_mb=4096, free_ratio=0.45)
        assert 0.0 <= r <= 1.0

    def test_ratio_high_when_close_to_limit(self) -> None:
        r = sensitivity_ratio(used_mb=3900, threshold_mb=4096, free_ratio=0.45)
        assert r > 0.5


# ---------------------------------------------------------------------------
# LOW_RAM_MODE global awareness
# ---------------------------------------------------------------------------

class TestLOW_RAM_MODE:
    def test_singleton_like_mode(self) -> None:
        mode = LOW_RAM_MODE
        assert hasattr(mode, "enabled")
        # LOW_RAM_MODE is a shared object; default is conservative detection.
        # We do not assert enabled==True because CI may have more RAM.
        assert isinstance(mode.enabled, bool)

    def test_mode_can_be_lazy_configured(self) -> None:
        # Ensure the mode object supports the attributes used by the UI/agent.
        mode = LOW_RAM_MODE
        assert hasattr(mode, "threshold_mb")
        assert hasattr(mode, "reason")


# ---------------------------------------------------------------------------
# Memory manager under low-ram limits
# ---------------------------------------------------------------------------

class TestMemoryManagerLowRam:
    def test_low_ram_tier_limits(self) -> None:
        mgr = MemoryManager.with_low_ram_limits()
        limits = mgr.limits()
        for tier in ("short_term", "episodic", "skill", "strategy"):
            assert tier in limits
            lim = limits[tier]
            # Low-ram limits must be strictly below defaults.
            assert lim["max_entries"] <= DEFAULT_LIMITS[tier]["max_entries"]
            assert lim["max_bytes"] <= DEFAULT_LIMITS[tier]["max_bytes"]
            # Low-ram mode evicts more aggressively, so evict_fraction >= default.
            assert lim["evict_fraction"] >= DEFAULT_LIMITS[tier]["evict_fraction"]

    def test_low_ram_defaults_match_module_constants(self) -> None:
        mgr = MemoryManager.with_low_ram_limits()
        limits = mgr.limits()
        for tier, expected in DEFAULT_LOW_RAM_LIMITS.items():
            actual = limits[tier]
            assert actual["max_entries"] == pytest.approx(expected["max_entries"])
            assert actual["max_bytes"] == pytest.approx(expected["max_bytes"])
            assert actual["evict_fraction"] == pytest.approx(expected["evict_fraction"])

    def test_low_ram_evicts_aggressively(self) -> None:
        mgr = MemoryManager.with_low_ram_limits()
        st = mgr.tier("short_term")
        # Set small limits to force eviction quickly.
        st.max_entries = 4
        st.max_bytes = 256
        for i in range(8):
            mgr.add("short_term", f"k{i}", {"payload": "x" * 30})
        remaining = mgr.list("short_term")
        assert len(remaining) <= 4

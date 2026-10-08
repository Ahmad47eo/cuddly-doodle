"""LOW_RAM_MODE support for constrained hardware.

Provides:

- Automatic detection of constrained RAM and recommendation of LOW_RAM_MODE.
- Aggressive tier limits for the memory system.
- Memory-usage estimation helpers.
- A shared LOW_RAM_MODE awareness object used by the agent, router, UI,
  and memory system.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

try:
    import psutil
except Exception:
    psutil = None  # type: ignore


# ---------------------------------------------------------------------------
# Default low-ram limits (mirrored from memory.py for convenience)
# ---------------------------------------------------------------------------

DEFAULT_LOW_RAM_LIMITS: dict[str, dict[str, int | float]] = {
    "short_term": {"max_entries": 16, "max_bytes": 512_000, "evict_fraction": 0.50},
    "episodic": {"max_entries": 32, "max_bytes": 1_000_000, "evict_fraction": 0.40},
    "skill": {"max_entries": 48, "max_bytes": 1_000_000, "evict_fraction": 0.40},
    "strategy": {"max_entries": 48, "max_bytes": 1_000_000, "evict_fraction": 0.40},
}


@dataclass
class LowRamDetectionResult:
    """Result of low-ram detection."""

    enabled: bool
    ram_mb: Optional[float]
    threshold_mb: float
    reason: dict[str, Any]


def detect_low_ram_mode() -> LowRamDetectionResult:
    """Detect whether LOW_RAM_MODE should be recommended.

    Uses psutil when available; falls back to conservative heuristics.
    This does NOT change system settings or the pagefile.
    """
    threshold_mb = 4096.0  # 4 GB is the target constraint boundary
    ram_mb: Optional[float] = None
    reasons: list[str] = []
    not_reasons: list[str] = []

    if psutil is not None:
        try:
            mem = psutil.virtual_memory()
            ram_total = mem.total
            ram_mb = ram_total / (1024 * 1024)
            if ram_mb < threshold_mb:
                reasons.append(f"total RAM {ram_mb:.0f} MB is below {threshold_mb:.0f} MB threshold")
            else:
                not_reasons.append(f"total RAM {ram_mb:.0f} MB is at or above {threshold_mb:.0f} MB threshold")
            if mem.available / mem.total < 0.25:
                reasons.append("available RAM is low relative to total")
        except Exception as e:
            not_reasons.append(f"psutil query failed: {e}")
    else:
        not_reasons.append("psutil not available — using conservative heuristic")

    # Conservative heuristic: if psutil isn't available, assume constrained
    # on low-end hosts unless environment explicitly opts out.
    constrained_by_default = os.environ.get("FORTNITE_AI_LOW_RAM_MODE") is not None or (
        psutil is None and sys.platform == "win32"
    )
    if constrained_by_default and not reasons:
        reasons.append("conservative low-ram heuristic active")

    enabled = bool(reasons)

    return LowRamDetectionResult(
        enabled=enabled,
        ram_mb=ram_mb,
        threshold_mb=threshold_mb,
        reason={"why": reasons, "not_why": not_reasons},
    )


def estimate_memory_usage_mb() -> float:
    """Estimate current process RSS in MB.

    Returns 0.0 when psutil is unavailable.
    """
    if psutil is None:
        return 0.0
    try:
        proc = psutil.Process()
        mem = proc.memory_info()
        return getattr(mem, "rss", 0) / (1024 * 1024)
    except Exception:
        return 0.0


def format_bytes(num_bytes: int) -> str:
    """Human-friendly byte formatting."""
    if num_bytes < 0:
        num_bytes = 0
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def sensitivity_ratio(used_mb: float, threshold_mb: float, free_ratio: float) -> float:
    """Return a 0..1 sensitivity measure for RAM pressure.

    Higher means closer to the constraint boundary.
    """
    if threshold_mb <= 0:
        return 0.0
    ratio = used_mb / threshold_mb
    ratio = max(0.0, min(1.0, ratio))
    return ratio


@dataclass
class LOW_RAM_MODE:
    """Shared read-only awareness object for LOW_RAM_MODE state.

    Updated lazily by detection. Consumers read `enabled` and `reason`
    without triggering repeated psutil queries on the UI thread.
    """

    enabled: bool = False
    threshold_mb: float = 4096.0
    reason: dict[str, Any] = None
    _last_ram_mb: Optional[float] = None

    def __post_init__(self) -> None:
        if self.reason is None:
            self.reason = {"why": [], "not_why": []}

    def refresh(self) -> None:
        result = detect_low_ram_mode()
        self.enabled = result.enabled
        self.threshold_mb = result.threshold_mb
        self.reason = result.reason
        self._last_ram_mb = result.ram_mb

    @property
    def ram_mb(self) -> Optional[float]:
        return self._last_ram_mb

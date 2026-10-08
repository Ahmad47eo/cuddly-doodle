"""Structured errors for the Fortnite AI Research platform.

The safety exception is the mechanism through which any attempt to cross
the real-game control boundary is rejected explicitly.
"""

from __future__ import annotations

from typing import Any


class ResearchError(Exception):
    """Base class for all platform errors."""

    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = dict(context) if context else {}


class SafetyViolationError(ResearchError):
    """Raised when an operation would cross the real-game control boundary.

    Real Fortnite MUST remain observation / analysis / coaching only.
    Full agent control exists ONLY inside the dedicated simulator.
    """

    def __init__(self, message: str, *, attempted: str | None = None) -> None:
        super().__init__(
            message,
            context={"fatality": "safety_boundary", "attempted_operation": attempted or "unknown"},
        )
        self.attempted = attempted or "unknown"


class SimulatorError(ResearchError):
    """Errors raised by the training/simulation environment."""


class PolicyError(ResearchError):
    """Errors raised by policies or decision-making components."""


class ModelError(ResearchError):
    """Errors raised by model providers or the router."""


class ObservationError(ResearchError):
    """Errors raised by the observation pipeline."""


class DatasetError(ResearchError):
    """Errors raised by dataset validation / loading."""


class MemoryError(ResearchError):
    """Errors raised by the memory system."""


class ConfigError(ResearchError):
    """Errors raised by invalid configuration."""


class LowRamError(ResearchError):
    """Errors raised when LOW_RAM_MODE cannot satisfy a request."""

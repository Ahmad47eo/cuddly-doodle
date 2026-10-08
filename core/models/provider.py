"""Model provider abstraction.

Provider-independent AI architecture with:

* ModelProvider — capability discovery, inference, load state
* Capability — what a model can do, with strengths/weaknesses/task hints
* ScoringContext — task, latency budget, cost, small-first preference
* simple_scoring — deterministic scoring for lightweight routing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional

# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------

class CapabilityLevel(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XL = "xl"


@dataclass
class Capability:
    """Describes what a model/provider can do."""

    # Convenience aliases so callers can write Capability.LEVEL_SMALL etc.
    LEVEL_SMALL = CapabilityLevel.SMALL
    LEVEL_MEDIUM = CapabilityLevel.MEDIUM
    LEVEL_LARGE = CapabilityLevel.LARGE
    LEVEL_XL = CapabilityLevel.XL

    name: str
    level: CapabilityLevel
    strengths: list[str]
    weaknesses: list[str]
    task_hints: list[str] = field(default_factory=list)
    cost_usd_per_1k_tokens: float = 0.0
    estimated_latency_ns: int = 0

    @classmethod
    def describe(
        cls,
        *,
        name: str,
        level: CapabilityLevel,
        strengths: list[str],
        weaknesses: list[str],
        task_hints: list[str] | None = None,
        cost_usd_per_1k_tokens: float = 0.0,
        estimated_latency_ns: int = 0,
    ) -> "Capability":
        return cls(
            name=name,
            level=level,
            strengths=strengths,
            weaknesses=weaknesses,
            task_hints=task_hints or [],
            cost_usd_per_1k_tokens=cost_usd_per_1k_tokens,
            estimated_latency_ns=estimated_latency_ns,
        )

    def estimated_latency_ns_for(self, context: "ScoringContext") -> int:
        if self.estimated_latency_ns:
            return self.estimated_latency_ns
        # Level-based fallback
        fallback = {
            CapabilityLevel.SMALL: 400,
            CapabilityLevel.MEDIUM: 8000,
            CapabilityLevel.LARGE: 60000,
            CapabilityLevel.XL: 250000,
        }
        return fallback.get(self.level, 8000)


# ---------------------------------------------------------------------------
# Latency budget
# ---------------------------------------------------------------------------

@dataclass
class LatencyBudget:
    soft_ns: int = 50000
    hard_ns: int = 200000

    def __post_init__(self) -> None:
        if self.soft_ns < 0 or self.hard_ns < 0:
            raise ValueError("latency budget must be non-negative")
        if self.hard_ns < self.soft_ns:
            raise ValueError("hard_ns must be >= soft_ns")


# ---------------------------------------------------------------------------
# Scoring context
# ---------------------------------------------------------------------------

@dataclass
class ScoringContext:
    task_hint: str = ""
    max_latency_ns: int = 100000
    latency_budget: LatencyBudget = field(default_factory=LatencyBudget)
    max_cost_usd_per_1k: float = 1.0
    prefer_small_first: bool = True

    def __post_init__(self) -> None:
        if self.max_cost_usd_per_1k < 0:
            raise ValueError("max_cost_usd_per_1k must be >= 0")


# ---------------------------------------------------------------------------
# Inference result
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    provider: str = "unknown"
    latency_ns: int = 0
    api_latency_ns: int = 0
    compute_latency_ns: int = 0
    confidence: float = 0.0
    intended_action: str = "noop"
    intended_params: dict[str, float] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Load state
# ---------------------------------------------------------------------------

@dataclass
class LoadState:
    loaded: bool = True
    error: str | None = None


# ---------------------------------------------------------------------------
# Model provider interface
# ---------------------------------------------------------------------------

class ModelProvider:
    """Abstract model/provider contract."""

    def capabilities(self, context: ScoringContext) -> list[Capability]:
        raise NotImplementedError

    def infer(
        self,
        context: ScoringContext,
        task_hint: str,
        payload: dict[str, Any],
    ) -> InferenceResult:
        raise NotImplementedError

    def load_state(self) -> LoadState:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Scoring result (lightweight container for router use)
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    capability_name: str
    score: float
    latency_ns: int
    cost_usd_per_1k_tokens: float
    strengths_hit: list[str] = field(default_factory=list)
    weakness_penalty: float = 0.0


# ---------------------------------------------------------------------------
# Simple scoring (deterministic, used by router and tests)
# ---------------------------------------------------------------------------

def simple_scoring(capabilities: list[Capability], context: ScoringContext) -> dict[str, float]:
    """Score each capability and return a mapping name->score."""
    scores: dict[str, float] = {}
    for cap in capabilities:
        score = _score_capability(cap, context)
        scores[cap.name] = score
    return scores


def _score_capability(cap: Capability, context: ScoringContext) -> float:
    base = 0.0
    strengths_hit: list[str] = []
    weakness_penalty = 0.0

    # Base score from level
    if cap.level is CapabilityLevel.SMALL:
        base += 1.0
    elif cap.level is CapabilityLevel.MEDIUM:
        base += 2.0
    elif cap.level is CapabilityLevel.LARGE:
        base += 3.0
    elif cap.level is CapabilityLevel.XL:
        base += 4.0

    # Task hint match
    if context.task_hint:
        hint = context.task_hint.lower()
        if any(hint in s.lower() for s in cap.strengths):
            base += 2.0
            strengths_hit.append("strength_match")
        else:
            base -= 0.5

    # Latency
    latency = cap.estimated_latency_ns_for(context)
    base -= latency / max(1, context.max_latency_ns) * 2.0

    # Cost
    if context.max_cost_usd_per_1k > 0:
        cost_ratio = cap.cost_usd_per_1k_tokens / context.max_cost_usd_per_1k
        base -= cost_ratio * 2.0

    # Small-model-first bonus
    if context.prefer_small_first and cap.level is CapabilityLevel.SMALL:
        base += 1.0

    # Weakness penalty
    for weakness in cap.weaknesses:
        if context.task_hint and weakness in context.task_hint.lower():
            weakness_penalty += 0.6
        elif context.max_latency_ns and weakness == "latency" and latency > context.max_latency_ns:
            weakness_penalty += 0.6

    score = max(0.0, base - weakness_penalty)
    return score

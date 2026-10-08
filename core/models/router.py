"""Model router — provider-independent AI routing.

Decides which model provider to use for a given task based on:

* Task complexity / task hint
* Latency budget
* Hardware constraints (including LOW_RAM_MODE)
* Model availability / load state
* Cost
* Context requirements
* Training vs inference mode

Uses the smallest capable provider whenever possible and escalates only
when necessary. If cloud inference is unavailable, gracefully falls back
to a local method.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from core.low_ram import LOW_RAM_MODE
from enum import Enum

from core.models.provider import (
    Capability,
    InferenceResult,
    LatencyBudget,
    LoadState,
    ModelProvider,
    ScoringContext,
    simple_scoring,
)


class RoutingReason(str, Enum):
    PERFECT_MATCH = "perfect_match"
    BEST_AVAILABLE = "best_available"
    SMALL_MODEL_FIRST = "small_model_first"
    FALLBACK = "fallback"
    TIMEOUT = "timeout"
    DISABLED_PRIMARY = "disabled_primary"
    UNKNOWN = "unknown"


@dataclass
class RoutingDecision:
    provider_name: str
    reason: RoutingReason
    capability: Capability | None = None
    explanation: str = ""
    meta: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {}


class ModelRouter:
    """Router that selects among configured providers for a task."""

    def __init__(
        self,
        *,
        fallback_enabled: bool = True,
        low_ram_escalate_threshold_ns: int = 10000,
    ) -> None:
        self.fallback_enabled = fallback_enabled
        self.low_ram_escalate_threshold_ns = low_ram_escalate_threshold_ns
        self._last_decision: RoutingDecision | None = None

    def choose(
        self,
        providers: list[tuple[str, ModelProvider]],
        context: ScoringContext,
    ) -> RoutingDecision:
        """Select a provider for the given task context.

        providers: ordered list of (name, provider). The router prefers
        earlier providers when they are capable and within latency budget.
        """
        t0 = time.perf_counter()
        try:
            return self._choose_internal(providers, context)
        finally:
            elapsed_ns = int((time.perf_counter() - t0) * 1e9)
            if self._last_decision is not None:
                self._last_decision.meta["router_latency_ns"] = elapsed_ns

    def _choose_internal(
        self,
        providers: list[tuple[str, ModelProvider]],
        context: ScoringContext,
    ) -> RoutingDecision:
        if not providers:
            return RoutingDecision(
                provider_name="none",
                reason=RoutingReason.UNKNOWN,
                explanation="no providers configured",
            )

        # Score each provider's capabilities for this task.
        # Skip providers that are not loaded / disabled.
        first_disabled = False
        scored: list[tuple[str, float, Capability | None, ModelProvider]] = []
        for idx, (name, provider) in enumerate(providers):
            try:
                state = provider.load_state()
            except Exception:
                state = LoadState(loaded=False, error="load_state failed")
            if not state.loaded:
                if idx == 0:
                    first_disabled = True
                continue
            try:
                caps = provider.capabilities(context)
            except Exception:
                caps = []
            if not caps:
                continue
            scores = simple_scoring(caps, context)
            best_cap = None
            best_score = -1.0
            for cap in caps:
                s = scores.get(cap.name, 0.0)
                if s > best_score:
                    best_score = s
                    best_cap = cap
            if best_cap is not None:
                scored.append((name, best_score, best_cap, provider))

        if not scored:
            # All providers failed to describe capabilities — fall back to
            # the first provider if possible, else none.
            if providers:
                return RoutingDecision(
                    provider_name=providers[0][0],
                    reason=RoutingReason.BEST_AVAILABLE,
                    capability=None,
                    explanation="no capabilities reported; using first provider",
                )
            return RoutingDecision(
                provider_name="none",
                reason=RoutingReason.UNKNOWN,
                explanation="no providers available",
            )

        # Sort by score descending, then by provider order for stability.
        scored.sort(key=lambda x: (-x[1], providers.index((x[0], x[3])) if (x[0], x[3]) in providers else 999))

        best_name, best_score, best_cap, best_provider = scored[0]

        # Decide reason.
        if first_disabled and best_name != providers[0][0]:
            reason = RoutingReason.FALLBACK
            explanation = f"primary provider {providers[0][0]} unavailable, using {best_name}"
        elif best_cap.level.value == "small" and context.prefer_small_first and best_score > 0:
            reason = RoutingReason.SMALL_MODEL_FIRST
            explanation = f"small model {best_name} is sufficient for {context.task_hint or 'task'}"
        elif best_score > 0:
            reason = RoutingReason.PERFECT_MATCH
            explanation = f"{best_name} best matches {context.task_hint or 'task'}"
        else:
            reason = RoutingReason.BEST_AVAILABLE
            explanation = f"{best_name} selected as best available"

        # Latency guard: if the best provider's estimated latency exceeds the
        # hard budget, try to fall back to a smaller/faster provider if one exists.
        if best_cap and best_cap.estimated_latency_ns_for(context) > context.latency_budget.hard_ns:
            smaller = None
            for name, score, cap, prov in scored:
                if cap.level.value in ("small", "medium") and cap.estimated_latency_ns_for(context) <= context.latency_budget.hard_ns:
                    smaller = (name, cap, prov)
                    break
            if smaller is not None:
                name, cap, prov = smaller
                return RoutingDecision(
                    provider_name=name,
                    reason=RoutingReason.SMALL_MODEL_FIRST,
                    capability=cap,
                    explanation=f"latency guard: prefer {name} over {best_name}",
                    meta={"avoided_provider": best_name},
                )

        # LOW_RAM_MODE guard: in low-ram mode, strongly prefer cheap/fast
        # providers over expensive/large ones when the task is simple.
        if LOW_RAM_MODE.enabled and context.task_hint in ("steer", "shoot", "move", "reload", "use_item", ""):
            simple_provider = None
            for name, score, cap, prov in scored:
                if cap.level.value == "small" and cap.estimated_latency_ns_for(context) <= context.latency_budget.hard_ns:
                    simple_provider = (name, cap, prov)
                    break
            if simple_provider is not None:
                name, cap, prov = simple_provider
                return RoutingDecision(
                    provider_name=name,
                    reason=RoutingReason.SMALL_MODEL_FIRST,
                    capability=cap,
                    explanation=f"LOW_RAM_MODE: using small local provider {name}",
                    meta={"avoided_provider": best_name},
                )

        # If the selected provider cannot meet the hard latency budget and no
        # fallback was available, report TIMEOUT.
        final_reason = reason
        if best_cap and best_cap.estimated_latency_ns_for(context) > context.latency_budget.hard_ns:
            final_reason = RoutingReason.TIMEOUT
            explanation = f"{best_name} exceeds hard latency budget ({best_cap.estimated_latency_ns_for(context)} ns > {context.latency_budget.hard_ns} ns)"

        return RoutingDecision(
            provider_name=best_name,
            reason=final_reason,
            capability=best_cap,
            explanation=explanation,
            meta={"latency_estimate_ns": best_cap.estimated_latency_ns_for(context) if best_cap else 0},
        )


def choose_provider(
    providers: list[tuple[str, ModelProvider]],
    context: ScoringContext,
    *,
    router: ModelRouter | None = None,
) -> RoutingDecision:
    """Convenience wrapper for ModelRouter.choose with a default router."""
    router = router or ModelRouter()
    return router.choose(providers, context)

"""Dummy cloud model provider.

Placeholder for an external cloud inference provider (for example an
OpenAI-style endpoint, a cloud GPU inference service, or a remote
training/inference backend). The real provider would be wired through
environment variables (never hard-coded) and selected by the router when
a task needs more capability than the local provider offers.

This dummy exists so the router, fallback chain, scoring, and evaluation
can be tested without requiring a live cloud service.
"""

from __future__ import annotations

import time

from core.models.provider import (
    Capability,
    CapabilityLevel,
    InferenceResult,
    LoadState,
    ModelProvider,
    ScoringContext,
)


class DummyCloudProvider(ModelProvider):
    """A dummy cloud provider that is slower but more capable."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        name: str = "dummy_cloud",
        base_latency_ns: int = 25000,
        confidence: float = 0.9,
        error: str | None = None,
    ) -> None:
        self._enabled = enabled
        self._name = name
        self._base_latency_ns = max(0, base_latency_ns)
        self._confidence = max(0.0, min(1.0, confidence))
        self._error = error
        self._loaded = not error

    def capabilities(self, context: ScoringContext) -> list[Capability]:
        return [
            Capability.describe(
                name=self._name,
                level=CapabilityLevel.LARGE,
                strengths=["planning", "strategy", "high_level", "reasoning"],
                weaknesses=["latency", "cost", "thin_local_reactivity"],
                task_hints=["plan", "strategy", "high_level", "reasoning"],
                cost_usd_per_1k_tokens=1.5,
                estimated_latency_ns=self._base_latency_ns,
            ),
        ]

    def infer(
        self,
        context: ScoringContext,
        task_hint: str,
        payload: dict[str, Any],
    ) -> InferenceResult:
        if not self._enabled:
            raise RuntimeError(f"provider '{self._name}' is disabled")
        t0 = time.perf_counter()
        action, params = self._decide(task_hint, payload)
        elapsed_ns = int((time.perf_counter() - t0) * 1e9)
        total_latency = max(self._base_latency_ns, elapsed_ns)
        api_latency = max(0, total_latency - elapsed_ns)
        return InferenceResult(
            provider=self._name,
            latency_ns=total_latency,
            api_latency_ns=api_latency,
            compute_latency_ns=elapsed_ns,
            confidence=self._confidence,
            intended_action=action,
            intended_params=params,
            payload=dict(payload),
        )

    def load_state(self) -> LoadState:
        if self._error:
            return LoadState(loaded=False, error=self._error)
        return LoadState(loaded=self._enabled, error=None)

    def _decide(self, task_hint: str, payload: dict[str, Any]) -> tuple[str, dict[str, float]]:
        hint = (task_hint or "").lower()
        if hint in ("plan", "high_level", "strategy", "reasoning"):
            return "high_level_plan", {}
        if hint in ("steer", "move", "shoot", "fire", "aim", "reload"):
            # Cloud provider can still handle simple tasks but is slower.
            return "noop", {}
        return "noop", {}

"""Dummy local model provider.

Used for testing the router and as a placeholder for the real local
inference provider that would be wired later (for example a lightweight
ONNX/TensorFlow/PyTorch policy evaluated locally on CPU when GPU is
unavailable).

This provider is intentionally lightweight: it runs on CPU, stays small,
and returns fast deterministic results so the platform can be tested
without any external service.
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


class DummyLocalProvider(ModelProvider):
    """A small, fast, local dummy provider for development/testing."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        name: str = "dummy_local",
        latency_ns: int = 200,
        confidence: float = 0.85,
        error: str | None = None,
    ) -> None:
        self._enabled = enabled
        self._name = name
        self._latency_ns = max(0, latency_ns)
        self._confidence = max(0.0, min(1.0, confidence))
        self._error = error
        self._loaded = not error

    def capabilities(self, context: ScoringContext) -> list[Capability]:
        return [
            Capability.describe(
                name=self._name,
                level=CapabilityLevel.SMALL,
                strengths=["speed", "steer", "shoot", "move", "reload"],
                weaknesses=["planning", "strategy", "thin_context"],
                task_hints=["steer", "shoot", "move", "reload", "use_item"],
                cost_usd_per_1k_tokens=0.0,
                estimated_latency_ns=self._latency_ns,
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
        # Ensure total latency is at least the configured base latency (simulates
        # deterministic local inference cost) plus real compute time.
        total_latency = max(self._latency_ns, elapsed_ns)
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
        if hint in ("steer", "move", "mobility"):
            return "move", {"dx": 10.0, "dy": 5.0}
        if hint in ("shoot", "fire", "aim"):
            return "fire", {"target": 1.0}
        if hint in ("reload",):
            return "reload", {}
        if hint in ("use_item", "heal", "item"):
            return "use_item", {}
        if hint in ("plan", "strategy", "high_level"):
            return "high_level_plan", {}
        if hint in ("loot", "resource", "gather"):
            return "interact", {}
        return "noop", {}

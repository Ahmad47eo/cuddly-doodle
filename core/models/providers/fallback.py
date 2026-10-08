"""Fallback provider wrapper.

Wraps a primary provider with a fallback provider. If the primary is
disabled or raises, the fallback is used instead. The wrapper reports
through the same ModelProvider interface so the router can treat it as
a single provider while still benefiting from graceful degradation.

This is part of the small-model-first / graceful-fallback model routing
story: if cloud inference is unavailable, the system falls back to a
local method instead of failing completely.
"""

from __future__ import annotations

from typing import Any

from core.models.provider import (
    Capability,
    InferenceResult,
    LoadState,
    ModelProvider,
    ScoringContext,
)


class FallbackProvider(ModelProvider):
    """Provider that falls back to a secondary provider on failure/disabling."""

    def __init__(
        self,
        *,
        primary: ModelProvider,
        fallback: ModelProvider,
        name: str = "fallback",
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._name = name

    def capabilities(self, context: ScoringContext) -> list[Capability]:
        caps = list(self._primary.capabilities(context))
        for c in self._fallback.capabilities(context):
            if not any(c.name == x.name for x in caps):
                caps.append(c)
        return caps

    def infer(
        self,
        context: ScoringContext,
        task_hint: str,
        payload: dict[str, Any],
    ) -> InferenceResult:
        try:
            if self._primary.load_state().loaded:
                return self._primary.infer(context, task_hint, payload)
        except Exception:
            pass
        # Primary unavailable or failed — use fallback.
        return self._fallback.infer(context, task_hint, payload)

    def load_state(self) -> LoadState:
        primary_state = self._primary.load_state()
        if primary_state.loaded:
            return LoadState(loaded=True, error=primary_state.error)
        fallback_state = self._fallback.load_state()
        if fallback_state.loaded:
            return LoadState(loaded=True, error="primary unavailable, using fallback")
        return LoadState(loaded=False, error="primary and fallback unavailable")

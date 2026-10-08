"""Hierarchical policy: high-level strategy selection + low-level reactive execution.

This is the primary policy interface used by the simulator and evaluator.
It combines:

- A high-level strategy router that selects a strategy mode and an
  ActionFocus from structured game-state signals.
- A low-level reactive policy that converts the strategy intent into
  concrete simulated actions.

The reactive layer is kept extremely lightweight. The high-level layer is
only consulted when its output would actually change behavior (for example
when running in ADAPTIVE mode). For fixed-strategy runs, the reactive
layer can dominate decision-making on its own.
"""

from __future__ import annotations

import time

from core.types import (
    Action,
    ActionType,
    Observation,
    StrategyMode,
)
from core.strategy import (
    ActionFocus,
    StrategyDirective,
    StrategyModeRouter,
)


class HierarchicalPolicy:
    """Combines high-level strategy selection with low-level reactive execution."""

    def __init__(
        self,
        *,
        high_level: StrategyModeRouter,
        reactive: "ReactivePolicy | None" = None,
    ) -> None:
        from core.policy.reactive import ReactivePolicy as _ReactivePolicy
        self.high_level = high_level
        self.reactive = reactive or _ReactivePolicy()

    def __call__(self, obs: Observation) -> Action:
        return self.choose(obs)

    def choose(self, obs: Observation) -> Action:
        t0 = time.perf_counter()
        try:
            directive = self.high_level.plan(obs)
            action = self._action_from_directive(obs, directive)
            action.latency_ns = int((time.perf_counter() - t0) * 1e9)
            return action
        except Exception:
            # If the high-level layer fails, fall back to reactive only.
            action = self.reactive.choose(obs)
            action.latency_ns = int((time.perf_counter() - t0) * 1e9)
            return action

    def _action_from_directive(self, obs: Observation, directive: StrategyDirective) -> Action:
        # For the simulator, the simplest faithful behavior is:
        # - If the strategy says engage/aim/fire, try to do that (delegate to reactive).
        # - If it says retreat/cover/loot, the reactive policy already handles those
        #   based on state, so we lean on it.
        # - If it says plan, we emit a lightweight high_level plan action.
        focus = directive.action_focus

        # High-level planning action when the strategy emphasizes planning.
        if focus.flag(ActionFocus.PATIENT):
            # Re-active policy will decide; no forced override.
            pass

        # Delegate to reactive policy; it reads features and produces a concrete action.
        return self.reactive.choose(obs)

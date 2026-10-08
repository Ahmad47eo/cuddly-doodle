"""Reactive (low-level) policy for the simulator.

Fast, deterministic, lightweight decisions that do NOT depend on any
real Fortnite client. This is the layer that should be optimized heavily:
it is called often and should make simple decisions with minimal overhead.

The policy reads structured observation features and emits simulator
actions: move, reload, use item, aim/fire toward a target, or noop.
"""

from __future__ import annotations

import math
import time

from core.types import (
    Action,
    ActionType,
    Observation,
    StrategyMode,
)


class ReactivePolicy:
    """Low-level reactive policy for the simulated environment."""

    def __init__(self, *, step_size: float = 22.0) -> None:
        self.step_size = step_size
        self.last_latency_ns: int = 0

    def __call__(self, obs: Observation) -> Action:
        return self.choose(obs)

    def choose(self, obs: Observation) -> Action:
        t0 = time.perf_counter()
        try:
            return self._decide(obs)
        finally:
            self.last_latency_ns = int((time.perf_counter() - t0) * 1e9)

    def _decide(self, obs: Observation) -> Action:
        self_entity = obs.self_entity
        features = obs.features

        # 1) Survival/restoration first
        if self_entity.health < 35.0 and obs.map_info.safe_zone_radius > 30.0:
            return self._move_toward_zone(obs)

        if not self_entity.loaded:
            return Action(ActionType.RELOAD, latency_ns=0)

        if self_entity.health < 50.0 and self_entity.Shield < 20.0 and obs.map_info.safe_zone_radius > 30.0:
            return Action(ActionType.USE_ITEM, latency_ns=0)

        # 2) Engage if there is a good target
        target_id, target_info = self._best_target(obs)
        if target_id is not None:
            sx, sy = self_entity.x, self_entity.y
            tx, ty = target_info["x"], target_info["y"]
            dist = math.hypot(tx - sx, ty - sy)
            confidence = max(0.3, 1.0 - (dist / 300.0))

            # Close enough to fire
            if dist < 120.0 and self_entity.loaded:
                return Action(
                    ActionType.FIRE,
                    target=target_id,
                    params={"dx": tx - sx, "dy": ty - sy},
                    confidence=confidence,
                    latency_ns=0,
                )

            # Move toward target
            if dist > 15.0:
                dx = (tx - sx) / (dist or 1.0)
                dy = (ty - sy) / (dist or 1.0)
                mag = self.step_size
                return Action(
                    ActionType.MOVE,
                    params={"dx": dx * mag, "dy": dy * mag},
                    target=target_id,
                    confidence=confidence,
                    latency_ns=0,
                )

            # Close but not firing (e.g., out of ammo handled above)
            return Action(ActionType.AIM, target=target_id, confidence=confidence, latency_ns=0)

        # 3) No relevant threat — reposition toward zone or hold
        if not obs.map_info.safe_zone_radius or obs.map_info.safe_zone_radius <= 0:
            return Action(ActionType.NOOP, latency_ns=0)

        return self._move_toward_zone(obs)

    def _best_target(self, obs: Observation):
        """Return (target_id, info) for the nearest enemy with positive health, or (None, None)."""
        self_entity = obs.self_entity
        best_id = None
        best_dist = 1e9
        best_info: dict[str, float] = {}
        for eid, e in obs.entities.items():
            if e.id == self_entity.id:
                continue
            # Only consider enemy entities (team 2) with health.
            if e.team.value != 2:
                continue
            if e.health <= 0:
                continue
            d = math.hypot(e.x - self_entity.x, e.y - self_entity.y)
            if d < best_dist:
                best_dist = d
                best_id = eid
                best_info = {"x": e.x, "y": e.y, "health": e.health}
        if best_id is None:
            return None, None
        return best_id, best_info

    def _move_toward_zone(self, obs: Observation) -> Action:
        self_entity = obs.self_entity
        zc = obs.map_info.safe_zone_center
        zr = obs.map_info.safe_zone_radius
        dx = zc[0] - self_entity.x
        dy = zc[1] - self_entity.y
        dist = math.hypot(dx, dy) or 1.0
        if zr > 0 and dist < zr * 0.6:
            return Action(ActionType.NOOP, latency_ns=0)
        step = self.step_size
        return Action(
            ActionType.MOVE,
            params={"dx": dx / dist * step, "dy": dy / dist * step},
            confidence=0.7,
            latency_ns=0,
        )

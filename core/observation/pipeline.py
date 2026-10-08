"""Observation pipeline for the platform.

Transforms raw simulator observations into feature-rich, cached,
temporally-contextualized representations used by the policy and coaching
layers. Optimized for low latency with:

* Lightweight preprocessing
* Feature extraction
* Observation deduplication / caching
* Temporal context windows
* Event detection
* Confidence estimation
* Timestamps
* Optional persistence

This pipeline is modular: new preprocessors/features can be plugged in
without rewriting the whole architecture.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from core.types import (
    Action,
    ActionType,
    Observation,
    Reward,
    RewardComponent,
    StrategyMode,
    TeamId,
)


# ---------------------------------------------------------------------------
# Pipeline errors
# ---------------------------------------------------------------------------

class ObservationError(Exception):
    """Raised by the observation pipeline."""


# ---------------------------------------------------------------------------
# Feature extraction contract
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Extracts a feature dict from an observation."""

    def extract(self, obs: Observation) -> dict[str, float]:
        raise NotImplementedError


class DefaultFeatureExtractor(FeatureExtractor):
    """Simple, fast feature extractor for the simulator."""

    def extract(self, obs: Observation) -> dict[str, float]:
        agent = obs.self_entity
        feats: dict[str, float] = {
            "self_health": agent.health,
            "self_shield": agent.Shield,
            "loaded": 1.0 if agent.loaded else 0.0,
            "enemy_count": float(sum(1 for e in obs.entities.values() if e.team == TeamId.ENEMY and e.health > 0)),
            "ally_count": float(sum(1 for e in obs.entities.values() if e.team == TeamId.ALLY and e.health > 0)),
            "zone_radius": obs.map_info.safe_zone_radius,
            "distance_to_zone_center": math.hypot(agent.x - obs.map_info.safe_zone_center[0], agent.y - obs.map_info.safe_zone_center[1]),
            "resources_nearby": float(sum(1 for it in obs.items.values() if not it.get("collected", False))),
            "enemy_threat_proxy": self._threat_proxy(obs),
            "resources_low_proxy": 1.0 if agent.loaded else 0.4,
            "aim_headshot_rate_proxy": obs.features.get("aim_headshot_rate_proxy", 0.5),
            "aim_accuracy_proxy": obs.features.get("aim_accuracy_proxy", 0.5),
            "aim_time_to_target_proxy": obs.features.get("aim_time_to_target_proxy", 0.5),
        }
        return feats

    def _threat_proxy(self, obs: Observation) -> float:
        agent = obs.self_entity
        enemies = [
            (eid, e) for eid, e in obs.entities.items()
            if e.team == TeamId.ENEMY and e.health > 0
        ]
        if not enemies:
            return 0.0
        nearest_d = min(math.hypot(e.x - agent.x, e.y - agent.y) for _, e in enemies)
        threat = max(0.0, 1.0 - nearest_d / 250.0)
        threat = min(1.0, threat * (1.0 + 0.1 * len(enemies)))
        return threat


# ---------------------------------------------------------------------------
# Caching / deduplication
# ---------------------------------------------------------------------------

@dataclass
class CachedFeature:
    tick: int
    timestamp: float
    features: dict[str, float]
    confidence: float
    hit: bool = False


class ObservationCache:
    """Bounded FIFO cache for observation features.

    Used to avoid re-extracting identical features when the observation
    has not meaningfully changed. In low-ram mode the cache is much
    smaller.
    """

    def __init__(self, *, max_entries: int = 64, low_ram: bool = False) -> None:
        if low_ram:
            max_entries = min(max_entries, 16)
        self._max = max(1, max_entries)
        self._store: dict[int, CachedFeature] = {}
        self._order: deque[int] = deque()

    def get(self, obs: Observation) -> Optional[CachedFeature]:
        key = self._key(obs)
        entry = self._store.get(key)
        if entry is not None:
            entry.hit = True
            return entry
        return None

    def put(self, obs: Observation, features: dict[str, float], confidence: float) -> None:
        key = self._key(obs)
        if key in self._store:
            self._store[key] = CachedFeature(
                tick=obs.tick,
                timestamp=time.time(),
                features=features,
                confidence=confidence,
                hit=True,
            )
            return
        if len(self._store) >= self._max:
            oldest = self._order.popleft()
            self._store.pop(oldest, None)
        self._store[key] = CachedFeature(
            tick=obs.tick,
            timestamp=time.time(),
            features=features,
            confidence=confidence,
            hit=False,
        )
        self._order.append(key)

    def _key(self, obs: Observation) -> int:
        # Deterministic cheap key: tuple of salient state hashed.
        parts = (
            obs.tick,
            obs.scenario,
            obs.difficulty,
            obs.self_entity.id,
            round(obs.self_entity.x, 1),
            round(obs.self_entity.y, 1),
            round(obs.self_entity.health, 1),
            round(obs.self_entity.Shield, 1),
            obs.map_info.safe_zone_radius,
            tuple(sorted((eid, round(e.health, 1), e.team.value) for eid, e in obs.entities.items())),
        )
        h = hash(parts)
        return h


# ---------------------------------------------------------------------------
# Temporal context window
# ---------------------------------------------------------------------------

@dataclass
class TemporalContext:
    recent_features: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=8))
    recent_events: deque[str] = field(default_factory=lambda: deque(maxlen=32))
    recent_actions: deque[str] = field(default_factory=lambda: deque(maxlen=16))
    last_reward: float = 0.0
    reward_history: deque[float] = field(default_factory=lambda: deque(maxlen=32))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class ObservationPipeline:
    """Modular observation pipeline for simulator or permitted observations."""

    def __init__(
        self,
        *,
        extractor: Optional[FeatureExtractor] = None,
        cache: Optional[ObservationCache] = None,
        temporal_window: int = 8,
        low_ram: bool = False,
    ) -> None:
        self.extractor = extractor or DefaultFeatureExtractor()
        self.cache = cache or ObservationCache(low_ram=low_ram)
        self.temporal = TemporalContext()
        self._temporal_window = temporal_window if not low_ram else min(temporal_window, 4)
        self._stats: PipelineStats = PipelineStats()
        self._low_ram = low_ram

    def process(self, obs: Observation, *, force_refresh: bool = False) -> dict[str, Any]:
        """Run the pipeline on an observation and return enriched context."""
        if not isinstance(obs, Observation):
            raise ObservationError("Expected an Observation instance")

        t0 = time.perf_counter()

        # 1) cache lookup
        cached = self.cache.get(obs) if not force_refresh else None
        if cached is not None and not force_refresh:
            self._stats.cache_hits += 1
            features = cached.features
            confidence = cached.confidence
        else:
            self._stats.cache_misses += 1
            features = self.extractor.extract(obs)
            confidence = self._estimate_confidence(obs, features)
            self.cache.put(obs, features, confidence)

        # 2) temporal context
        self.temporal.recent_features.append(features)
        for ev in obs.events:
            if ev:
                self.temporal.recent_events.append(ev)
        if obs.last_action:
            self.temporal.recent_actions.append(obs.last_action)
        self.temporal.last_reward = obs.last_reward
        self.temporal.reward_history.append(obs.last_reward)

        # 3) event detection
        events = self._detect_events(obs, features)

        # 4) timing
        elapsed_ns = int((time.perf_counter() - t0) * 1e9)
        self._stats.total_calls += 1
        self._stats.total_latency_ns += elapsed_ns
        if elapsed_ns > self._stats.max_latency_ns:
            self._stats.max_latency_ns = elapsed_ns
        if self._stats.min_latency_ns == 0 or elapsed_ns < self._stats.min_latency_ns:
            self._stats.min_latency_ns = elapsed_ns

        return {
            "features": features,
            "confidence": confidence,
            "events": events,
            "temporal": self._snapshot_temporal(),
            "latency_ns": elapsed_ns,
            "cached": cached is not None and not force_refresh,
        }

    def _estimate_confidence(self, obs: Observation, features: dict[str, float]) -> float:
        # Simple confidence heuristic based on feature completeness and
        # consistency with the observation's own confidence.
        base = float(obs.confidence)
        missing = sum(1 for k in ("self_health", "enemy_count", "zone_radius") if k not in features)
        penalty = min(0.3, missing * 0.1)
        return max(0.1, min(1.0, base - penalty))

    def _detect_events(self, obs: Observation, features: dict[str, float]) -> list[str]:
        events: list[str] = []
        if features.get("enemy_threat_proxy", 0.0) > 0.6:
            events.append("high_threat")
        if features.get("self_health", 100.0) < 30.0:
            events.append("low_health")
        if features.get("loaded", 1.0) < 0.5:
            events.append("low_ammo")
        if features.get("zone_radius", 1000.0) < 80.0:
            events.append("endgame_zone")
        if any(e in obs.events for e in ("zone_shrinking",)):
            events.append("zone_shrinking")
        return events

    def _snapshot_temporal(self) -> dict[str, Any]:
        return {
            "recent_features_count": len(self.temporal.recent_features),
            "recent_events_count": len(self.temporal.recent_events),
            "recent_actions_count": len(self.temporal.recent_actions),
            "last_reward": self.temporal.last_reward,
            "reward_history": list(self.temporal.reward_history),
        }

    def stats(self) -> PipelineStats:
        return PipelineStats(
            total_calls=self._stats.total_calls,
            cache_hits=self._stats.cache_hits,
            cache_misses=self._stats.cache_misses,
            total_latency_ns=self._stats.total_latency_ns,
            max_latency_ns=self._stats.max_latency_ns,
            min_latency_ns=self._stats.min_latency_ns,
            avg_latency_ns=(
                self._stats.total_latency_ns // self._stats.total_calls
                if self._stats.total_calls else 0
            ),
        )


@dataclass
class PipelineStats:
    total_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ns: int = 0
    max_latency_ns: int = 0
    min_latency_ns: int = 0
    avg_latency_ns: int = 0

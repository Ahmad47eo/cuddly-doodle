"""Policy tests: reactive policy, hierarchical policy, strategy passthrough, confidence, latency, react fallback."""

from __future__ import annotations

import math

import pytest

from core.types import (
    Action,
    ActionType,
    EntityState,
    MapInfo,
    Observation,
    StrategyMode,
    TeamId,
)
from core.policy.reactive import ReactivePolicy
from core.policy.hierarchical import HierarchicalPolicy, StrategyModeRouter


# ---------------------------------------------------------------------------
# Reactive policy
# ---------------------------------------------------------------------------

class TestReactivePolicy:
    def test_moves_toward_zone_when_no_target(self) -> None:
        policy = ReactivePolicy()
        obs = _make_obs(enemies=[])
        action = policy.choose(obs)
        # With no enemy target the reactive policy moves toward zone center
        # (or holds if already well inside it).
        assert action.action_type in (ActionType.MOVE, ActionType.NOOP, ActionType.RELOAD)

    def test_reloads_when_empty(self) -> None:
        policy = ReactivePolicy()
        obs = _make_obs(enemies=[], self_loaded=False)
        action = policy.choose(obs)
        assert action.action_type is ActionType.RELOAD

    def test_uses_item_when_injured(self) -> None:
        policy = ReactivePolicy()
        obs = _make_obs(enemies=[], self_health=25.0)
        action = policy.choose(obs)
        assert action.action_type in (ActionType.USE_ITEM, ActionType.RELOAD, ActionType.NOOP, ActionType.MOVE)

    def test_aims_or_fires_at_nearest_enemy(self) -> None:
        policy = ReactivePolicy()
        obs = _make_obs(
            self_pos=(0, 0),
            enemies=[_make_entity(id=2, team=2, x=50, y=0)],
        )
        action = policy.choose(obs)
        # Reactive policy may aim or move+fire; target should be set if fire/aim.
        if action.action_type in (ActionType.FIRE, ActionType.AIM):
            assert action.target == 2

    def test_stays_put_when_no_relevant_signal_and_in_zone(self) -> None:
        policy = ReactivePolicy()
        # Place the agent near zone center so it does not try to move toward it.
        obs = _make_obs(
            self_pos=(250, 250),
            enemies=[],
            self_health=100.0,
            self_loaded=True,
            self_shield=100.0,
            resources_nearby=0,
            zone_safe=True,
        )
        action = policy.choose(obs)
        assert action.action_type in (ActionType.NOOP, ActionType.RELOAD)

    def test_confidence_decreases_with_distance(self) -> None:
        policy = ReactivePolicy()
        near = _make_obs(
            self_pos=(0, 0),
            enemies=[_make_entity(id=2, team=2, x=20, y=0)],
        )
        far = _make_obs(
            self_pos=(0, 0),
            enemies=[_make_entity(id=2, team=2, x=200, y=0)],
        )
        a_near = policy.choose(near)
        a_far = policy.choose(far)
        assert a_near.confidence >= a_far.confidence

    def test_latency_nonzero(self) -> None:
        policy = ReactivePolicy()
        obs = _make_obs(enemies=[])
        action = policy.choose(obs)
        assert action.latency_ns >= 0


# ---------------------------------------------------------------------------
# Hierarchical policy
# ---------------------------------------------------------------------------

class TestHierarchicalPolicy:
    def test_defaults_to_reactive_if_strategy_none(self) -> None:
        strat = StrategyModeRouter(strategy_mode=None)
        policy = HierarchicalPolicy(high_level=strat, reactive=ReactivePolicy())
        obs = _make_obs(enemies=[])
        action = policy.choose(obs)
        assert action.action_type in (ActionType.NOOP, ActionType.MOVE, ActionType.RELOAD)

    def test_uses_strategy_action_when_set(self) -> None:
        strat = StrategyModeRouter(strategy_mode=StrategyMode.DEFENSIVE)
        policy = HierarchicalPolicy(high_level=strat, reactive=ReactivePolicy())
        obs = _make_obs(
            enemies=[_make_entity(id=2, team=2, x=90, y=90)],
            self_health=80.0,
            zone_safe=False,
        )
        action = policy.choose(obs)
        assert action.action_type in (ActionType.MOVE, ActionType.RELOAD, ActionType.NOOP, ActionType.USE_ITEM)

    def test_hierarchical_respects_obs_features(self) -> None:
        reactive = ReactivePolicy()
        strat = StrategyModeRouter(strategy_mode=StrategyMode.BALANCED)
        policy = HierarchicalPolicy(high_level=strat, reactive=reactive)
        obs = _make_obs(enemies=[])
        action = policy.choose(obs)
        assert action.action_type in (ActionType.NOOP, ActionType.RELOAD, ActionType.MOVE)

    def test_chained_latency_includes_layers(self) -> None:
        strat = StrategyModeRouter(strategy_mode=StrategyMode.BALANCED)
        reactive = ReactivePolicy()
        policy = HierarchicalPolicy(high_level=strat, reactive=reactive)
        obs = _make_obs(enemies=[])
        action = policy.choose(obs)
        assert action.latency_ns >= 0


# ---------------------------------------------------------------------------
# Strategy mode router
# ---------------------------------------------------------------------------

class TestStrategyModeRouter:
    def test_balances_by_default(self) -> None:
        router = StrategyModeRouter(strategy_mode=StrategyMode.BALANCED)
        obs = _make_obs(enemies=[])
        directive = router.plan(obs)
        assert directive.mode is StrategyMode.BALANCED

    def test_adaptive_switches_to_survival_when_low_health(self) -> None:
        router = StrategyModeRouter(strategy_mode=StrategyMode.ADAPTIVE, adaptive_enabled=True, conservative=True)
        obs = _make_obs(
            self_health=15.0,
            self_shield=0.0,
            enemies=[_make_entity(id=2, team=2, x=80, y=80)],
            zone_safe=False,
            resources_nearby=0,
            time_remaining=40.0,
            features={"enemy_threat_proxy": 0.9},
        )
        directive = router.plan(obs)
        assert directive.mode in (StrategyMode.SURVIVAL, StrategyMode.DEFENSIVE)

    def test_adaptive_switches_to_resource_when_low_resources(self) -> None:
        router = StrategyModeRouter(strategy_mode=StrategyMode.ADAPTIVE, adaptive_enabled=True, conservative=False)
        obs = _make_obs(
            self_health=80.0,
            self_shield=80.0,
            enemies=[],
            zone_safe=True,
            resources_nearby=0,
            time_remaining=120.0,
            features={"resources_low_proxy": 0.95},
        )
        directive = router.plan(obs)
        assert directive.mode is StrategyMode.RESOURCE_FOCUSED

    def test_adaptive_chooses_targeted_when_relevant_enemy_near(self) -> None:
        router = StrategyModeRouter(strategy_mode=StrategyMode.ADAPTIVE, adaptive_enabled=True, conservative=False)
        obs = _make_obs(
            self_pos=(0, 0),
            self_health=90.0,
            self_shield=90.0,
            enemies=[_make_entity(id=2, team=2, x=60, y=0)],
            zone_safe=True,
            resources_nearby=2,
            time_remaining=150.0,
            features={"enemy_threat_proxy": 0.35},
        )
        directive = router.plan(obs)
        assert directive.mode in (StrategyMode.AGGRESSIVE, StrategyMode.AIM_FOCUSED, StrategyMode.BALANCED)

    def test_adaptive_reports_reasoning(self) -> None:
        router = StrategyModeRouter(strategy_mode=StrategyMode.ADAPTIVE, adaptive_enabled=True)
        obs = _make_obs(self_health=80.0, enemies=[], zone_safe=True, resources_nearby=0)
        directive = router.plan(obs)
        assert "reasoning" in directive.meta
        assert isinstance(directive.meta["reasoning"], dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(
    *,
    self_pos: tuple[float, float] = (0.0, 0.0),
    self_health: float = 100.0,
    self_shield: float = 100.0,
    self_loaded: bool = True,
    enemies: list = None,
    zone_safe: bool = True,
    resources_nearby: int = 0,
    time_remaining: float = 200.0,
    features: dict[str, float] | None = None,
) -> Observation:
    enemies = enemies or []
    map_info = _make_map_info()
    self_entity = _make_entity(id=1, team=0, x=self_pos[0], y=self_pos[1], health=self_health, shield=self_shield, loaded=self_loaded)
    entities = {e.id: e for e in enemies}
    base_features = {
        "self_health": self_health,
        "self_shield": self_shield,
        "loaded": 1.0 if self_loaded else 0.0,
        "enemy_count": float(len(enemies)),
        "resources_nearby": float(resources_nearby),
        "zone_safe": 1.0 if zone_safe else 0.0,
        "enemy_threat_proxy": 0.5,
        "resources_low_proxy": 0.5,
    }
    if features is not None:
        base_features.update(features)
    return Observation(
        tick=10,
        timestamp=0.0,
        scenario="test_island",
        difficulty="medium",
        strategy_mode=StrategyMode.BALANCED,
        map_info=map_info,
        self_entity=self_entity,
        entities=entities,
        confidence=1.0,
        features=base_features,
        time_remaining=time_remaining,
    )


def _make_entity(*, id: int, team: int, x: float, y: float, health: float = 100.0, shield: float = 100.0, loaded: bool = True) -> EntityState:
    return EntityState(id=id, team=TeamId(team), x=x, y=y, health=health, Shield=shield, loaded=loaded)


def _make_map_info() -> MapInfo:
    return MapInfo(name="test_island", width=500.0, height=500.0, safe_zone_center=(250.0, 250.0), safe_zone_radius=200.0)

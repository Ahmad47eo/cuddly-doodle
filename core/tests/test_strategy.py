"""Strategy mode tests: all 14 modes, adaptive selection, switching rules, config validation."""

from __future__ import annotations

import pytest

from core.strategy import (
    ActionPriority,
    AdaptiveStrategyEngine,
    StrategyDirective,
    StrategyMode,
    StrategySettings,
    decide_strategy,
)


# ---------------------------------------------------------------------------
# All 14 modes are present and valid
# ---------------------------------------------------------------------------

class TestAllStrategyModesExist:
    def test_mode_count_is_fourteen(self) -> None:
        modes = list(StrategyMode)
        assert len(modes) == 14
        names = {m.value for m in modes}
        assert "balanced" in names
        assert "adaptive" in names
        assert "custom" in names

    @pytest.mark.parametrize("mode", list(StrategyMode))
    def test_each_mode_has_stable_value(self, mode: StrategyMode) -> None:
        assert isinstance(mode.value, str)
        assert len(mode.value) > 0


# ---------------------------------------------------------------------------
# decide_strategy (deterministic selection)
# ---------------------------------------------------------------------------

class TestDecideStrategy:
    def test_default_returns_balanced(self) -> None:
        directive = decide_strategy(StrategyMode.BALANCED, StrategySettings())
        assert directive.mode is StrategyMode.BALANCED

    def test_balanced_favors_positioning_when_safe(self) -> None:
        df = self._make_state(health=80, shield=80, enemies_near=0, zone_safe=True, resources_low=False, time_remaining=100, base_mode=StrategyMode.BALANCED)
        d = decide_strategy(StrategyMode.BALANCED, df)
        assert d.mode is StrategyMode.BALANCED

    def test_aggressive_selected_via_state(self) -> None:
        df = self._make_state(health=95, shield=95, enemies_near=3, zone_safe=True, resources_low=False, time_remaining=200, base_mode=StrategyMode.AGGRESSIVE, enemy_threat=0.3)
        d = decide_strategy(StrategyMode.AGGRESSIVE, df)
        assert d.mode is StrategyMode.AGGRESSIVE

    def test_defensive_selected_when_threat_high(self) -> None:
        df = self._make_state(health=35, shield=20, enemies_near=3, zone_safe=False, resources_low=True, time_remaining=60, base_mode=StrategyMode.DEFENSIVE, enemy_threat=0.85)
        d = decide_strategy(StrategyMode.DEFENSIVE, df)
        assert d.mode is StrategyMode.DEFENSIVE
        assert d.action_focus.flag(ActionPriority.COVER) is True
        assert d.action_focus.flag(ActionPriority.RETREAT) is True

    def test_survival_selected_when_very_low_health(self) -> None:
        df = self._make_state(health=12, shield=0, enemies_near=2, zone_safe=False, resources_low=True, time_remaining=30, base_mode=StrategyMode.SURVIVAL, enemy_threat=0.9)
        d = decide_strategy(StrategyMode.SURVIVAL, df)
        assert d.mode is StrategyMode.SURVIVAL
        assert d.action_focus.ordinal(ActionPriority.RETREAT) >= d.action_focus.ordinal(ActionPriority.ENGAGE)

    def test_endgame_selected_when_zone_small(self) -> None:
        df = self._make_state(health=60, shield=60, enemies_near=2, zone_safe=False, resources_low=False, time_remaining=40, zone_radius=40.0, base_mode=StrategyMode.ENDGAME, enemy_threat=0.6)
        d = decide_strategy(StrategyMode.ENDGAME, df)
        assert d.mode is StrategyMode.ENDGAME
        assert d.action_focus.flag(ActionPriority.PATIENT) is True

    def test_objective_focused_selected(self) -> None:
        df = self._make_state(health=70, shield=60, enemies_near=1, zone_safe=True, resources_low=False, time_remaining=120, objective_active=True, base_mode=StrategyMode.OBJECTIVE_FOCUSED)
        d = decide_strategy(StrategyMode.OBJECTIVE_FOCUSED, df)
        assert d.mode is StrategyMode.OBJECTIVE_FOCUSED
        assert d.action_focus.flag(ActionPriority.OBJECTIVE) is True

    def test_aim_focused_selected(self) -> None:
        df = self._make_state(health=80, shield=80, enemies_near=2, zone_safe=True, resources_low=False, time_remaining=150, base_mode=StrategyMode.AIM_FOCUSED, enemy_threat=0.4)
        d = decide_strategy(StrategyMode.AIM_FOCUSED, df)
        assert d.mode is StrategyMode.AIM_FOCUSED
        assert d.action_focus.flag(ActionPriority.ENGAGE) is True

    def test_resource_focused_selected_when_low(self) -> None:
        df = self._make_state(health=60, shield=60, enemies_near=1, zone_safe=True, resources_low=True, time_remaining=120, base_mode=StrategyMode.RESOURCE_FOCUSED)
        d = decide_strategy(StrategyMode.RESOURCE_FOCUSED, df)
        assert d.mode is StrategyMode.RESOURCE_FOCUSED
        assert d.action_focus.flag(ActionPriority.LOOT) is True
        assert d.action_focus.flag(ActionPriority.RETREAT) is True

    def test_team_support_selected(self) -> None:
        df = self._make_state(health=80, shield=80, enemies_near=1, zone_safe=True, resources_low=False, time_remaining=120, ally_in_danger=True, base_mode=StrategyMode.TEAM_SUPPORT)
        d = decide_strategy(StrategyMode.TEAM_SUPPORT, df)
        assert d.mode is StrategyMode.TEAM_SUPPORT
        assert d.action_focus.flag(ActionPriority.SUPPORT) is True

    def test_build_edit_selected(self) -> None:
        df = self._make_state(health=75, shield=70, enemies_near=1, zone_safe=True, resources_low=False, time_remaining=100, under_fire=True, base_mode=StrategyMode.BUILD_EDIT_FOCUSED)
        d = decide_strategy(StrategyMode.BUILD_EDIT_FOCUSED, df)
        assert d.mode is StrategyMode.BUILD_EDIT_FOCUSED
        assert d.action_focus.flag(ActionPriority.BUILD) is True
        assert d.action_focus.flag(ActionPriority.COVER) is True

    def test_mobility_focused_selected(self) -> None:
        df = self._make_state(health=60, shield=60, enemies_near=2, zone_safe=False, resources_low=False, time_remaining=90, base_mode=StrategyMode.MOBILITY_FOCUSED, enemy_threat=0.5)
        d = decide_strategy(StrategyMode.MOBILITY_FOCUSED, df)
        assert d.mode is StrategyMode.MOBILITY_FOCUSED
        assert d.action_focus.flag(ActionPriority.MOVE) is True

    def test_competitive_selected(self) -> None:
        df = self._make_state(health=90, shield=90, enemies_near=2, zone_safe=True, resources_low=False, time_remaining=200, base_mode=StrategyMode.COMPETITIVE, enemy_threat=0.4)
        d = decide_strategy(StrategyMode.COMPETITIVE, df)
        assert d.mode is StrategyMode.COMPETITIVE

    def test_custom_passes_through(self) -> None:
        df = self._make_state(health=80, shield=80, enemies_near=0, zone_safe=True, resources_low=False, time_remaining=100, base_mode=StrategyMode.CUSTOM)
        d = decide_strategy(StrategyMode.CUSTOM, df)
        assert d.mode is StrategyMode.CUSTOM

    def test_invalid_base_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid StrategyMode"):
            decide_strategy(StrategyMode(-99), StrategySettings())  # type: ignore

    def test_settings_validation(self) -> None:
        # aggression and defense_sensitivity must be in [0, 1]
        with pytest.raises(ValueError, match="in \[0, 1\]"):
            StrategySettings(aggression=-1)
        with pytest.raises(ValueError, match="in \[0, 1\]"):
            StrategySettings(defense_sensitivity=-0.5)
        with pytest.raises(ValueError, match="in \[0, 1\]"):
            StrategySettings(enemy_threat=1.2)
        # health_low must be > health_critical
        with pytest.raises(ValueError, match="health_low must be > health_critical"):
            StrategySettings(health_low=1.5)
        with pytest.raises(ValueError, match="health_low must be > health_critical"):
            StrategySettings(health_critical=50.0)
        # threat_high must be < threat_critical
        with pytest.raises(ValueError, match="threat_high must be < threat_critical"):
            StrategySettings(threat_high=0.95)
        with pytest.raises(ValueError, match="threat_high must be < threat_critical"):
            StrategySettings(threat_critical=0.3)
        # endgame_zone_radius must be > 0
        with pytest.raises(ValueError, match="must be >= 0"):
            StrategySettings(endgame_zone_radius=-1.0)
        # resource_low must be bool
        with pytest.raises(ValueError, match="must be bool"):
            StrategySettings(resources_low="yes")
        # adaptive_min_confidence must be in [0, 1] when adaptive is enabled
        with pytest.raises(ValueError, match="adaptive_min_confidence must be in"):
            StrategySettings(adaptive_enabled=True, adaptive_min_confidence=-0.1)
        with pytest.raises(ValueError, match="adaptive_min_confidence must be in"):
            StrategySettings(adaptive_enabled=True, adaptive_min_confidence=1.1)
        # base_mode must be a StrategyMode
        with pytest.raises(ValueError, match="base_mode must be a StrategyMode"):
            StrategySettings(base_mode="not a mode")

    # ---- helpers ----

    @staticmethod
    def _make_state(
        *,
        health: float,
        shield: float,
        enemies_near: int,
        zone_safe: bool,
        resources_low: bool,
        time_remaining: float,
        zone_radius: float = 200.0,
        enemy_threat: float = 0.5,
        ally_in_danger: bool = False,
        under_fire: bool = False,
        objective_active: bool = False,
        base_mode: StrategyMode = StrategyMode.BALANCED,
    ) -> StrategySettings:
        return StrategySettings(
            health=health,
            shield=shield,
            enemies_near=enemies_near,
            zone_safe=zone_safe,
            resources_low=resources_low,
            time_remaining=time_remaining,
            zone_radius=zone_radius,
            enemy_threat=enemy_threat,
            ally_in_danger=ally_in_danger,
            under_fire=under_fire,
            objective_active=objective_active,
            base_mode=base_mode,
        )


# ---------------------------------------------------------------------------
# Adaptive strategy engine
# ---------------------------------------------------------------------------

class TestAdaptiveStrategyEngine:
    def test_no_data_returns_plan_indicating_no_history(self) -> None:
        engine = AdaptiveStrategyEngine(conservative=True)
        plan = engine.select_history_based(
            current=StrategySettings(base_mode=StrategyMode.ADAPTIVE),
            modes=[StrategyMode.BALANCED],
        )
        # With no history the engine cannot prefer one mode over another;
        # it returns a plan that falls back to the current base mode and
        # reports that there is no historical data.
        assert plan.selected_mode is StrategyMode.BALANCED
        assert "no historical data" in plan.reasoning.get("reason", "")
        assert plan.reasoning.get("no_historical_data") is True

    def test_identical_histories_selects_most_common(self) -> None:
        engine = AdaptiveStrategyEngine(conservative=True)
        history = [
            {"mode": StrategyMode.AGGRESSIVE, "reward": 10.0, "survived": True, "latency_ns": 1000, "success": True},
            {"mode": StrategyMode.AGGRESSIVE, "reward": 9.0, "survived": True, "latency_ns": 1100, "success": True},
            {"mode": StrategyMode.DEFENSIVE, "reward": 5.0, "survived": True, "latency_ns": 1200, "success": False},
        ]
        current = StrategySettings(base_mode=StrategyMode.ADAPTIVE)
        plan = engine.select_history_based(current=current, modes=[StrategyMode.AGGRESSIVE, StrategyMode.DEFENSIVE], history=history)
        assert plan.selected_mode is StrategyMode.AGGRESSIVE
        assert "aggressive_reward" in plan.reasoning.get("summary", "")

    def test_defeated_mode_demoted(self) -> None:
        engine = AdaptiveStrategyEngine(conservative=True)
        history = [
            {"mode": StrategyMode.AGGRESSIVE, "reward": -50.0, "survived": False, "latency_ns": 1000, "success": False},
            {"mode": StrategyMode.BALANCED, "reward": 8.0, "survived": True, "latency_ns": 1100, "success": True},
        ]
        current = StrategySettings(base_mode=StrategyMode.ADAPTIVE)
        plan = engine.select_history_based(current=current, modes=[StrategyMode.AGGRESSIVE, StrategyMode.BALANCED], history=history)
        assert plan.selected_mode is StrategyMode.BALANCED
        assert "defeated_mode" in plan.reasoning

    def test_empty_history_reverts_to_default(self) -> None:
        engine = AdaptiveStrategyEngine(conservative=False)
        current = StrategySettings(base_mode=StrategyMode.BALANCED)
        plan = engine.select_history_based(current=current, modes=[], history=[])
        assert plan.selected_mode is StrategyMode.BALANCED
        assert "no_historical_data" in plan.reasoning

    def test_plan_is_immutable_like(self) -> None:
        engine = AdaptiveStrategyEngine(conservative=True)
        history = [
            {"mode": StrategyMode.DEFENSIVE, "reward": 8.0, "survived": True, "latency_ns": 1000, "success": True},
        ]
        current = StrategySettings(base_mode=StrategyMode.ADAPTIVE)
        plan = engine.select_history_based(current=current, modes=[StrategyMode.DEFENSIVE], history=history)
        assert plan.selected_mode is StrategyMode.DEFENSIVE
        assert isinstance(plan.reasoning, dict)
        assert isinstance(plan.metrics, dict)

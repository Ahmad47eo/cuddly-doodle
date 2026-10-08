"""Strategy system for simulated gameplay AI.

Implements 14 configurable pro-style strategy modes and a deterministic
adaptive mode selector that uses structured game-state signals (not
real-game automation) to pick the most suitable strategy.

Modes:
    BALANCED, AGGRESSIVE, DEFENSIVE, COMPETITIVE, AIM_FOCUSED,
    MOBILITY_FOCUSED, BUILD_EDIT_FOCUSED, TEAM_SUPPORT, RESOURCE_FOCUSED,
    ENDGAME, SURVIVAL, OBJECTIVE_FOCUSED, ADAPTIVE, CUSTOM
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from core.types import StrategyMode, TeamId

# ---------------------------------------------------------------------------
# Action priorities used by strategies to express tactical emphasis
# ---------------------------------------------------------------------------

class ActionPriority(str, Enum):
    NOOP = "noop"
    MOVE = "move"
    ROTATE = "rotate"
    AIM = "aim"
    FIRE = "fire"
    RELOAD = "reload"
    BUILD = "build"
    EDIT = "edit"
    LOOT = "loot"
    COVER = "cover"
    RETREAT = "retreat"
    ENGAGE = "engage"
    SUPPORT = "support"
    OBJECTIVE = "objective"
    PATIENT = "patient"
    PLAN = "plan"
    RISK_MGMT = "risk_mgmt"


@dataclass
class ActionFocus:
    """Per-mode emphasis per action priority (higher = stronger emphasis)."""

    priorities: Mapping[ActionPriority, int] = field(default_factory=dict)

    def flag(self, priority: ActionPriority) -> bool:
        return self.priorities.get(priority, 0) >= 2

    def ordinal(self, priority: ActionPriority) -> int:
        return self.priorities.get(priority, 0)


# ---------------------------------------------------------------------------
# Per-mode strategy settings
# ---------------------------------------------------------------------------

_STRATEGY_PRESETS: dict[StrategyMode, ActionFocus] = {
    StrategyMode.BALANCED: ActionFocus({
        ActionPriority.MOVE: 2,
        ActionPriority.AIM: 2,
        ActionPriority.FIRE: 1,
        ActionPriority.RELOAD: 1,
        ActionPriority.LOOT: 1,
        ActionPriority.COVER: 1,
        ActionPriority.PATIENT: 1,
        ActionPriority.PLAN: 1,
    }),
    StrategyMode.AGGRESSIVE: ActionFocus({
        ActionPriority.MOVE: 2,
        ActionPriority.AIM: 2,
        ActionPriority.FIRE: 3,
        ActionPriority.RELOAD: 1,
        ActionPriority.ENGAGE: 3,
        ActionPriority.PATIENT: 0,
        ActionPriority.PLAN: 1,
    }),
    StrategyMode.DEFENSIVE: ActionFocus({
        ActionPriority.MOVE: 2,
        ActionPriority.AIM: 2,
        ActionPriority.RELOAD: 2,
        ActionPriority.COVER: 3,
        ActionPriority.RETREAT: 3,
        ActionPriority.PATIENT: 3,
        ActionPriority.PLAN: 2,
        ActionPriority.BUILD: 2,
    }),
    StrategyMode.COMPETITIVE: ActionFocus({
        ActionPriority.MOVE: 2,
        ActionPriority.AIM: 2,
        ActionPriority.FIRE: 2,
        ActionPriority.RELOAD: 2,
        ActionPriority.COVER: 2,
        ActionPriority.PATIENT: 2,
        ActionPriority.OBJECTIVE: 2,
        ActionPriority.RISK_MGMT: 2,
        ActionPriority.PLAN: 2,
    }),
    StrategyMode.AIM_FOCUSED: ActionFocus({
        ActionPriority.AIM: 3,
        ActionPriority.FIRE: 3,
        ActionPriority.RELOAD: 1,
        ActionPriority.ENGAGE: 2,
        ActionPriority.OBJECTIVE: 1,
        ActionPriority.COVER: 2,
        ActionPriority.MOVE: 2,
    }),
    StrategyMode.MOBILITY_FOCUSED: ActionFocus({
        ActionPriority.MOVE: 3,
        ActionPriority.ROTATE: 2,
        ActionPriority.RELOAD: 1,
        ActionPriority.COVER: 2,
        ActionPriority.PATIENT: 1,
        ActionPriority.OBJECTIVE: 1,
        ActionPriority.PLAN: 1,
    }),
    StrategyMode.BUILD_EDIT_FOCUSED: ActionFocus({
        ActionPriority.BUILD: 3,
        ActionPriority.EDIT: 3,
        ActionPriority.MOVE: 2,
        ActionPriority.AIM: 2,
        ActionPriority.COVER: 3,
        ActionPriority.PLAN: 2,
        ActionPriority.RELOAD: 1,
    }),
    StrategyMode.TEAM_SUPPORT: ActionFocus({
        ActionPriority.MOVE: 2,
        ActionPriority.AIM: 2,
        ActionPriority.FIRE: 1,
        ActionPriority.RELOAD: 1,
        ActionPriority.RETREAT: 2,
        ActionPriority.SUPPORT: 3,
        ActionPriority.COVER: 3,
        ActionPriority.PLAN: 2,
    }),
    StrategyMode.RESOURCE_FOCUSED: ActionFocus({
        ActionPriority.LOOT: 3,
        ActionPriority.RELOAD: 3,
        ActionPriority.MOVE: 2,
        ActionPriority.COVER: 2,
        ActionPriority.RETREAT: 2,
        ActionPriority.PATIENT: 2,
        ActionPriority.PLAN: 2,
    }),
    StrategyMode.ENDGAME: ActionFocus({
        ActionPriority.COVER: 3,
        ActionPriority.PATIENT: 3,
        ActionPriority.MOVE: 2,
        ActionPriority.AIM: 2,
        ActionPriority.FIRE: 2,
        ActionPriority.RELOAD: 2,
        ActionPriority.LOOT: 1,
        ActionPriority.OBJECTIVE: 2,
        ActionPriority.PLAN: 2,
    }),
    StrategyMode.SURVIVAL: ActionFocus({
        ActionPriority.COVER: 3,
        ActionPriority.RETREAT: 3,
        ActionPriority.LOOT: 2,
        ActionPriority.RELOAD: 2,
        ActionPriority.MOVE: 2,
        ActionPriority.PATIENT: 3,
        ActionPriority.PLAN: 2,
        ActionPriority.OBJECTIVE: 1,
    }),
    StrategyMode.OBJECTIVE_FOCUSED: ActionFocus({
        ActionPriority.OBJECTIVE: 3,
        ActionPriority.MOVE: 2,
        ActionPriority.AIM: 2,
        ActionPriority.FIRE: 2,
        ActionPriority.COVER: 2,
        ActionPriority.PLAN: 2,
        ActionPriority.PATIENT: 1,
        ActionPriority.RETREAT: 1,
    }),
    StrategyMode.ADAPTIVE: ActionFocus({}),  # computed dynamically
    StrategyMode.CUSTOM: ActionFocus({}),    # user-defined defaults here
}


# ---------------------------------------------------------------------------
# Public strategy decision types
# ---------------------------------------------------------------------------

@dataclass
class StrategyDirective:
    mode: StrategyMode
    action_focus: ActionFocus
    reasoning: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "action_focus": {p.value: v for p, v in self.action_focus.priorities.items()},
            "reasoning": self.reasoning,
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# Strategy state / settings (used by adaptive selection)
# ---------------------------------------------------------------------------

@dataclass
class StrategySettings:
    """Structured state signals for strategy selection.

    Populated by the observation pipeline and/or simulator from permitted
    game-state data — never from real-game automation.
    """

    health: float = 100.0
    shield: float = 100.0
    enemies_near: int = 0
    enemy_threat: float = 0.5
    zone_safe: bool = True
    zone_radius: float = 200.0
    resources_low: bool = False
    time_remaining: float = 200.0
    objective_active: bool = False
    ally_in_danger: bool = False
    under_fire: bool = False

    base_mode: StrategyMode = StrategyMode.BALANCED

    aggression: float = 0.6
    defense_sensitivity: float = 0.5
    health_low: float = 45.0
    health_critical: float = 20.0
    threat_high: float = 0.65
    threat_critical: float = 0.85
    endgame_zone_radius: float = 80.0
    adaptive_enabled: bool = False
    adaptive_min_confidence: float = 0.5
    adaptive_history: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool):
                # bools are subtypes of int; handle explicitly to avoid misclassifying.
                continue
            if isinstance(value, (int, float)):
                if name in ("aggression", "defense_sensitivity", "enemy_threat", "adaptive_min_confidence"):
                    fvalue = float(value)
                    if not 0.0 <= fvalue <= 1.0:
                        raise ValueError(f"{name} must be in [0, 1]")
                elif name in ("health_low", "health_critical", "threat_high", "threat_critical", "endgame_zone_radius", "time_remaining"):
                    if float(value) < 0:
                        raise ValueError(f"{name} must be >= 0")
            elif name in ("resources_low", "zone_safe", "objective_active", "ally_in_danger", "under_fire", "adaptive_enabled"):
                raise ValueError(f"{name} must be bool, not {type(value).__name__}")
            elif name == "adaptive_history" and value is not None and not isinstance(value, list):
                raise ValueError("adaptive_history must be a list or None")
            elif name == "base_mode" and not isinstance(value, StrategyMode):
                raise ValueError("base_mode must be a StrategyMode")
        if self.health_low <= self.health_critical:
            raise ValueError("health_low must be > health_critical")
        if self.threat_high >= self.threat_critical:
            raise ValueError("threat_high must be < threat_critical")
        if self.adaptive_enabled and (self.adaptive_min_confidence < 0.0 or self.adaptive_min_confidence > 1.0):
            raise ValueError("adaptive_min_confidence must be in [0, 1] when adaptive is enabled")


# ---------------------------------------------------------------------------
# Deterministic strategy selection
# ---------------------------------------------------------------------------

def decide_strategy(
    base_mode: StrategyMode,
    settings: StrategySettings | None = None,
) -> StrategyDirective:
    settings = settings or StrategySettings()
    if base_mode is StrategyMode.CUSTOM:
        return StrategyDirective(
            mode=StrategyMode.CUSTOM,
            action_focus=_STRATEGY_PRESETS.get(StrategyMode.CUSTOM, ActionFocus({})),
            reasoning="CUSTOM mode — user-defined behavior expected",
            meta={"category": StateCategory.NORMAL.value},
        )
    if base_mode is StrategyMode.ADAPTIVE and settings.adaptive_enabled:
        return _adaptive_choice(settings)
    # For fixed strategy modes, use that mode's own preset focus so the
    # strategy semantics are preserved (the reactive policy can still override
    # based on immediate state). For ADAPTIVE without history, fall back to
    # category-based focus.
    focus = _STRATEGY_PRESETS.get(base_mode, _stable_focus_for(_classify_state(settings)))
    category = _classify_state(settings)
    reasoning = _reasoning_for(category, settings)
    return StrategyDirective(
        mode=base_mode,
        action_focus=focus,
        reasoning=reasoning,
        meta={"category": category.value},
    )


# ---------------------------------------------------------------------------
# Internal: state classification and focus selection
# ---------------------------------------------------------------------------

class StateCategory(str, Enum):
    NORMAL = "normal"
    AGGRESSIVE_OK = "aggressive_ok"
    DEFENSIVE_NEEDED = "defensive_needed"
    SURVIVAL = "survival"
    ENDGAME = "endgame"
    RESOURCE_NEEDED = "resource_needed"
    TEAM_SUPPORT_NEEDED = "team_support_needed"
    OBJECTIVE_DRIVEN = "objective_driven"


def _classify_state(s: StrategySettings) -> StateCategory:
    if s.health <= s.health_critical or s.shield <= 5.0:
        return StateCategory.SURVIVAL
    if s.zone_radius < s.endgame_zone_radius and s.health > s.health_low:
        return StateCategory.ENDGAME
    if s.resources_low and s.enemies_near <= 1:
        return StateCategory.RESOURCE_NEEDED
    if s.ally_in_danger or (s.enemies_near >= 2 and s.under_fire):
        return StateCategory.TEAM_SUPPORT_NEEDED
    if s.objective_active and s.enemy_threat < s.threat_high:
        return StateCategory.OBJECTIVE_DRIVEN
    if s.enemy_threat >= s.threat_critical or (s.enemies_near >= 2 and s.under_fire and s.health < s.health_low):
        return StateCategory.DEFENSIVE_NEEDED
    if s.enemy_threat > s.threat_high and s.health > s.health_low and s.aggression > 0.5:
        return StateCategory.AGGRESSIVE_OK
    return StateCategory.NORMAL


def _stable_focus_for(category: StateCategory) -> ActionFocus:
    if category is StateCategory.SURVIVAL:
        return ActionFocus({
            ActionPriority.COVER: 3,
            ActionPriority.RETREAT: 3,
            ActionPriority.LOOT: 2,
            ActionPriority.RELOAD: 2,
            ActionPriority.MOVE: 2,
            ActionPriority.PATIENT: 3,
            ActionPriority.PLAN: 2,
            ActionPriority.OBJECTIVE: 1,
        })
    if category is StateCategory.ENDGAME:
        return ActionFocus({
            ActionPriority.COVER: 3,
            ActionPriority.PATIENT: 3,
            ActionPriority.MOVE: 2,
            ActionPriority.AIM: 2,
            ActionPriority.FIRE: 2,
            ActionPriority.RELOAD: 2,
            ActionPriority.LOOT: 1,
            ActionPriority.OBJECTIVE: 2,
            ActionPriority.PLAN: 2,
        })
    if category is StateCategory.RESOURCE_NEEDED:
        return ActionFocus({
            ActionPriority.LOOT: 3,
            ActionPriority.RELOAD: 3,
            ActionPriority.MOVE: 2,
            ActionPriority.COVER: 2,
            ActionPriority.RETREAT: 2,
            ActionPriority.PATIENT: 2,
            ActionPriority.PLAN: 2,
        })
    if category is StateCategory.TEAM_SUPPORT_NEEDED:
        return ActionFocus({
            ActionPriority.MOVE: 2,
            ActionPriority.AIM: 2,
            ActionPriority.FIRE: 1,
            ActionPriority.RELOAD: 1,
            ActionPriority.RETREAT: 2,
            ActionPriority.SUPPORT: 3,
            ActionPriority.COVER: 3,
            ActionPriority.PLAN: 2,
        })
    if category is StateCategory.OBJECTIVE_DRIVEN:
        return ActionFocus({
            ActionPriority.OBJECTIVE: 3,
            ActionPriority.MOVE: 2,
            ActionPriority.AIM: 2,
            ActionPriority.FIRE: 2,
            ActionPriority.COVER: 2,
            ActionPriority.PLAN: 2,
            ActionPriority.PATIENT: 1,
            ActionPriority.RETREAT: 1,
        })
    if category is StateCategory.DEFENSIVE_NEEDED:
        return ActionFocus({
            ActionPriority.MOVE: 2,
            ActionPriority.AIM: 2,
            ActionPriority.RELOAD: 2,
            ActionPriority.COVER: 3,
            ActionPriority.RETREAT: 3,
            ActionPriority.PATIENT: 3,
            ActionPriority.PLAN: 2,
            ActionPriority.BUILD: 2,
        })
    if category is StateCategory.AGGRESSIVE_OK:
        return ActionFocus({
            ActionPriority.MOVE: 2,
            ActionPriority.AIM: 2,
            ActionPriority.FIRE: 3,
            ActionPriority.RELOAD: 1,
            ActionPriority.ENGAGE: 3,
            ActionPriority.PATIENT: 0,
            ActionPriority.PLAN: 1,
        })
    return ActionFocus({
        ActionPriority.MOVE: 2,
        ActionPriority.AIM: 2,
        ActionPriority.FIRE: 1,
        ActionPriority.RELOAD: 1,
        ActionPriority.LOOT: 1,
        ActionPriority.COVER: 1,
        ActionPriority.PATIENT: 1,
        ActionPriority.PLAN: 1,
    })


def _category_mode_preference(category: StateCategory) -> list[StrategyMode]:
    # Ordered preference of modes for a given state category. Used by the
    # adaptive engine as a fallback when there is no historical data.
    if category is StateCategory.SURVIVAL:
        return [StrategyMode.SURVIVAL, StrategyMode.DEFENSIVE, StrategyMode.ENDGAME, StrategyMode.BALANCED]
    if category is StateCategory.ENDGAME:
        return [StrategyMode.ENDGAME, StrategyMode.DEFENSIVE, StrategyMode.SURVIVAL, StrategyMode.BALANCED]
    if category is StateCategory.RESOURCE_NEEDED:
        return [StrategyMode.RESOURCE_FOCUSED, StrategyMode.BALANCED, StrategyMode.DEFENSIVE]
    if category is StateCategory.TEAM_SUPPORT_NEEDED:
        return [StrategyMode.TEAM_SUPPORT, StrategyMode.DEFENSIVE, StrategyMode.BALANCED]
    if category is StateCategory.OBJECTIVE_DRIVEN:
        return [StrategyMode.OBJECTIVE_FOCUSED, StrategyMode.BALANCED, StrategyMode.AGGRESSIVE]
    if category is StateCategory.DEFENSIVE_NEEDED:
        return [StrategyMode.DEFENSIVE, StrategyMode.SURVIVAL, StrategyMode.ENDGAME, StrategyMode.BALANCED]
    if category is StateCategory.AGGRESSIVE_OK:
        return [StrategyMode.AGGRESSIVE, StrategyMode.AIM_FOCUSED, StrategyMode.BALANCED]
    return [StrategyMode.BALANCED]


def _reasoning_for(category: StateCategory, s: StrategySettings) -> str:
    parts: list[str] = []
    if category is StateCategory.SURVIVAL:
        parts.append(f"low health ({s.health:.0f}) and/or low shield — prioritize survival")
    elif category is StateCategory.ENDGAME:
        parts.append(f"small zone (r={s.zone_radius:.0f}) — patient positioning favors endgame")
    elif category is StateCategory.RESOURCE_NEEDED:
        parts.append("resources low — prioritize loot/reload")
    elif category is StateCategory.TEAM_SUPPORT_NEEDED:
        parts.append("ally in danger and/or under fire with multiple enemies — support/cover")
    elif category is StateCategory.OBJECTIVE_DRIVEN:
        parts.append("objective active and threat manageable — prioritize objective")
    elif category is StateCategory.DEFENSIVE_NEEDED:
        parts.append(f"high threat ({s.enemy_threat:.2f}) and/or multiple enemies under fire — defensive posture")
    elif category is StateCategory.AGGRESSIVE_OK:
        parts.append(f"threat manageable ({s.enemy_threat:.2f}) and health sufficient — can afford aggression")
    else:
        parts.append("conditions normal — balanced posture")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Adaptive mode (history + current-state-based selection)
# ---------------------------------------------------------------------------

@dataclass
class AdaptivePlan:
    selected_mode: StrategyMode
    reasoning: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)


class AdaptiveStrategyEngine:
    """Deterministic adaptive strategy selector.

    Uses a small history of per-mode outcomes (reward, survived, latency,
    success) plus the current state category to pick a mode. No randomness
    in release paths; behaviour is explainable and reproducible.
    """

    def __init__(self, *, conservative: bool = True) -> None:
        self.conservative = conservative

    def select_history_based(
        self,
        *,
        current: StrategySettings,
        modes: list[StrategyMode],
        history: list[dict[str, Any]] | None = None,
    ) -> AdaptivePlan:
        history = history or []
        if not modes:
            return AdaptivePlan(
                selected_mode=current.base_mode,
                reasoning={"reason": "no historical data — no candidate modes provided", "no_historical_data": True},
                metrics={"history_modes": 0},
            )

        agg: dict[str, dict[str, Any]] = {}
        for entry in history:
            mode = entry.get("mode")
            if not mode:
                continue
            key = mode
            if key not in agg:
                agg[key] = {"total_reward": 0.0, "survived": 0, "attempts": 0, "total_latency": 0, "successes": 0}
            a = agg[key]
            a["total_reward"] += float(entry.get("reward", 0.0))
            a["attempts"] += 1
            if entry.get("survived", False):
                a["survived"] += 1
            a["total_latency"] += int(entry.get("latency_ns", 0))
            if entry.get("success", False):
                a["successes"] += 1

        if not agg:
            # No historical data for any candidate mode. When adaptive is on,
            # fall back to the current state category's recommended mode rather
            # than blindly using the base mode (which may be ADAPTIVE itself).
            category = _classify_state(current)
            fallback = self._mode_for_category(category, modes) if modes else current.base_mode
            return AdaptivePlan(
                selected_mode=fallback,
                reasoning={"reason": "no historical data — using current state category", "no_historical_data": True, "category": category.value},
                metrics={"history_modes": len(modes)},
            )

        current_category = _classify_state(current)

        scores: dict[str, float] = {}
        for mode in modes:
            key = mode
            if key not in agg:
                scores[key] = self._mode_base_score(mode, current_category, conservative=self.conservative)
                continue
            a = agg[key]
            attempts = max(1, a["attempts"])
            reward_rate = a["total_reward"] / attempts
            survival_rate = a["survived"] / attempts
            success_rate = a["successes"] / attempts
            avg_latency = a["total_latency"] / attempts if a["total_latency"] else 0
            score = (
                reward_rate * 1.0
                + survival_rate * 50.0
                + success_rate * 30.0
                - (avg_latency / 100000.0) * 10.0
            )
            if self.conservative and not a.get("survived_last", True):
                score -= 40.0
            scores[key] = score

        best_mode = self._best_mode(scores, modes, current_category)

        # Build per-mode summary strings for observability/tests.
        summary_parts: list[str] = []
        for m in modes:
            key = m.value
            if key in agg:
                a = agg[key]
                attempts = max(1, a["attempts"])
                summary_parts.append(f"{key}_reward={round(a['total_reward']/attempts, 2)}")
        reasoning = {
            "selected_mode": best_mode.value,
            "scores": {m.value: round(scores.get(m.value, 0.0), 3) for m in modes},
            "current_category": current_category.value,
            "history_modes": list(agg.keys()),
            "summary": ", ".join(summary_parts),
        }
        # Include human-readable reason fragments the tests assert on.
        if any(agg.get(m.value, {}).get("successes", 0) == 0 and agg[m.value].get("attempts", 0) > 0 for m in modes if m.value in agg):
            defeated = [m.value for m in modes if m.value in agg and agg[m.value].get("successes", 0) == 0 and agg[m.value].get("attempts", 0) > 0]
            if defeated:
                reasoning["defeated_mode"] = defeated[0]
        return AdaptivePlan(
            selected_mode=best_mode,
            reasoning=reasoning,
            metrics={"history_count": len(history), "candidate_modes": len(modes)},
        )

    def _mode_base_score(self, mode: StrategyMode, category: StateCategory, *, conservative: bool) -> float:
        base = 10.0
        if conservative:
            if category is StateCategory.SURVIVAL and mode in (StrategyMode.SURVIVAL, StrategyMode.DEFENSIVE):
                base += 20.0
            elif category is StateCategory.ENDGAME and mode is StrategyMode.ENDGAME:
                base += 20.0
            elif category is StateCategory.DEFENSIVE_NEEDED and mode is StrategyMode.DEFENSIVE:
                base += 15.0
        return base

    def _mode_for_category(self, category: StateCategory, modes: list[StrategyMode]) -> StrategyMode:
        # Pick the mode that best fits the current state category, falling back
        # to BALANCED if none of the candidates match.
        preference = _category_mode_preference(category)
        for preferred in preference:
            if preferred in modes:
                return preferred
        return modes[0] if modes else StrategyMode.BALANCED

    def _best_mode(self, scores: dict[str, float], modes: list[StrategyMode], category: StateCategory) -> StrategyMode:
        ranked = sorted(modes, key=lambda m: (-scores.get(m.value, 0.0), m.value))
        best = ranked[0]
        best_score = scores.get(best.value, 0.0)
        if best_score < 0 and category is StateCategory.SURVIVAL:
            if StrategyMode.SURVIVAL in modes:
                return StrategyMode.SURVIVAL
            if StrategyMode.DEFENSIVE in modes:
                return StrategyMode.DEFENSIVE
        return best


# ---------------------------------------------------------------------------
# StrategyModeRouter — high-level strategy selection for the hierarchical policy
# ---------------------------------------------------------------------------

class StrategyModeRouter:
    """High-level strategy selector used by the hierarchical policy.

    Reads structured signals from the observation and returns a
    StrategyDirective with a mode and ActionFocus. In ADAPTIVE mode,
    uses deterministic rule-based switching grounded in game state.
    """

    def __init__(
        self,
        *,
        strategy_mode: StrategyMode = StrategyMode.BALANCED,
        adaptive_enabled: bool = False,
        conservative: bool = True,
        engine: AdaptiveStrategyEngine | None = None,
    ) -> None:
        self.strategy_mode = strategy_mode
        self.adaptive_enabled = adaptive_enabled
        self.conservative = conservative
        self.engine = engine or AdaptiveStrategyEngine(conservative=conservative)
        self.last_latency_ns: int = 0

    def plan(self, obs: Observation) -> StrategyDirective:
        t0 = __import__("time").perf_counter()
        try:
            settings = self._settings_from_obs(obs)
            if self.adaptive_enabled and self.strategy_mode is StrategyMode.ADAPTIVE:
                plan = self.engine.select_history_based(
                    current=settings,
                    modes=list(StrategyMode),
                    history=settings.adaptive_history,
                )
                return self._directive_from_plan(plan, obs)
            return decide_strategy(self.strategy_mode, settings)
        finally:
            self.last_latency_ns = int((__import__("time").perf_counter() - t0) * 1e9)

    def _settings_from_obs(self, obs: Observation) -> StrategySettings:
        self_entity = obs.self_entity
        enemies_near = sum(
            1 for e in obs.entities.values()
            if e.team == TeamId.ENEMY and e.health > 0
            and math.hypot(e.x - self_entity.x, e.y - self_entity.y) < 150.0
        )
        features = obs.features
        return StrategySettings(
            health=float(self_entity.health),
            shield=float(self_entity.Shield),
            enemies_near=enemies_near,
            enemy_threat=float(features.get("enemy_threat_proxy", 0.5)),
            zone_safe=bool(obs.map_info.safe_zone_radius > 0),
            zone_radius=float(obs.map_info.safe_zone_radius),
            resources_low=bool(not self_entity.loaded or features.get("resources_low_proxy", 0.0) > 0.7),
            time_remaining=float(obs.features.get("time_remaining", 200.0)) if "time_remaining" in obs.features else 200.0,
            objective_active=bool(features.get("objective_active", 0.0) > 0.5),
            ally_in_danger=bool(sum(1 for e in obs.entities.values() if e.team == TeamId.ALLY and e.health < 50.0) > 0),
            under_fire=bool(any("shot" in ev.lower() for ev in obs.events[-4:])),
            base_mode=self.strategy_mode,
            adaptive_enabled=self.adaptive_enabled,
            adaptive_history=obs.features.get("adaptive_history") or [],
        )

    def _directive_from_plan(self, plan: AdaptivePlan, obs: Observation) -> StrategyDirective:
        focus = self._focus_for_plan(plan)
        return StrategyDirective(
            mode=plan.selected_mode,
            action_focus=focus,
            reasoning=plan.reasoning.get("reason", f"adaptive selected {plan.selected_mode.value}"),
            meta={
                "reasoning": plan.reasoning,
                "metrics": plan.metrics,
                "router_latency_ns": self.last_latency_ns,
            },
        )

    def _focus_for_plan(self, plan: AdaptivePlan) -> ActionFocus:
        # Map the selected adaptive mode to its preset focus or a default.
        if plan.selected_mode in _STRATEGY_PRESETS:
            return _STRATEGY_PRESETS[plan.selected_mode]
        return _STRATEGY_PRESETS.get(StrategyMode.BALANCED, ActionFocus({}))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ActionFocus",
    "ActionPriority",
    "AdaptivePlan",
    "AdaptiveStrategyEngine",
    "StateCategory",
    "StrategyDirective",
    "StrategyModeRouter",
    "StrategySettings",
    "decide_strategy",
]

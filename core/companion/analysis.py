"""Real-game observation-only analysis / coaching.

This module runs on PERMITTED observation inputs only:
- User-provided screenshots
- User-provided videos
- Pre-recorded gameplay
- Structured telemetry the user explicitly supplied
- Simulator state (when used inside the simulator)

It may provide strategy suggestions, positioning advice, decision analysis,
aim analysis, rotation analysis, post-game review, performance statistics,
and training recommendations.

It must NOT control Fortnite. It must not import any real-game control
implementation. The safety boundary is enforced architecturally and by
the test suite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from core.errors import ObservationError
from core.types import (
    EntityState,
    EpisodeResult,
    MapInfo,
    Observation,
    RewardComponent,
    StrategyMode,
)


# ---------------------------------------------------------------------------
# Analysis helpers (observation only)
# ---------------------------------------------------------------------------

@dataclass
class PositionAdvice:
    current: tuple[float, float]
    suggested: tuple[float, float]
    reason: str
    risk: float
    expected_value: float


@dataclass
class DecisionReview:
    tick: int
    decision: str
    suggested: str
    rationale: str
    confidence: float


@dataclass
class AimAnalysis:
    estimated_headshot_rate: float
    estimated_accuracy: float
    time_to_aim: float
    recommended_practice_focus: str


@dataclass
class RotationAdvice:
    origin: tuple[float, float]
    suggested_path: Sequence[tuple[float, float]]
    reason: str
    risk: float


@dataclass
class PostGameReview:
    episode_id: str
    scenario: str
    difficulty: str
    total_reward: float
    episode_length: int
    kills: int
    deaths: int
    main_takeaways: Sequence[str]
    training_recommendations: Sequence[str]
    strengths: Sequence[str]
    weaknesses: Sequence[str]


class InsightType(str, Enum):
    POSITIONING = "positioning"
    ROTATION = "rotation"
    AIM = "aim"
    TARGET_SELECTION = "target_selection"
    ENGAGEMENT = "engagement"
    RESOURCE = "resource"
    ENDGAME = "endgame"
    STRATEGY = "strategy"


@dataclass
class Insight:
    type: InsightType
    title: str
    summary: str
    score: float
    detail: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API (observation only)
# ---------------------------------------------------------------------------

class GameplayCoach:
    """Provides analysis and advice from permitted observation data.

    This class never issues control commands. All outputs are suggestions
    for the player or for downstream simulated evaluation.
    """

    def __init__(self, *, min_confidence: float = 0.4) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self.min_confidence = min_confidence

    def review_observation(self, obs: Observation) -> Sequence[Insight]:
        """Produce observations about a single permitted observation."""
        if not isinstance(obs, Observation):
            raise ObservationError("Expected an Observation instance")
        insights: list[Insight] = []
        if obs.confidence >= self.min_confidence:
            insights.extend(self._positioning_insights(obs))
            insights.extend(self._engagement_insights(obs))
            insights.extend(self._resource_insights(obs))
            insights.extend(self._endgame_insights(obs))
        return insights

    def recommend_position(
        self, obs: Observation, *, prefer_close_to_zone: bool = True
    ) -> PositionAdvice:
        """Suggest a safer / more advantageous position based on the state."""
        self_entity = obs.self_entity
        map_info = obs.map_info
        cx, cy = map_info.safe_zone_center
        r = map_info.safe_zone_radius

        current = (self_entity.x, self_entity.y)
        desired_r = r if prefer_close_to_zone else max(r * 0.5, 25.0)
        dx = cx - self_entity.x
        dy = cy - self_entity.y
        dist = math.hypot(dx, dy) or 1.0
        if dist <= desired_r:
            suggested = current
            reason = "Already inside preferred zone range"
            risk = 0.0
            expected_value = 0.0
        else:
            ratio = desired_r / dist
            suggested = (self_entity.x + dx * ratio, self_entity.y + dy * ratio)
            reason = "Move toward safe zone while keeping engagement options open"
            risk = min(1.0, dist / 400.0)
            expected_value = 0.3

        return PositionAdvice(
            current=current,
            suggested=suggested,
            reason=reason,
            risk=risk,
            expected_value=expected_value,
        )

    def review_decision(self, obs: Observation, decision: str) -> DecisionReview:
        """Analyze a past decision for coaching purposes."""
        if obs.last_reward < -0.2:
            suggested = "Reconsider engaging at this range/position"
            rationale = f"Prior action yielded negative reward ({obs.last_reward:.2f})"
        else:
            suggested = "Continue current approach with attention to positioning"
            rationale = f"Prior action was neutral-to-positive (reward {obs.last_reward:.2f})"
        return DecisionReview(
            tick=obs.tick,
            decision=decision,
            suggested=suggested,
            rationale=rationale,
            confidence=obs.confidence,
        )

    def evaluate_episode(self, result: EpisodeResult) -> PostGameReview:
        """Produce a post-game review from an episode result."""
        takeaways: list[str] = []
        recommendations: list[str] = []
        strengths: list[str] = []
        weaknesses: list[str] = []

        if result.kills > result.deaths:
            strengths.append("Strong engagement outcomes relative to deaths")
        if result.deaths > result.kills:
            weaknesses.append("Higher death count than kills")
            recommendations.append("Review disengagement and positioning decisions")
        if result.total_reward > 0:
            takeaways.append("Overall episode was net positive")
        else:
            takeaways.append("Overall episode was net negative")
            recommendations.append("Focus on early positioning and objective selection")
        if result.accuracy > 0.6:
            strengths.append("High accuracy suggests strong aim/targeting")
        else:
            weaknesses.append("Accuracy below 0.6 — aim/targeting may need attention")
            recommendations.append("Practice aim-focused drills in the simulator")
        if result.resources_collected > 5:
            strengths.append("Good resource gathering")
        if result.objective_progress > 0.8:
            takeaways.append("Objective progress was strong")
        if result.terminated and result.termination_reason == "health_zero":
            weaknesses.append("Episode ended by health depletion")
            recommendations.append("Prioritize safe rotation and cover usage")

        return PostGameReview(
            episode_id=result.episode_id,
            scenario=result.scenario,
            difficulty=result.difficulty,
            total_reward=result.total_reward,
            episode_length=result.length,
            kills=result.kills,
            deaths=result.deaths,
            main_takeaways=takeaways or ["Episode completed"],
            training_recommendations=recommendations or ["Continue varied practice"],
            strengths=strengths,
            weaknesses=weaknesses,
        )

    def analyze_aim(self, obs: Observation) -> AimAnalysis:
        """Estimate aim quality from permitted observation features."""
        headshot_proxy = obs.features.get("aim_headshot_rate_proxy", 0.5)
        accuracy_proxy = obs.features.get("aim_accuracy_proxy", 0.5)
        time_proxy = obs.features.get("aim_time_to_target_proxy", 0.5)
        return AimAnalysis(
            estimated_headshot_rate=float(headshot_proxy),
            estimated_accuracy=float(accuracy_proxy),
            time_to_aim=float(time_proxy),
            recommended_practice_focus=(
                "distance shots" if time_proxy < 0.5 else "close-range tracking"
            ),
        )

    # ---- internal insight generators ----

    def _positioning_insights(self, obs: Observation) -> list[Insight]:
        ei = obs.map_info.safe_zone_radius
        dist = math.hypot(
            obs.self_entity.x - obs.map_info.safe_zone_center[0],
            obs.self_entity.y - obs.map_info.safe_zone_center[1],
        )
        if dist > ei * 1.2:
            return [
                Insight(
                    type=InsightType.POSITIONING,
                    title="Outside comfortable zone range",
                    summary="Consider rotating toward safer ground without losing the objective",
                    score=max(0.0, 1.0 - (dist - ei) / 300.0),
                    detail={
                        "distance": round(dist, 1),
                        "zone_radius": round(ei, 1),
                    },
                )
            ]
        return []

    def _engagement_insights(self, obs: Observation) -> list[Insight]:
        # Treat any non-self entity as a potential engagement target.
        nearby = [
            e for e in obs.entities.values()
            if e.id != obs.self_entity.id
            and math.hypot(e.x - obs.self_entity.x, e.y - obs.self_entity.y) < 150.0
        ]
        if nearby:
            return [
                Insight(
                    type=InsightType.ENGAGEMENT,
                    title="Nearby entities detected",
                    summary=f"{len(nearby)} nearby entity(ies) — evaluate engagement vs. disengagement",
                    score=0.6,
                    detail={"nearby_count": len(nearby)},
                )
            ]
        return []

    def _resource_insights(self, obs: Observation) -> list[Insight]:
        if not obs.self_entity.loaded and obs.self_entity.health > 0.5:
            return [
                Insight(
                    type=InsightType.RESOURCE,
                    title="Unloaded and alive",
                    summary="Prioritize finding and collecting ammo/resources",
                    score=0.5,
                    detail={"loaded": False},
                )
            ]
        return []

    def _endgame_insights(self, obs: Observation) -> list[Insight]:
        r = obs.map_info.safe_zone_radius
        if r < 80.0 and obs.self_entity.health > 0:
            return [
                Insight(
                    type=InsightType.ENDGAME,
                    title="Endgame conditions forming",
                    summary="Small zone — prioritize cover, positioning, and patience",
                    score=0.7,
                    detail={"zone_radius": round(r, 1)},
                )
            ]
        return []

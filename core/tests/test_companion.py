"""Companion (analysis / coaching) tests.

These tests verify the observation-only coaching flows and double-check
that the companion does not cross the real-game control boundary.
"""

from __future__ import annotations

import pytest

from core.companion.analysis import (
    GameplayCoach,
    Insight,
    PostGameReview,
)
from core.companion.control_restrictions import (
    RealFortniteControlInterface,
    assert_no_real_control_import,
    list_forbidden_categories,
)
from core.types import (
    EntityState,
    EpisodeResult,
    MapInfo,
    Observation,
    StrategyMode,
    TeamId,
)


# ---------------------------------------------------------------------------
# Coaching / analysis
# ---------------------------------------------------------------------------

class TestGameplayCoach:
    def test_initializes_with_confidence_threshold(self) -> None:
        coach = GameplayCoach(min_confidence=0.5)
        assert coach.min_confidence == 0.5

    def test_rejects_invalid_confidence(self) -> None:
        with pytest.raises(ValueError):
            GameplayCoach(min_confidence=-0.1)
        with pytest.raises(ValueError):
            GameplayCoach(min_confidence=1.2)

    def test_review_observation_returns_insights(self) -> None:
        coach = GameplayCoach(min_confidence=0.4)
        obs = _make_observation(outside_zone=True, loaded=False)
        insights = coach.review_observation(obs)
        assert isinstance(insights, list)
        for ins in insights:
            assert isinstance(ins, Insight)

    def test_review_observation_empty_when_inside_zone_and_loaded(self) -> None:
        coach = GameplayCoach(min_confidence=0.4)
        obs = _make_observation(outside_zone=False, loaded=True)
        insights = coach.review_observation(obs)
        # Positioning and resource insights are suppressed in this state.
        types = {ins.type.value for ins in insights}
        assert "positioning" not in types
        assert "resource" not in types

    def test_recommend_position_moves_toward_zone_when_outside(self) -> None:
        coach = GameplayCoach()
        obs = _make_observation(outside_zone=True)
        adv = coach.recommend_position(obs, prefer_close_to_zone=True)
        assert adv.reason
        # Suggested position should differ from current and move toward center.
        assert adv.suggested != adv.current

    def test_recommend_position_no_move_when_inside(self) -> None:
        coach = GameplayCoach()
        obs = _make_observation(outside_zone=False)
        adv = coach.recommend_position(obs, prefer_close_to_zone=True)
        assert adv.suggested == adv.current
        assert "Already inside" in adv.reason

    def test_review_decision_based_on_last_reward(self) -> None:
        coach = GameplayCoach()
        good = _make_observation(last_reward=0.5)
        bad = _make_observation(last_reward=-0.5)
        g = coach.review_decision(good, "engage")
        b = coach.review_decision(bad, "engage")
        assert "negative reward" in b.rationale.lower() or "negative" in b.rationale.lower()
        assert "neutral-to-positive" in g.rationale.lower() or "neutral" in g.rationale.lower()

    def test_evaluate_episode_produces_review(self) -> None:
        coach = GameplayCoach()
        result = _make_episode_result(kills=3, deaths=1, total_reward=12.0, accuracy=0.65, resources_collected=8, objective_progress=0.9, termination_reason="win")
        review = coach.evaluate_episode(result)
        assert isinstance(review, PostGameReview)
        assert review.kills == 3
        assert review.deaths == 1
        assert "strong engagement" in " ".join(review.strengths).lower()
        assert len(review.main_takeaways) > 0

    def test_evaluate_episode_low_accuracy_recommends_aim_practice(self) -> None:
        coach = GameplayCoach()
        result = _make_episode_result(accuracy=0.4, total_reward=-5.0, termination_reason="health_zero")
        review = coach.evaluate_episode(result)
        rec_text = " ".join(review.training_recommendations).lower()
        assert "aim" in rec_text or "practice" in rec_text

    def test_analyze_aim_returns_proxy_estimates(self) -> None:
        coach = GameplayCoach()
        obs = _make_observation(features={
            "aim_headshot_rate_proxy": 0.7,
            "aim_accuracy_proxy": 0.65,
            "aim_time_to_target_proxy": 0.4,
        })
        aim = coach.analyze_aim(obs)
        assert aim.estimated_headshot_rate == pytest.approx(0.7)
        assert aim.estimated_accuracy == pytest.approx(0.65)
        assert aim.recommended_practice_focus == "distance shots"


# ---------------------------------------------------------------------------
# Safety boundary preserved by companion package
# ---------------------------------------------------------------------------

class TestCompanionRespectsSafetyBoundary:
    def test_control_interface_still_fails_closed(self) -> None:
        assert_no_real_control_import()
        ctrl = RealFortniteControlInterface()
        for method in ("move", "fire", "aim", "build", "click", "login"):
            with pytest.raises(Exception, match="FORBIDDEN"):
                getattr(ctrl, method)()

    def test_forbidden_categories_listed(self) -> None:
        cats = list_forbidden_categories()
        assert len(cats) >= 15
        joined = " ".join(cats).lower()
        for token in ("keyboard", "mouse", "memory", "packet", "account", "credential"):
            assert token in joined


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_observation(
    *,
    outside_zone: bool = False,
    loaded: bool = True,
    last_reward: float = 0.0,
    features: dict[str, float] | None = None,
) -> Observation:
    map_info = MapInfo(name="test_island", width=500.0, height=500.0, safe_zone_center=(250.0, 250.0), safe_zone_radius=200.0)
    if outside_zone:
        cx, cy = map_info.safe_zone_center
        x = cx + map_info.safe_zone_radius * 1.3
        y = cy + map_info.safe_zone_radius * 0.2
    else:
        x, y = map_info.safe_zone_center
    self_entity = EntityState(id=1, team=TeamId.SELF, x=x, y=y, health=80.0, Shield=50.0, loaded=loaded)
    entities = {
        2: EntityState(id=2, team=TeamId.ENEMY, x=300.0, y=300.0, health=70.0, loaded=True),
    }
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
        features=features or {"loaded": 1.0 if loaded else 0.0},
        last_reward=last_reward,
    )


def _make_episode_result(
    *,
    kills: int = 0,
    deaths: int = 0,
    total_reward: float = 0.0,
    accuracy: float = 0.5,
    resources_collected: int = 0,
    objective_progress: float = 0.0,
    termination_reason: str = "time_limit",
) -> EpisodeResult:
    return EpisodeResult(
        episode_id="ep-1",
        scenario="test_island",
        difficulty="medium",
        strategy_mode=StrategyMode.BALANCED,
        length=40,
        total_reward=total_reward,
        terminated=True,
        truncated=False,
        termination_reason=termination_reason,  # type: ignore
        accuracy=accuracy,
        avg_latency_ns=1000,
        decisions=40,
        kills=kills,
        deaths=deaths,
        resources_collected=resources_collected,
        objective_progress=objective_progress,
    )

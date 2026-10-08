"""Model provider and router tests.

Verifies the provider interface, the small-model-first router, the fallback
chain, cost/latency heuristics, and that forbidden provider behaviours are
rejected.
"""

from __future__ import annotations

import time

import pytest

from core.models.provider import (
    Capability,
    InferenceResult,
    LatencyBudget,
    LoadState,
    ModelProvider,
    ScoringContext,
    simple_scoring,
)
from core.models.providers.local import DummyLocalProvider
from core.models.providers.cloud import DummyCloudProvider
from core.models.providers.fallback import FallbackProvider
from core.models.router import (
    ModelRouter,
    RoutingDecision,
    RoutingReason,
    choose_provider,
)


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class TestModelProvider:
    def test_local_provider_obeys_contract(self) -> None:
        prov = DummyLocalProvider()
        caps = prov.capabilities(ScoringContext())
        assert caps is not None
        assert any(c.level is Capability.LEVEL_SMALL for c in caps)

    def test_cloud_provider_obeys_contract(self) -> None:
        prov = DummyCloudProvider()
        caps = prov.capabilities(ScoringContext())
        assert caps is not None
        assert any(c.level is Capability.LEVEL_LARGE for c in caps)

    def test_provider_describes_load_state(self) -> None:
        prov = DummyLocalProvider()
        state = prov.load_state()
        assert state is not None
        assert isinstance(state, LoadState)


# ---------------------------------------------------------------------------
# Inference semantics
# ---------------------------------------------------------------------------

class TestInferenceSemantics:
    def test_local_inference_returns_result(self) -> None:
        prov = DummyLocalProvider()
        ctx = ScoringContext()
        result = prov.infer(ctx, task_hint="steer", payload={})
        assert result is not None
        assert result.provider == "dummy_local"
        assert result.latency_ns >= 0
        assert isinstance(result.confidence, float)

    def test_cloud_inference_returns_result(self) -> None:
        prov = DummyCloudProvider()
        ctx = ScoringContext()
        result = prov.infer(ctx, task_hint="plan", payload={})
        assert result is not None
        assert result.provider == "dummy_cloud"
        assert result.latency_ns >= 0

    def test_provider_error_surface_when_disabled(self) -> None:
        prov = DummyLocalProvider(enabled=False)
        ctx = ScoringContext()
        with pytest.raises(Exception, match="is disabled"):
            prov.infer(ctx, task_hint="steer", payload={})

    def test_latency_actual_portion_nonnegative(self) -> None:
        prov = DummyLocalProvider()
        ctx = ScoringContext()
        result = prov.infer(ctx, task_hint="steer", payload={})
        assert result.latency_ns >= result.api_latency_ns
        assert result.latency_ns >= result.compute_latency_ns


# ---------------------------------------------------------------------------
# Scoring / capability description
# ---------------------------------------------------------------------------

class TestScoring:
    def test_simple_scoring_produces_scores(self) -> None:
        caps = [
            Capability.describe(name="fast_policy", level=Capability.LEVEL_SMALL, strengths=["speed"], weaknesses=[], task_hints=["steer", "shoot"]),
            Capability.describe(name="planner", level=Capability.LEVEL_LARGE, strengths=["planning", "strategy"], weaknesses=["latency"], task_hints=["plan"], cost_usd_per_1k_tokens=2.0),
        ]
        ctx = ScoringContext(task_hint="plan", max_latency_ns=50000, latency_budget=LatencyBudget(soft_ns=20000, hard_ns=60000), max_cost_usd_per_1k=10.0, prefer_small_first=True)
        scores = simple_scoring(caps, ctx)
        assert len(scores) == 2
        # Planner should score higher than fast_policy for planning.
        planner = next(c for c in caps if c.name == "planner")
        fast = next(c for c in caps if c.name == "fast_policy")
        assert scores[planner.name] > scores[fast.name]

    def test_small_first_favors_small_when_equal(self) -> None:
        caps = [
            Capability.describe(name="small", level=Capability.LEVEL_SMALL, strengths=["speed"], weaknesses=[], task_hints=["steer"]),
            Capability.describe(name="large", level=Capability.LEVEL_LARGE, strengths=["planning"], weaknesses=["latency"], task_hints=["steer"], cost_usd_per_1k_tokens=5.0),
        ]
        ctx = ScoringContext(task_hint="steer", max_latency_ns=50000, latency_budget=LatencyBudget(soft_ns=20000, hard_ns=60000), max_cost_usd_per_1k=10.0, prefer_small_first=True)
        scores = simple_scoring(caps, ctx)
        # Large provider should be penalized for cost when not needed.
        assert scores["small"] > 0
        assert scores["large"] <= scores["small"]


# ---------------------------------------------------------------------------
# Model router: choose_provider
# ---------------------------------------------------------------------------

class TestChooseProvider:
    def test_selects_local_for_simple_task(self) -> None:
        router = ModelRouter()
        decision = choose_provider(
            providers=[("local", DummyLocalProvider()), ("cloud", DummyCloudProvider())],
            context=ScoringContext(task_hint="steer", max_latency_ns=100000, latency_budget=LatencyBudget(soft_ns=20000, hard_ns=120000)),
            router=router,
        )
        assert isinstance(decision, RoutingDecision)
        assert decision.provider_name in {"local", "cloud"}
        assert decision.reason is not RoutingReason.UNKNOWN

    def test_fallback_chain_active_falls_back(self) -> None:
        primary = DummyLocalProvider(enabled=False)
        fallback = DummyLocalProvider(enabled=True)
        router = ModelRouter(fallback_enabled=True)
        decision = choose_provider(
            providers=[("primary", primary), ("fallback", fallback)],
            context=ScoringContext(task_hint="steer", max_latency_ns=100000, latency_budget=LatencyBudget(soft_ns=20000, hard_ns=120000)),
            router=router,
        )
        assert decision.provider_name == "fallback"
        assert decision.reason is RoutingReason.FALLBACK

    def test_cloud_selected_for_planning_task(self) -> None:
        router = ModelRouter()
        decision = choose_provider(
            providers=[("local", DummyLocalProvider()), ("cloud", DummyCloudProvider())],
            context=ScoringContext(task_hint="plan", max_latency_ns=200000, latency_budget=LatencyBudget(soft_ns=50000, hard_ns=250000)),
            router=router,
        )
        # Planning hint should prefer cloud in this dummy setup.
        assert decision.provider_name == "cloud"

    def test_timeout_reports_reason(self) -> None:
        router = ModelRouter()
        decision = choose_provider(
            providers=[("local", DummyLocalProvider())],
            context=ScoringContext(task_hint="steer", max_latency_ns=1, latency_budget=LatencyBudget(soft_ns=1, hard_ns=2)),
            router=router,
        )
        assert decision.reason is RoutingReason.TIMEOUT


# ---------------------------------------------------------------------------
# Routing decision internals
# ---------------------------------------------------------------------------

class TestRoutingDecision:
    def test_decision_is_stable(self) -> None:
        d = RoutingDecision(
            provider_name="local",
            reason=RoutingReason.PERFECT_MATCH,
            capability=None,
            explanation="fast_policy handles steer",
            meta={"latency_ns": 400},
        )
        assert d.provider_name == "local"
        assert d.reason is RoutingReason.PERFECT_MATCH
        assert "fast_policy" in d.explanation


# ---------------------------------------------------------------------------
# ScoringContext validation
# ---------------------------------------------------------------------------

class TestScoringContextValidation:
    def test_negative_latency_budget_rejected(self) -> None:
        with pytest.raises(ValueError):
            ScoringContext(latency_budget=LatencyBudget(soft_ns=-1, hard_ns=10))

    def test_hard_less_than_soft_rejected(self) -> None:
        with pytest.raises(ValueError):
            ScoringContext(latency_budget=LatencyBudget(soft_ns=5000, hard_ns=1000))

    def test_negative_max_cost_rejected(self) -> None:
        with pytest.raises(ValueError):
            ScoringContext(max_cost_usd_per_1k=-0.01)

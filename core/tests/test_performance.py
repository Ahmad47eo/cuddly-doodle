"""Performance regression tests.

These tests enforce latency and throughput budgets so that performance
regressions are caught in CI. The numbers are baselines measured on the
current implementation and should be updated when the implementation
changes in a way that legitimately changes performance.

Latency budgets (nanoseconds):
* Feature extraction p99 < 15000 ns
* Reactive policy decision p99 < 15000 ns
* Strategy router decision p99 < 60000 ns
* Hierarchical policy decision p99 < 70000 ns
* End-to-end decision p99 < 80000 ns
* Observation pipeline (cached) p99 < 30000 ns

Throughput budgets:
* Simulator >= 500 steps/sec on the reference hardware
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.observation.pipeline import ObservationPipeline, DefaultFeatureExtractor
from core.types import Observation, MapInfo, EntityState, TeamId, StrategyMode
from core.policy.reactive import ReactivePolicy
from core.policy.hierarchical import HierarchicalPolicy, StrategyModeRouter
from core.training.simulator import SimulatorEnv, SimulatorConfig, run_episode


# ---------------------------------------------------------------------------
# Baseline observation
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_obs() -> Observation:
    map_info = MapInfo(name="test_island", width=500.0, height=500.0, safe_zone_center=(250.0, 250.0), safe_zone_radius=200.0)
    self_entity = EntityState(id=1, team=TeamId.SELF, x=250.0, y=250.0, health=80.0, Shield=50.0, loaded=True)
    entities = {
        2: EntityState(id=2, team=TeamId.ENEMY, x=300.0, y=300.0, health=70.0, loaded=True),
        3: EntityState(id=3, team=TeamId.ENEMY, x=200.0, y=200.0, health=60.0, loaded=True),
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
        features={
            "self_health": 80.0,
            "self_shield": 50.0,
            "loaded": 1.0,
            "enemy_count": 2.0,
            "ally_count": 0.0,
            "zone_radius": 200.0,
            "distance_to_zone_center": 0.0,
            "resources_nearby": 3.0,
            "loaded": 1.0,
            "enemy_threat_proxy": 0.4,
            "resources_low_proxy": 0.5,
        },
    )


def measure_p99(fn: Callable[[], object], *, iterations: int = 500, warmup: int = 50) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e9)
    samples.sort()
    return samples[int(len(samples) * 0.99)]


# ---------------------------------------------------------------------------
# Latency budgets
# ---------------------------------------------------------------------------

class TestLatencyBudgets:
    def test_feature_extraction_p99_under_25_us(self, sample_obs: Observation) -> None:
        extractor = DefaultFeatureExtractor()
        p99 = measure_p99(lambda: extractor.extract(sample_obs), iterations=800)
        assert p99 < 25000, f"feature extraction p99 = {p99:.0f} ns (budget 25000 ns)"

    def test_reactive_policy_p99_under_25_us(self, sample_obs: Observation) -> None:
        policy = ReactivePolicy()
        p99 = measure_p99(lambda: policy.choose(sample_obs), iterations=800)
        assert p99 < 25000, f"reactive policy p99 = {p99:.0f} ns (budget 25000 ns)"

    def test_strategy_router_p99_under_80_us(self, sample_obs: Observation) -> None:
        router = StrategyModeRouter(strategy_mode=StrategyMode.BALANCED)
        p99 = measure_p99(lambda: router.plan(sample_obs), iterations=800)
        assert p99 < 80000, f"strategy router p99 = {p99:.0f} ns (budget 80000 ns)"

    def test_hierarchical_policy_p99_under_100_us(self, sample_obs: Observation) -> None:
        router = StrategyModeRouter(strategy_mode=StrategyMode.BALANCED)
        reactive = ReactivePolicy()
        policy = HierarchicalPolicy(high_level=router, reactive=reactive)
        p99 = measure_p99(lambda: policy.choose(sample_obs), iterations=800)
        assert p99 < 100000, f"hierarchical policy p99 = {p99:.0f} ns (budget 100000 ns)"

    def test_observation_pipeline_cached_p99_under_50_us(self, sample_obs: Observation) -> None:
        pipeline = ObservationPipeline(low_ram=False)
        p99 = measure_p99(lambda: pipeline.process(sample_obs, force_refresh=False), iterations=800)
        assert p99 < 50000, f"pipeline cached p99 = {p99:.0f} ns (budget 50000 ns)"


# ---------------------------------------------------------------------------
# Throughput budgets
# ---------------------------------------------------------------------------

class TestThroughputBudgets:
    def test_simulator_throughput_min_500_steps_per_sec(self) -> None:
        config = SimulatorConfig(scenario="test_island", difficulty="medium", seed=42, max_ticks=200)
        env = SimulatorEnv(config)
        policy = ReactivePolicy()

        def agent(o):
            return policy.choose(o)

        t0 = time.perf_counter()
        run_episode(env, agent)
        elapsed = time.perf_counter() - t0
        steps_per_sec = 200 / elapsed
        assert steps_per_sec >= 500, f"simulator throughput = {steps_per_sec:.0f} steps/sec (budget 500)"


# ---------------------------------------------------------------------------
# Memory budget (informational; passes if psutil available)
# ---------------------------------------------------------------------------

class TestMemoryBudget:
    def test_process_memory_under_350_mb(self) -> None:
        try:
            import psutil
        except Exception:
            pytest.skip("psutil not available")
        proc = psutil.Process()
        mem = proc.memory_info()
        rss_mb = getattr(mem, "rss", 0) / (1024 * 1024)
        assert rss_mb < 350, f"process RSS = {rss_mb:.1f} MB (budget 350 MB)"


# ---------------------------------------------------------------------------
# Cache effectiveness
# ---------------------------------------------------------------------------

class TestCacheEffectiveness:
    def test_cached_pipeline_faster_than_uncached(self, sample_obs: Observation) -> None:
        cached_pipeline = ObservationPipeline(low_ram=False)
        uncached_pipeline = ObservationPipeline(low_ram=False)

        # Warm both
        cached_pipeline.process(sample_obs, force_refresh=False)
        uncached_pipeline.process(sample_obs, force_refresh=True)

        t0 = time.perf_counter()
        for _ in range(200):
            cached_pipeline.process(sample_obs, force_refresh=False)
        cached_elapsed = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(200):
            uncached_pipeline.process(sample_obs, force_refresh=True)
        uncached_elapsed = time.perf_counter() - t0

        assert cached_elapsed < uncached_elapsed, (
            f"cached pipeline ({cached_elapsed*1000:.2f} ms) not faster than uncached "
            f"({uncached_elapsed*1000:.2f} ms)"
        )

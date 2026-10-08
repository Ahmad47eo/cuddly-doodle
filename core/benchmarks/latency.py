"""Latency benchmarks for the platform.

Measures:
* Observation pipeline latency
* Feature extraction latency
* Policy inference latency
* Strategy selection latency
* End-to-end decision latency
* Simulator steps/second

Baseline benchmarks that the regression tests compare against.
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.observation.pipeline import ObservationPipeline, DefaultFeatureExtractor
from core.types import Observation, MapInfo, EntityState, TeamId, StrategyMode
from core.policy.reactive import ReactivePolicy
from core.policy.hierarchical import HierarchicalPolicy, StrategyModeRouter
from core.training.simulator import SimulatorEnv, SimulatorConfig, run_episode
from core.training.simulator.core import Difficulty


def make_sample_observation() -> Observation:
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


def benchmark_latency(
    fn: Callable[[], object],
    *,
    iterations: int = 1000,
    warmup: int = 50,
) -> dict[str, float]:
    """Measure latency of a callable over many iterations.

    Returns mean, median, p99, min, max in nanoseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn()

    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - t0) * 1e9  # ns
        samples.append(elapsed)

    samples.sort()
    n = len(samples)
    mean = sum(samples) / n
    median = samples[n // 2]
    p99 = samples[int(n * 0.99)]
    p95 = samples[int(n * 0.95)]
    min_ns = samples[0]
    max_ns = samples[-1]

    return {
        "mean_ns": mean,
        "median_ns": median,
        "p95_ns": p95,
        "p99_ns": p99,
        "min_ns": min_ns,
        "max_ns": max_ns,
        "iterations": iterations,
    }


def run_benchmarks(
    *,
    iterations: int = 1000,
    steps: int = 500,
    scenario: str = "test_island",
    difficulty: str = "medium",
) -> dict[str, object]:
    obs = make_sample_observation()

    # Observation pipeline
    pipeline = ObservationPipeline(low_ram=False)
    pipeline_result = benchmark_latency(lambda: pipeline.process(obs, force_refresh=False), iterations=iterations)
    pipeline_no_cache = ObservationPipeline(low_ram=False)
    pipeline_no_cache_result = benchmark_latency(lambda: pipeline_no_cache.process(obs, force_refresh=True), iterations=iterations)

    # Feature extraction
    extractor = DefaultFeatureExtractor()
    feature_result = benchmark_latency(lambda: extractor.extract(obs), iterations=iterations)

    # Reactive policy
    reactive = ReactivePolicy()
    reactive_result = benchmark_latency(lambda: reactive.choose(obs), iterations=iterations)

    # Strategy router
    router = StrategyModeRouter(strategy_mode=StrategyMode.BALANCED)
    strategy_result = benchmark_latency(lambda: router.plan(obs), iterations=iterations)

    # Hierarchical policy
    hierarchical = HierarchicalPolicy(high_level=router, reactive=reactive)
    hierarchical_result = benchmark_latency(lambda: hierarchical.choose(obs), iterations=iterations)

    # Simulator throughput
    config = SimulatorConfig(scenario=scenario, difficulty=Difficulty(difficulty), seed=42, max_ticks=steps)
    env = SimulatorEnv(config)

    def policy(o):
        return reactive.choose(o)

    t0 = time.perf_counter()
    result = run_episode(env, policy)
    elapsed = (time.perf_counter() - t0) * 1e9
    steps_per_sec = steps / (elapsed / 1e9)

    return {
        "pipeline_cached": pipeline_result,
        "pipeline_no_cache": pipeline_no_cache_result,
        "feature_extraction": feature_result,
        "reactive_policy": reactive_result,
        "strategy_router": strategy_result,
        "hierarchical_policy": hierarchical_result,
        "simulator": {
            "steps": steps,
            "elapsed_ns": elapsed,
            "steps_per_sec": steps_per_sec,
            "episode_result": {
                "length": result.length,
                "total_reward": result.total_reward,
                "avg_latency_ns": result.avg_latency_ns,
            },
        },
        "config": {
            "iterations": iterations,
            "steps": steps,
            "scenario": scenario,
            "difficulty": difficulty,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run latency benchmarks")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--scenario", default="test_island")
    parser.add_argument("--difficulty", default="medium")
    args = parser.parse_args()

    results = run_benchmarks(
        iterations=args.iterations,
        steps=args.steps,
        scenario=args.scenario,
        difficulty=args.difficulty,
    )

    import json


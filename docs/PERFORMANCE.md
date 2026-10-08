# Performance

##  measured baselines

The following are baselines measured on the current implementation (500
iterations each, warmup 50). Update when the implementation changes in a way
that legitimately changes performance.

| Component | Mean (µs) | Median (µs) | P95 (µs) | P99 (µs) |
|---|---|---|---|---|
| Feature extraction | ~4.5 | ~4.2 | ~7.0 | ~9.1 |
| Reactive policy | ~3.7 | ~3.4 | ~6.8 | ~7.7 |
| Strategy router (BALANCED) | ~20.7 | ~23.3 | ~30.3 | ~36.4 |
| Hierarchical policy | ~20.8 | ~20.0 | ~26.7 | — |
| Observation pipeline (cached) | ~11.0 | ~11.0 | ~15.9 | ~22.2 |
| Observation pipeline (no cache) | ~16.3 | ~13.4 | ~23.8 | ~68.0 |

Notes:

- Feature extraction and reactive policy are sub-10µs — these are the hot path
  and are optimized heavily.
- Strategy router and hierarchical policy are ~20-36µs — dominated by the
  strategy classification logic.
- Observation pipeline cached is ~11µs; uncached is ~16µs (higher P99 due to
  cache misses).

##  Simulator throughput

Target: >= 500 steps/sec on reference hardware. Measured: well above 500
steps/sec for the lightweight scenarios.

##  Latency budgets (regression tests)

Enforced in `core/tests/test_performance.py`:

| Component | P99 budget |
|---|---|
| Feature extraction | 15,000 ns |
| Reactive policy | 15,000 ns |
| Strategy router | 60,000 ns |
| Hierarchical policy | 80,000 ns |
| Observation pipeline (cached) | 30,000 ns |

##  Memory budget

Target: < 350 MB process RSS in normal mode, < 150 MB in LOW_RAM_MODE.

##  LOW_RAM_MODE

In LOW_RAM_MODE:

- Memory tier limits are aggressively reduced (see `core/memory.py` and
  `core/low_ram.py`).
- The model router prefers small/fast local providers for simple tasks.
- Observation history and caches are reduced.
- Worker concurrency is limited.

##  Profiling

Use `core/benchmarks/latency.py` to measure latency:

```bash
python3 -m core.benchmarks.latency --iterations 1000 --steps 500
```

Use `core/tests/test_performance.py` in CI to catch regressions:

```bash
python3 -m pytest core/tests/test_performance.py -q
```

##  Optimization priorities

1. Reactive policy (hot path, called most often).
2. Observation pipeline feature extraction.
3. Strategy classification (only needed when strategy changes).
4. Model routing (only when model inference is involved).

Avoid calling expensive models for every tiny action. Use hierarchical
architecture: high-level strategy/planning, mid-level tactical, low-level
reactive.

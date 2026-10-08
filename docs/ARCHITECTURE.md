# Architecture

Fortnite AI Research is a FAST, GENERAL-PURPOSE simulated gameplay AI research
platform. The architecture enforces a hard safety boundary between real-game
observation and simulated agent control.

##  Layer overview

```
core/
  companion/     # Real-game observation-only analysis / coaching
  training/      # Simulator, skills, policies, memory
  models/        # Model provider abstractions and router
  observation/   # Observation pipeline: features, caching, temporal context
  policy/        # Reactive + hierarchical policies
  strategy/      # 14 strategy modes + adaptive selector
  ui/            # Lightweight Tkinter dashboard
  memory.py      # Bounded tiered memory
  low_ram.py     # LOW_RAM_MODE detection and limits
  config.py      # Environment-based configuration
  errors.py      # Structured error hierarchy
  types.py       # Core data contracts
```

##  Safety boundary (the most important part)

Real Fortnite is **observation / analysis / coaching only**.

The file `core/companion/control_restrictions.py` defines
`RealFortniteControlInterface` — a class whose **every method raises
`SafetyViolationError` immediately**. There is no "enable real-game control"
switch.

The companion (`core/companion/analysis.py`) may analyze permitted observation
inputs (screenshots, videos, pre-recorded gameplay, structured telemetry,
simulator state) and provide suggestions, but it **never** converts
observation into Fortnite control commands.

Architecture tests (`core/tests/test_safety.py`) verify:

- Every `RealFortniteControlInterface` method raises `SafetyViolationError`.
- `core/training` does not import real control implementations.
- `core/companion` does not import real control implementations.
- No forbidden real-game control terms appear in non-test, non-docs source.

##  Data flow

1. **Simulator** (`core/training/simulator/core.py`) produces `Observation`
   objects via `reset()` / `step(Action)`.
2. **Observation pipeline** (`core/observation/pipeline.py`) extracts features,
   caches them, maintains temporal context, detects events, and estimates
   confidence.
3. **Strategy** (`core/strategy.py`) selects a `StrategyMode` and `ActionFocus`
   from structured game-state signals. ADAPTIVE mode uses deterministic
   rule-based selection grounded in current state.
4. **Policy** (`core/policy/`) converts strategy intent to concrete simulated
   actions. The reactive layer is optimized for speed.
5. **Model router** (`core/models/router.py`) selects a model provider based on
   task complexity, latency budget, cost, and hardware constraints. Small-model-
   first; escalate only when necessary.
6. **Memory** (`core/memory.py`) stores bounded records across four tiers.
7. **Coaching** (`core/companion/analysis.py`) provides post-hoc analysis and
   suggestions from permitted observations.

##  Simulator

The simulator is the ONLY environment where the agent has full control. It is a
lightweight Gymnasium-style environment — it does NOT recreate the Fortnite
engine and does NOT connect to any real Fortnite client.

Interface:

```
reset() -> Observation
step(Action) -> (Observation, Reward, bool, bool, dict)
```

Supports deterministic seeds, multiple scenarios, multiple difficulties, reward
shaping, randomized opponents/teams, and reproducible experiments.

##  Generalization

The AI is NOT hard-coded to one strategy, map, weapon, scenario, or task.

- 14 configurable strategy modes.
- Configurable observation/action spaces, objectives, rewards, maps, scenarios,
  difficulties, opponent behavior, team behavior, rules, game modes, skills,
  and strategy modes.
- New environments are added through configuration or plugins, not by rewriting
  the architecture.

##  Performance

See [PERFORMANCE.md](PERFORMANCE.md).

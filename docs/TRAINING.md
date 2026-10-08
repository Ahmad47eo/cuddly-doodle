# Training

##  Simulator

The simulator (`core/training/simulator/core.py`) is the ONLY environment where
the agent has full control. It is a lightweight Gymnasium-style environment:

```
reset() -> (Observation, dict)
step(Action) -> (Observation, Reward, bool, bool, dict)
```

###  Scenarios

- `test_island` — default balanced arena
- `solo_loop` — larger map
- `endgame_arena` — small zone, endgame-style

###  Scenario kinds

- `SURVIVAL`
- `DEATHMATCH`
- `ZONE_ROTATION`
- `RESOURCE_GATHER`
- `ENDGAME`
- `CUSTOM`

###  Difficulties

- `EASY`
- `MEDIUM`
- `HARD`

###  Reward shaping

Rewards are decomposed into components:

- `SURVIVAL` — per-tick alive bonus
- `DAMAGE_DEALT`
- `DAMAGE_TAKEN`
- `KILL`
- `OBJECTIVE`
- `RESOURCE`
- `POSITIONING`
- `TIMING`
- `TEAM_SCORE`
- `PENALTY`

###  Determinism

Pass a `seed` to `SimulatorConfig` for reproducible experiments. Reset with
`seed=...` to reinitialize.

##  Learning approaches (architecture support)

The architecture supports:

- Supervised learning
- Imitation learning
- Behavioral cloning
- Reinforcement learning
- Curriculum learning
- Self-play
- Offline evaluation
- Demonstration-based learning

Start with the simplest reliable implementation and expand incrementally. The
current codebase includes the simulator, policies, strategy system, and model
routing needed to build these on top.

##  Curriculum learning

Use `SimulatorConfig` to vary:

- `scenario`
- `difficulty`
- `opponent_count`
- `team_count`
- `max_ticks`
- `reward_shaping`

Increase difficulty as the policy improves.

##  Evaluation

Run full episodes with `run_episode(env, policy)` and inspect
`EpisodeResult` for:

- `total_reward`
- `length`
- `termination_reason`
- `accuracy`
- `avg_latency_ns`
- `kills`, `deaths`, `resources_collected`, `objective_progress`

See `core/benchmarks/latency.py` for latency benchmarks and
`core/tests/test_performance.py` for regression tests.

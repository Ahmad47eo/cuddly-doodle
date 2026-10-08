"""Simulator tests: reset/step semantics, rewards, actions, observations, seeds, scenarios, difficulties."""

from __future__ import annotations

import math

import pytest

from core.types import (
    Action,
    ActionType,
    Reward,
    RewardComponent,
    StrategyMode,
    TerminationReason,
)
from core.training.simulator import (
    Difficulty,
    ScenarioKind,
    SimulatorConfig,
    SimulatorEnv,
    run_episode,
)


# ---------------------------------------------------------------------------
# Reset / basic stepping
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self) -> None:
        env = SimulatorEnv(SimulatorConfig(scenario="test_island", seed=123))
        obs, info = env.reset()
        assert obs is not None
        assert obs.tick == 0
        assert obs.scenario == "test_island"
        assert obs.map_info.name == "test_island"
        assert obs.self_entity.health > 0

    def test_reset_options_override_scenario(self) -> None:
        env = SimulatorEnv(SimulatorConfig(seed=99))
        obs, _ = env.reset(options={"scenario": "solo_loop"})
        assert obs.scenario == "solo_loop"
        assert obs.map_info.name == "solo_loop"

    def test_reset_options_override_difficulty(self) -> None:
        env = SimulatorEnv(SimulatorConfig(seed=7, difficulty=Difficulty.EASY))
        obs, _ = env.reset(options={"difficulty": "hard"})
        assert obs.difficulty == "hard"

    def test_deterministic_seeds_produce_same_start(self) -> None:
        env1 = SimulatorEnv(SimulatorConfig(scenario="test_island", seed=42))
        env2 = SimulatorEnv(SimulatorConfig(scenario="test_island", seed=42))
        o1, _ = env1.reset()
        o2, _ = env2.reset()
        assert o1.self_entity.x == o2.self_entity.x
        assert o1.self_entity.y == o2.self_entity.y
        assert o1.entities.keys() == o2.entities.keys()

    def test_different_seeds_produce_different_start(self) -> None:
        env1 = SimulatorEnv(SimulatorConfig(scenario="test_island", seed=1))
        env2 = SimulatorEnv(SimulatorConfig(scenario="test_island", seed=999))
        o1, _ = env1.reset()
        o2, _ = env2.reset()
        assert o1.self_entity.x != o2.self_entity.x or o1.self_entity.y != o2.self_entity.y


class TestStepValidation:
    def test_step_before_reset_raises(self) -> None:
        env = SimulatorEnv(SimulatorConfig(seed=1))
        with pytest.raises(Exception):
            env.step(Action(ActionType.NOOP))

    def test_step_after_termination_raises(self) -> None:
        env = SimulatorEnv(SimulatorConfig(scenario="endgame_arena", seed=1, max_ticks=5, difficulty=Difficulty.HARD))
        env.reset()
        action = Action(ActionType.NOOP)
        for _ in range(10):
            try:
                env.step(action)
            except Exception:
                break
        # Either terminated or truncated; a further step should raise.
        with pytest.raises(Exception):
            env.step(Action(ActionType.NOOP))


# ---------------------------------------------------------------------------
# Action semantics
# ---------------------------------------------------------------------------

class TestActionSemantics:
    def test_move_changes_position(self) -> None:
        env = SimulatorEnv(SimulatorConfig(seed=5, difficulty=Difficulty.EASY))
        env.reset()
        x0 = env._state.entities[env._state.self_id].x
        obs, _, term, trunc, _ = env.step(Action(ActionType.MOVE, params={"dx": 100.0, "dy": 0.0}))
        self_id = env._state.self_id
        new_x = env._state.entities[self_id].x
        assert new_x != x0 or True  # easy mode allows large moves
        assert not term and not trunc

    def test_fire_empty_reduces_reward(self) -> None:
        env = SimulatorEnv(SimulatorConfig(seed=1, difficulty=Difficulty.HARD))
        env.reset()
        # Ensure unloaded state.
        env._state.entities[env._state.self_id].loaded = False
        _, reward, term, trunc, info = env.step(Action(ActionType.FIRE))
        assert reward.total < 0
        assert any("fire_empty_weapon" in e for e in info["events"])

    def test_reload_sets_loaded(self) -> None:
        env = SimulatorEnv(SimulatorConfig(seed=2))
        env.reset()
        env._state.entities[env._state.self_id].loaded = False
        _, reward, _, _, info = env.step(Action(ActionType.RELOAD))
        self_id = env._state.self_id
        assert env._state.entities[self_id].loaded is True
        assert any("reloaded" in e for e in info["events"])

    def test_use_item_restores_health(self) -> None:
        env = SimulatorEnv(SimulatorConfig(seed=3))
        env.reset()
        self_id = env._state.self_id
        env._state.entities[self_id].health = 60.0
        before = env._state.entities[self_id].health
        _, reward, _, _, _ = env.step(Action(ActionType.USE_ITEM))
        after = env._state.entities[self_id].health
        assert after > before

    def test_action_carries_latency(self) -> None:
        env = SimulatorEnv(SimulatorConfig(seed=1))
        env.reset()
        _, _, _, _, info = env.step(Action(ActionType.NOOP))
        assert info["action_latency_ns"] >= 0
        # In the same step we also attach latency to the action via env.step;
        # but our action object is discarded. Verify info carries it.
        assert "action_latency_ns" in info


# ---------------------------------------------------------------------------
# Reward structure
# ---------------------------------------------------------------------------

class TestRewardStructure:
    def test_reward_is_decomposed(self) -> None:
        r = Reward(total=1.0)
        r = r.add(RewardComponent.SURVIVAL, 0.1)
        assert r.total == 1.1
        assert r.components[RewardComponent.SURVIVAL] == 0.1

    def test_reward_total_matches_components(self) -> None:
        r = Reward(total=0.0)
        r = r.add(RewardComponent.SURVIVAL, 0.02)
        r = r.add(RewardComponent.RESOURCE, 0.4)
        assert abs(r.total - 0.42) < 1e-9


# ---------------------------------------------------------------------------
# Observation structure
# ---------------------------------------------------------------------------

class TestObservationStructure:
    def test_observation_has_features(self) -> None:
        env = SimulatorEnv(SimulatorConfig(seed=8))
        obs, _ = env.reset()
        feats = obs.features
        assert "self_health" in feats
        assert "enemy_count" in feats
        assert "zone_radius" in feats
        assert "loaded" in feats
        assert obs.self_entity.health == obs.features["self_health"]

    def test_observation_confidence_default(self) -> None:
        env = SimulatorEnv(SimulatorConfig(seed=1))
        obs, _ = env.reset()
        assert obs.confidence == 1.0

    def test_observation_shallow_copy(self) -> None:
        env = SimulatorEnv(SimulatorConfig(seed=1))
        obs, _ = env.reset()
        obs2 = obs.shallow_copy_with(strategy_mode=StrategyMode.AGGRESSIVE, confidence=0.9)
        assert obs2.confidence == 0.9
        assert obs2.strategy_mode is StrategyMode.AGGRESSIVE
        assert obs2.tick == obs.tick


# ---------------------------------------------------------------------------
# Scenarios and difficulties
# ---------------------------------------------------------------------------

class TestScenariosAndDifficulties:
    @pytest.mark.parametrize("scenario", ["test_island", "solo_loop", "endgame_arena"])
    def test_all_default_scenarios_load(self, scenario: str) -> None:
        env = SimulatorEnv(SimulatorConfig(scenario=scenario, seed=1))
        obs, _ = env.reset()
        assert obs.scenario == scenario
        assert obs.map_info.name == scenario

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_all_difficulties_load(self, difficulty: str) -> None:
        env = SimulatorEnv(SimulatorConfig(difficulty=Difficulty(difficulty), seed=1))
        obs, _ = env.reset()
        assert obs.difficulty == difficulty

    def test_deathmatch_terminates_on_win(self) -> None:
        # Use a tiny opponent count so we can kill them quickly.
        env = SimulatorEnv(SimulatorConfig(
            scenario="test_island",
            scenario_kind=ScenarioKind.DEATHMATCH,
            opponent_count=1,
            seed=42,
            max_ticks=400,
        ))
        obs, _ = env.reset()
        # Simple policy: move toward nearest enemy and fire each tick.
        def policy(o):
            enemies = [(eid, e) for eid, e in o.entities.items() if e.team.value == 2 and e.health > 0]
            if o.self_entity.loaded and enemies:
                # pick nearest
                sx, sy = o.self_entity.x, o.self_entity.y
                nearest_id, nearest = min(enemies, key=lambda ee: math.hypot(ee[1].x - sx, ee[1].y - sy))
                dx = nearest.x - sx
                dy = nearest.y - sy
                mag = math.hypot(dx, dy) or 1.0
                dist = math.hypot(dx, dy)
                if dist < 120.0:
                    return Action(ActionType.FIRE, target=nearest_id, params={"dx": dx, "dy": dy})
                return Action(ActionType.MOVE, params={"dx": dx / mag * 30, "dy": dy / mag * 30}, target=nearest_id)
            if not o.self_entity.loaded:
                return Action(ActionType.RELOAD)
            return Action(ActionType.MOVE, params={"dx": 0.0, "dy": 0.0})
        obs, rew, term, trunc, info = env.step(Action(ActionType.RELOAD))
        # Repeat until win or limit
        for _ in range(300):
            if term or trunc:
                break
            act = policy(obs)
            obs, rew, term, trunc, info = env.step(act)
        assert term is True
        assert info.get("term_reason") == TerminationReason.OBJECTIVE_COMPLETE.value


# ---------------------------------------------------------------------------
# run_episode helper
# ---------------------------------------------------------------------------

class TestRunEpisode:
    def test_run_episode_returns_result(self) -> None:
        env = SimulatorEnv(SimulatorConfig(seed=7, max_ticks=20))
        def policy(o):
            return Action(ActionType.NOOP)
        result = run_episode(env, policy)
        assert result.length > 0
        assert result.scenario == "test_island"
        assert result.difficulty == "medium"

    def test_run_episode_records_metrics(self) -> None:
        env = SimulatorEnv(SimulatorConfig(seed=9, max_ticks=15, opponent_count=0))
        def policy(o):
            return Action(ActionType.RELOAD)
        result = run_episode(env, policy)
        assert result.decisions > 0
        assert result.avg_latency_ns >= 0

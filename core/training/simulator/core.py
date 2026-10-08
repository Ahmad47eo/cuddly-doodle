"""Training / simulation environment core.

This is the ONLY environment where the agent may have full control.
The simulator is a lightweight, deterministic, Gymnasium-style
environment used for research — it does NOT recreate the Fortnite
engine and it does NOT connect to any real Fortnite client.

Interface:

    reset() -> Observation
    step(Action) -> (Observation, Reward, bool, bool, dict)

Supports:

* Deterministic seeds
* Multiple scenarios
* Multiple difficulties
* Reward shaping
* Randomized opponents/teams
* Evaluation episodes
* Reproducible experiments
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Mapping, Optional

import gymnasium as gym
import numpy as np

from core.errors import SimulatorError
from core.types import (
    Action,
    ActionType,
    EntityState,
    EpisodeResult,
    MapInfo,
    Observation,
    Reward,
    RewardComponent,
    StrategyMode,
    TeamId,
    TerminationReason,
)


# ---------------------------------------------------------------------------
# Default maps / scenarios
# ---------------------------------------------------------------------------

_DEFAULT_MAPS: dict[str, MapInfo] = {
    "test_island": MapInfo(name="test_island", width=500.0, height=500.0, safe_zone_center=(250.0, 250.0), safe_zone_radius=200.0),
    "solo_loop": MapInfo(name="solo_loop", width=800.0, height=800.0, safe_zone_center=(400.0, 400.0), safe_zone_radius=350.0),
    "endgame_arena": MapInfo(name="endgame_arena", width=200.0, height=200.0, safe_zone_center=(100.0, 100.0), safe_zone_radius=60.0),
}


class ScenarioKind(str, Enum):
    SURVIVAL = "survival"
    DEATHMATCH = "deathmatch"
    ZONE_ROTATION = "zone_rotation"
    RESOURCE_GATHER = "resource_gather"
    ENDGAME = "endgame"
    CUSTOM = "custom"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Simulator config
# ---------------------------------------------------------------------------

@dataclass
class SimulatorConfig:
    scenario: str = "test_island"
    scenario_kind: ScenarioKind = ScenarioKind.SURVIVAL
    difficulty: Difficulty = Difficulty.MEDIUM
    max_ticks: int = 400
    seed: int | None = None
    opponent_count: int = 2
    team_count: int = 1
    reward_shaping: bool = True
    observation_history: int = 0  # unused by core sim; pipeline owns it
    low_ram_mode: bool = False


@dataclass
class SimulatorState:
    tick: int
    entities: dict[int, EntityState]
    items: dict[int, dict[str, Any]]
    zone_center: tuple[float, float]
    zone_radius: float
    events: Deque[str]
    self_id: int
    active: bool
    terminated: bool
    truncated: bool
    term_reason: TerminationReason | None
    total_reward: float
    metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class SimulatorEnv(gym.Env):
    """Lightweight Gymnasium environment for simulated gameplay research."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        super().__init__()
        self.config = config or SimulatorConfig()
        self._normalize_config()
        self._rng = np.random.default_rng(self.config.seed)
        self._map = _DEFAULT_MAPS.get(self.config.scenario, _DEFAULT_MAPS["test_island"])
        self._state: Optional[SimulatorState] = None
        self._action_history: Deque[str] = deque(maxlen=8)
        self._reward_acc: float = 0.0
        self._metrics: dict[str, Any] = {
            "steps_taken": 0,
            "total_reward": 0.0,
            "kills": 0,
            "deaths": 0,
            "resources_collected": 0,
            "objective_progress": 0.0,
            "accuracy_proxy": 0.0,
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = dict(options or {})
        self._configure_from_options(options)
        self._state = self._build_initial_state()
        obs = self._state_to_observation(self._state, last_action=None, last_reward=0.0)
        return obs, {}

    def step(self, action: Action) -> tuple[Observation, Reward, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise SimulatorError("Call reset() before step()")
        if self._state.terminated or self._state.truncated:
            raise SimulatorError("Environment already finished; call reset()")

        t0 = time.perf_counter()
        reward, events, metrics_delta = self._apply_action(action)
        elapsed_ns = int((time.perf_counter() - t0) * 1e9)
        action.latency_ns = elapsed_ns

        new_state = self._advance_state(self._state, action, reward, events)
        terminated = new_state.terminated
        truncated = new_state.truncated
        obs = self._state_to_observation(new_state, last_action=action.action_type.value, last_reward=reward.total)
        info = self._build_info(new_state, action, metrics_delta, elapsed_ns)

        self._state = new_state
        self._reward_acc += reward.total
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Any:
        if mode == "human":
            return self._render_text()
        if mode == "rgb_array":
            return self._render_array()
        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self) -> None:
        self._state = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _normalize_config(self) -> None:
        if isinstance(self.config.difficulty, str):
            self.config.difficulty = Difficulty(self.config.difficulty)

    def _configure_from_options(self, options: Mapping[str, Any]) -> None:
        if "scenario" in options:
            self.config.scenario = str(options["scenario"])
            self._map = _DEFAULT_MAPS.get(self.config.scenario, _DEFAULT_MAPS["test_island"])
        if "difficulty" in options:
            self.config.difficulty = Difficulty(str(options["difficulty"]))
        if "scenario_kind" in options:
            self.config.scenario_kind = ScenarioKind(str(options["scenario_kind"]))
        if "opponent_count" in options:
            self.config.opponent_count = int(options["opponent_count"])
        if "team_count" in options:
            self.config.team_count = int(options["team_count"])
        if "max_ticks" in options:
            self.config.max_ticks = int(options["max_ticks"])

    def _build_initial_state(self) -> SimulatorState:
        map_cfg = self._map
        entities: dict[int, EntityState] = {}
        self_id = 1
        entities[self_id] = EntityState(
            id=self_id,
            team=TeamId.SELF,
            x=map_cfg.safe_zone_center[0] + self._rng.normal(0, 40),
            y=map_cfg.safe_zone_center[1] + self._rng.normal(0, 40),
            health=100.0,
            Shield=50.0,
            loaded=True,
        )
        ally_count = max(0, self.config.team_count - 1)
        next_id = 2
        for _ in range(ally_count):
            entities[next_id] = EntityState(
                id=next_id,
                team=TeamId.ALLY,
                x=map_cfg.safe_zone_center[0] + self._rng.normal(0, 120),
                y=map_cfg.safe_zone_center[1] + self._rng.normal(0, 120),
                health=100.0,
                Shield=25.0,
                loaded=self._rng.random() > 0.3,
            )
            next_id += 1

        enemy_count = self.config.opponent_count
        for _ in range(enemy_count):
            entities[next_id] = EntityState(
                id=next_id,
                team=TeamId.ENEMY,
                x=map_cfg.safe_zone_center[0] + self._rng.normal(0, 180),
                y=map_cfg.safe_zone_center[1] + self._rng.normal(0, 180),
                health=100.0,
                Shield=25.0,
                loaded=self._rng.random() > 0.25,
            )
            next_id += 1

        items = self._generate_items(6 + enemy_count * 2)

        difficulty_mul = {"easy": 0.7, "medium": 1.0, "hard": 1.4}[self.config.difficulty.value]
        zone_radius = map_cfg.safe_zone_radius * (0.6 if self.config.scenario_kind is ScenarioKind.ENDGAME else 1.0) * difficulty_mul

        return SimulatorState(
            tick=0,
            entities=entities,
            items=items,
            zone_center=map_cfg.safe_zone_center,
            zone_radius=zone_radius,
            events=deque(maxlen=32),
            self_id=self_id,
            active=True,
            terminated=False,
            truncated=False,
            term_reason=None,
            total_reward=0.0,
            metrics=dict(self._metrics),
        )

    def _generate_items(self, count: int) -> dict[int, dict[str, Any]]:
        items: dict[int, dict[str, Any]] = {}
        kinds = ["ammo", "health", "shield", "mat", "metal", "food"]
        for i in range(count):
            x = self._map.safe_zone_center[0] + self._rng.normal(0, 180)
            y = self._map.safe_zone_center[1] + self._rng.normal(0, 180)
            items[i + 100] = {
                "id": i + 100,
                "type": kinds[int(self._rng.integers(0, len(kinds)))],
                "x": x,
                "y": y,
                "amount": float(self._rng.integers(5, 30)),
                "collected": False,
            }
        return items

    def _apply_action(self, action: Action) -> tuple[Reward, list[str], dict[str, float]]:
        events: list[str] = []
        reward = Reward(total=0.0)
        metrics_delta: dict[str, float] = {}

        state = self._state
        self_entity = state.entities[state.self_id]
        at = action.action_type

        if at is ActionType.NOOP:
            pass
        elif at is ActionType.MOVE:
            dx = float(action.params.get("dx", 0.0))
            dy = float(action.params.get("dy", 0.0))
            mag = math.hypot(dx, dy)
            max_step = 40.0 if self.config.difficulty is Difficulty.EASY else 25.0
            step = min(mag, max_step) if mag > 0 else max_step * 0.1
            self_entity.x += dx / (mag or 1.0) * step
            self_entity.y += dy / (mag or 1.0) * step
            self_entity.moving = step > 0.01
            self._clamp_to_map(self_entity)
        elif at is ActionType.ROTATE:
            pass
        elif at is ActionType.AIM:
            target = action.target
            if target is not None and target in state.entities:
                t = state.entities[target]
                self_entity.aiming = True
        elif at is ActionType.FIRE:
            if not self_entity.loaded:
                reward = reward.add(RewardComponent.PENALTY, -0.5)
                events.append("fire_empty_weapon")
            else:
                target = action.target
                hit = False
                if target is not None and target in state.entities:
                    t = state.entities[target]
                    dmg = 8.0
                    t.health -= dmg
                    hit = True
                    reward = reward.add(RewardComponent.DAMAGE_DEALT, dmg * 0.02)
                    self_entity.aiming = True
                    self_entity.loaded = self._rng.random() > 0.05
                if hit:
                    events.append("shot_fired")
                else:
                    events.append("shot_missed")
        elif at is ActionType.RELOAD:
            self_entity.loaded = True
            events.append("reloaded")
            reward = reward.add(RewardComponent.RESOURCE, 0.1)
        elif at is ActionType.BUILD:
            if self_entity.build_cooldown > 0:
                reward = reward.add(RewardComponent.PENALTY, -0.2)
            else:
                self_entity.build_cooldown = 8.0 if self.config.difficulty is Difficulty.EASY else 12.0
                events.append("building")
        elif at is ActionType.USE_ITEM:
            self_entity.health = min(100.0, self_entity.health + 15.0)
            self_entity.Shield = min(100.0, self_entity.Shield + 10.0)
            events.append("used_item")
            reward = reward.add(RewardComponent.RESOURCE, 0.2)
        elif at is ActionType.DROP_ITEM:
            events.append("dropped_item")
        elif at is ActionType.INTERACT:
            events.append("interacted")
        elif at is ActionType.CHANGE_STRATEGY:
            pass
        elif at is ActionType.HIGH_LEVEL_PLAN:
            pass

        # resource collection proximity
        collected = False
        for item in state.items.values():
            if item["collected"]:
                continue
            if math.hypot(self_entity.x - item["x"], self_entity.y - item["y"]) < 20.0:
                item["collected"] = True
                collected = True
                if item["type"] in ("ammo", "mat", "metal"):
                    self_entity.loaded = True
                    reward = reward.add(RewardComponent.RESOURCE, 0.4)
                elif item["type"] in ("health", "food"):
                    self_entity.health = min(100.0, self_entity.health + item["amount"] * 0.5)
                    reward = reward.add(RewardComponent.RESOURCE, 0.3)
                elif item["type"] == "shield":
                    self_entity.Shield = min(100.0, self_entity.Shield + item["amount"] * 0.4)
                    reward = reward.add(RewardComponent.RESOURCE, 0.3)

        # survival reward (per tick alive)
        if self.config.reward_shaping:
            reward = reward.add(RewardComponent.SURVIVAL, 0.02)
            zr = state.zone_radius
            zc = state.zone_center
            dist = math.hypot(self_entity.x - zc[0], self_entity.y - zc[1])
            if dist > zr:
                reward = reward.add(RewardComponent.PENALTY, -0.1 * difficulty_mul)

        # objective progress proxy for zone rotation / endgame
        if self.config.scenario_kind in (ScenarioKind.ZONE_ROTATION, ScenarioKind.ENDGAME):
            progress = 1.0 - (dist / (self._map.safe_zone_radius or 1.0))
            metrics_delta["objective_progress"] = max(0.0, min(1.0, progress))

        self._update_metrics(state, action, hit=(at is ActionType.FIRE), collected=collected, metrics_delta=metrics_delta, events=events)
        return reward, events, metrics_delta

    def _apply_enemy_ai(self, state: SimulatorState) -> None:
        difficulty = self.config.difficulty
        aggression = {"easy": 0.2, "medium": 0.5, "hard": 0.8}[difficulty.value]
        for eid, e in state.entities.items():
            if e.id == state.self_id or e.team != TeamId.ENEMY:
                continue
            if e.health <= 0:
                continue
            self_e = state.entities[state.self_id]
            d = math.hypot(e.x - self_e.x, e.y - self_e.y)
            if self_e.health > 0 and (e.loaded or self._rng.random() < aggression) and d < 220.0:
                dx = self_e.x - e.x
                dy = self_e.y - e.y
                mag = math.hypot(dx, dy) or 1.0
                step = min(22.0, mag)
                e.x += dx / mag * step
                e.y += dy / mag * step
                e.moving = True
                e.aiming = True
                if self._rng.random() < 0.4:
                    dmg = 6.0 if difficulty is Difficulty.HARD else 4.0
                    self_e.health -= dmg
                    state.events.append(f"enemy_{eid}_shot")
                    if self_e.health <= 0:
                        self_e.health = 0.0
        # ally simple support behavior
        for eid, e in state.entities.items():
            if e.id == state.self_id or e.team != TeamId.ALLY:
                continue
            if e.health <= 0:
                continue
            self_e = state.entities[state.self_id]
            d = math.hypot(e.x - self_e.x, e.y - self_e.y)
            if d > 180.0:
                dx = self_e.x - e.x
                dy = self_e.y - e.y
                mag = math.hypot(dx, dy) or 1.0
                step = min(18.0, mag)
                e.x += dx / mag * step
                e.y += dy / mag * step
                e.moving = True

    def _advance_state(self, state: SimulatorState, action: Action, reward: Reward, events: list[str]) -> SimulatorState:
        state = SimulatorState(
            tick=state.tick + 1,
            entities={k: EntityState(**vars(v)) for k, v in state.entities.items()},
            items={k: dict(v) for k, v in state.items.items()},
            zone_center=state.zone_center,
            zone_radius=state.zone_radius * (0.998 if self.config.scenario_kind is ScenarioKind.ZONE_ROTATION else 1.0),
            events=deque(state.events, maxlen=32),
            self_id=state.self_id,
            active=True,
            terminated=False,
            truncated=False,
            term_reason=state.term_reason,
            total_reward=state.total_reward + reward.total,
            metrics=dict(state.metrics),
        )
        # cooldown decay
        for ent in state.entities.values():
            if ent.build_cooldown > 0:
                ent.build_cooldown = max(0.0, ent.build_cooldown - 1.0)
        # zone shrink events
        new_r = state.zone_radius
        old_r = self._map.safe_zone_radius * (0.6 if self.config.scenario_kind is ScenarioKind.ENDGAME else 1.0)
        if new_r < old_r and abs(new_r - old_r + 0.2) < 0.05:
            state.events.append("zone_shrinking")
        for ev in events:
            state.events.append(ev)
        # enemy ai
        self._apply_enemy_ai(state)
        # termination checks
        self_e = state.entities[state.self_id]
        if self_e.health <= 0:
            state.terminated = True
            state.term_reason = TerminationReason.HEALTH_ZERO
        if state.tick >= self.config.max_ticks:
            state.truncated = True
        # deathmatch win condition
        if self.config.scenario_kind is ScenarioKind.DEATHMATCH:
            enemies = [e for e in state.entities.values() if e.team == TeamId.ENEMY and e.health > 0]
            if not enemies:
                state.terminated = True
                state.term_reason = TerminationReason.OBJECTIVE_COMPLETE
        return state

    def _state_to_observation(self, state: SimulatorState, last_action: str | None, last_reward: float) -> Observation:
        mode = StrategyMode.BALANCED
        return Observation(
            tick=state.tick,
            timestamp=time.time(),
            scenario=self.config.scenario,
            difficulty=self.config.difficulty.value,
            strategy_mode=mode,
            map_info=self._map,
            self_entity=state.entities[state.self_id],
            entities=state.entities,
            items=state.items,
            last_action=last_action,
            last_reward=last_reward,
            events=list(state.events),
            confidence=1.0,
            features={
                "self_health": state.entities[state.self_id].health,
                "self_shield": state.entities[state.self_id].Shield,
                "enemy_count": sum(1 for e in state.entities.values() if e.team == TeamId.ENEMY and e.health > 0),
                "ally_count": sum(1 for e in state.entities.values() if e.team == TeamId.ALLY and e.health > 0),
                "zone_radius": state.zone_radius,
                "distance_to_zone_center": math.hypot(
                    state.entities[state.self_id].x - state.zone_center[0],
                    state.entities[state.self_id].y - state.zone_center[1],
                ),
                "resources_nearby": sum(1 for it in state.items.values() if not it["collected"]),
                "loaded": state.entities[state.self_id].loaded,
            },
        )

    def _build_info(
        self,
        state: SimulatorState,
        action: Action,
        metrics_delta: dict[str, float],
        latency_ns: int,
    ) -> dict[str, Any]:
        info = {
            "tick": state.tick,
            "terminated": state.terminated,
            "truncated": state.truncated,
            "term_reason": state.term_reason.value if state.term_reason else None,
            "action_latency_ns": latency_ns,
            "metrics": {**state.metrics, **metrics_delta},
            "events": list(state.events)[-5:],
            "state_dict": self._serialize_state(state),
        }
        return info

    def _serialize_state(self, state: SimulatorState) -> dict[str, Any]:
        return {
            "tick": state.tick,
            "self_id": state.self_id,
            "entities": {str(k): {"id": v.id, "team": int(v.team), "x": v.x, "y": v.y, "health": v.health, "shield": v.Shield, "loaded": v.loaded} for k, v in state.entities.items()},
            "zone_center": state.zone_center,
            "zone_radius": state.zone_radius,
        }

    def _update_metrics(
        self,
        state: SimulatorState,
        action: Action,
        hit: bool,
        collected: bool,
        metrics_delta: dict[str, float],
        events: list[str],
    ) -> None:
        m = self._metrics
        if hit:
            m["accuracy_proxy"] = max(m["accuracy_proxy"], 0.1)
        if collected:
            m["resources_collected"] = m.get("resources_collected", 0) + 1

    def _clamp_to_map(self, ent: EntityState) -> None:
        w, h = self._map.width, self._map.height
        ent.x = max(5.0, min(w - 5.0, ent.x))
        ent.y = max(5.0, min(h - 5.0, ent.y))

    def _render_text(self) -> str:
        if self._state is None:
            return "No active episode."
        s = self._state
        self_e = s.entities[s.self_id]
        lines = [
            f"Tick: {s.tick}/{self.config.max_ticks}",
            f"Zone: r={s.zone_radius:.1f} @ {s.zone_center}",
            f"Self: HP={self_e.health:.0f} SH={self_e.Shield:.0f} {'loaded' if self_e.loaded else 'empty'} at ({self_e.x:.0f},{self_e.y:.0f})",
            f"Enemies alive: {sum(1 for e in s.entities.values() if e.team==TeamId.ENEMY and e.health>0)}",
            f"Allies alive: {sum(1 for e in s.entities.values() if e.team==TeamId.ALLY and e.health>0)}",
            f"Items remaining: {sum(1 for it in s.items.values() if not it['collected'])}",
            f"Events: {', '.join(list(s.events)[-6:])}",
        ]
        return "\n".join(lines)

    def _render_array(self) -> np.ndarray:
        # Very small placeholder array; the UI/rendering layer can upgrade later.
        h, w = 48, 64
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        if self._state is None:
            return arr
        s = self._state
        scale_x = w / self._map.width
        scale_y = h / self._map.height
        for eid, ent in s.entities.items():
            color = (255, 50, 50) if ent.team is TeamId.ENEMY else (50, 150, 255) if ent.team is TeamId.ALLY else (50, 255, 120)
            px = int(ent.x * scale_x)
            py = int(ent.y * scale_y)
            if 0 <= px < w and 0 <= py < h:
                arr[py, px] = color
        return arr


# ---------------------------------------------------------------------------
# Convenience runners (used by evaluator and tests)
# ------------------------------------------------------------------

def run_episode(
    env: SimulatorEnv,
    policy: Callable[[Observation], Action],
    *,
    render: bool = False,
) -> EpisodeResult:
    obs, _ = env.reset()
    actions: list[Action] = []
    total_reward = 0.0
    lengths = 0
    accuracies: list[float] = []
    latencies: list[int] = []
    while True:
        if render:
            env.render(mode="human")
        action = policy(obs)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward.total
        lengths += 1
        latencies.append(action.latency_ns)
        accuracies.append(action.confidence)
        if terminated or truncated:
            break
    term_reason = None
    if terminated and obs.last_action is not None:
        term_reason = info.get("term_reason")
        if term_reason:
            try:
                term_reason = TerminationReason(term_reason)
            except ValueError:
                term_reason = None
    else:
        term_reason = TerminationReason.TIME_LIMIT if truncated else None
    return EpisodeResult(
        episode_id="",
        scenario=obs.scenario,
        difficulty=obs.difficulty,
        strategy_mode=obs.strategy_mode,
        length=lengths,
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        termination_reason=term_reason,
        accuracy=float(np.mean(accuracies)) if accuracies else 0.0,
        avg_latency_ns=int(np.mean(latencies)) if latencies else 0,
        decisions=len(actions),
        kills=int(info.get("metrics", {}).get("kills", 0)),
        deaths=int(info.get("metrics", {}).get("deaths", 0)),
        resources_collected=int(info.get("metrics", {}).get("resources_collected", 0)),
        objective_progress=float(info.get("metrics", {}).get("objective_progress", 0.0)),
    )

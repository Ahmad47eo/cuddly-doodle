"""Core data contracts for the Fortnite AI Research platform.

These types are deliberately framework-agnostic so that simulator,
policy, model-routing, observation, and coaching layers can all share
the same contracts without importing one another's implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Mapping, Sequence, TypedDict

# ---------------------------------------------------------------------------
# Strategy modes
# ---------------------------------------------------------------------------

class StrategyMode(str, Enum):
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    COMPETITIVE = "competitive"
    AIM_FOCUSED = "aim_focused"
    MOBILITY_FOCUSED = "mobility_focused"
    BUILD_EDIT_FOCUSED = "build_edit_focused"
    TEAM_SUPPORT = "team_support"
    RESOURCE_FOCUSED = "resource_focused"
    ENDGAME = "endgame"
    SURVIVAL = "survival"
    OBJECTIVE_FOCUSED = "objective_focused"
    ADAPTIVE = "adaptive"
    CUSTOM = "custom"

# ---------------------------------------------------------------------------
# Observation / action spaces (scalar-centric, low-overhead)
# ---------------------------------------------------------------------------

class SpaceType(str, Enum):
    DISCRETE = "discrete"
    BOX = "box"
    DICT = "dict"


@dataclass(frozen=True)
class Space:
    """A minimal description of an observation or action component."""

    name: str
    space_type: SpaceType
    shape: tuple[int, ...] = ()
    low: float | None = None
    high: float | None = None
    dtype: str = "float32"
    options: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.space_type is SpaceType.DISCRETE and self.options is None:
            raise ValueError("DISCRETE space must declare options")
        if self.space_type is SpaceType.BOX and self.shape == ():
            object.__setattr__(self, "shape", (1,))


# ---------------------------------------------------------------------------
# Game state / observation
# ---------------------------------------------------------------------------

class TeamId(IntEnum):
    SELF = 0
    ALLY = 1
    ENEMY = 2
    UNKNOWN = 3


@dataclass
class EntityState:
    """Compact state of one entity in the simulation."""

    id: int
    team: TeamId
    x: float
    y: float
    health: float
    Shield: float = 0.0
    loaded: bool = False
    aiming: bool = False
    moving: bool = False
    build_cooldown: float = 0.0


@dataclass
class MapInfo:
    """Static / slowly-changing map metadata for a scenario."""

    name: str
    width: float
    height: float
    safe_zone_center: tuple[float, float] = (0.0, 0.0)
    safe_zone_radius: float = 0.0
    loot_density: float = 0.25


@dataclass
class Observation:
    """A single simulator observation. Immutable after construction."""

    tick: int
    timestamp: float
    scenario: str
    difficulty: str
    strategy_mode: StrategyMode
    map_info: MapInfo
    self_entity: EntityState
    entities: Mapping[int, EntityState] = field(default_factory=dict)
    items: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    last_action: str | None = None
    last_reward: float = 0.0
    events: Sequence[str] = ()
    confidence: float = 1.0
    features: Mapping[str, float] = field(default_factory=dict)
    time_remaining: float = 200.0

    def shallow_copy_with(self, **overrides: Any) -> Observation:
        """Return a new observation with selected fields overridden."""
        data = {
            "tick": self.tick,
            "timestamp": self.timestamp,
            "scenario": self.scenario,
            "difficulty": self.difficulty,
            "strategy_mode": self.strategy_mode,
            "map_info": self.map_info,
            "self_entity": self.self_entity,
            "entities": self.entities,
            "items": self.items,
            "last_action": self.last_action,
            "last_reward": self.last_reward,
            "events": self.events,
            "confidence": self.confidence,
            "features": self.features,
        }
        data.update(overrides)
        return Observation(**data)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    NOOP = "noop"
    MOVE = "move"
    ROTATE = "rotate"
    AIM = "aim"
    FIRE = "fire"
    RELOAD = "reload"
    BUILD = "build"
    EDIT = "edit"
    USE_ITEM = "use_item"
    DROP_ITEM = "drop_item"
    INTERACT = "interact"
    CHANGE_STRATEGY = "change_strategy"
    HIGH_LEVEL_PLAN = "high_level_plan"


@dataclass
class Action:
    """One atomic simulated action emitted by the policy."""

    action_type: ActionType
    target: int | None = None
    params: Mapping[str, float] = field(default_factory=dict)
    strategy: StrategyMode | None = None
    confidence: float = 1.0
    latency_ns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "target": self.target,
            "params": dict(self.params),
            "strategy": self.strategy.value if self.strategy else None,
            "confidence": self.confidence,
            "latency_ns": self.latency_ns,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Action:
        at = ActionType(d["action_type"]) if "action_type" in d else ActionType.NOOP
        st = None
        if d.get("strategy"):
            try:
                st = StrategyMode(d["strategy"])
            except ValueError:
                st = None
        return cls(
            action_type=at,
            target=d.get("target"),
            params=d.get("params", {}),
            strategy=st,
            confidence=float(d.get("confidence", 1.0)),
            latency_ns=int(d.get("latency_ns", 0)),
        )


# ---------------------------------------------------------------------------
# Reward / outcome
# ---------------------------------------------------------------------------

class RewardComponent(str, Enum):
    SURVIVAL = "survival"
    DAMAGE_DEALT = "damage_dealt"
    DAMAGE_TAKEN = "damage_taken"
    KILL = "kill"
    OBJECTIVE = "objective"
    RESOURCE = "resource"
    POSITIONING = "positioning"
    TIMING = "timing"
    TEAM_SCORE = "team_score"
    PENALTY = "penalty"


@dataclass
class Reward:
    """Decomposition of a single-step reward for analysis / shaping."""

    total: float
    components: Mapping[RewardComponent, float] = field(default_factory=dict)
    note: str = ""

    def add(self, component: RewardComponent, value: float) -> Reward:
        new_components = dict(self.components)
        new_components[component] = new_components.get(component, 0.0) + value
        return Reward(total=self.total + value, components=new_components, note=self.note)


class TerminationReason(str, Enum):
    WIN = "win"
    LOSS = "loss"
    TIME_LIMIT = "time_limit"
    HEALTH_ZERO = "health_zero"
    OBJECTIVE_COMPLETE = "objective_complete"
    MANUAL = "manual"


@dataclass
class EpisodeResult:
    """Outcome of one full simulated episode."""

    episode_id: str
    scenario: str
    difficulty: str
    strategy_mode: StrategyMode
    length: int
    total_reward: float
    terminated: bool
    truncated: bool
    termination_reason: TerminationReason | None = None
    accuracy: float = 0.0
    avg_latency_ns: int = 0
    decisions: int = 0
    kills: int = 0
    deaths: int = 0
    resources_collected: int = 0
    objective_progress: float = 0.0


# ---------------------------------------------------------------------------
# Memory contracts
# ---------------------------------------------------------------------------

class MemoryTier(str, Enum):
    SHORT_TERM = "short_term"
    EPISODIC = "episodic"
    SKILL = "skill"
    STRATEGY = "strategy"


@dataclass
class MemoryEntry:
    """A single record stored in one memory tier."""

    tier: MemoryTier
    key: str
    payload: Mapping[str, Any]
    age: float = 0.0
    priority: float = 0.0

    @property
    def size_bytes(self) -> int:
        """Estimated serialized size in bytes."""
        try:
            return len(str(self.tier.value).encode()) + len(self.key.encode()) + len(str(self.payload).encode()) + 16
        except Exception:
            return 64


class MemoryStats(TypedDict):
    tier: str
    count: int
    bytes_used: int
    max_entries: int
    max_bytes: int

"""Virtual keyboard/mouse input for permitted training environments.

This module deliberately models input state without touching the OS, desktop,
or any real game client. Adapters for simulators/custom environments can
consume the state through snapshot().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet


@dataclass(frozen=True)
class MouseState:
    dx: float = 0.0
    dy: float = 0.0
    left: bool = False
    right: bool = False
    middle: bool = False
    wheel: float = 0.0


@dataclass(frozen=True)
class VirtualInputSnapshot:
    keys: FrozenSet[str] = field(default_factory=frozenset)
    mouse: MouseState = field(default_factory=MouseState)


class VirtualInputController:
    """Pure in-memory input state for the custom simulator/authorized env."""

    _MAX_MOUSE_DELTA = 1000.0
    _MAX_WHEEL = 20.0

    def __init__(self) -> None:
        self._keys: set[str] = set()
        self._mouse = MouseState()

    def key_down(self, key: str) -> None:
        self._keys.add(self._normalize_key(key))

    def key_up(self, key: str) -> None:
        self._keys.discard(self._normalize_key(key))

    def set_mouse(
        self,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        left: bool = False,
        right: bool = False,
        middle: bool = False,
        wheel: float = 0.0,
    ) -> None:
        self._mouse = MouseState(
            dx=self._clamp(float(dx), -self._MAX_MOUSE_DELTA, self._MAX_MOUSE_DELTA),
            dy=self._clamp(float(dy), -self._MAX_MOUSE_DELTA, self._MAX_MOUSE_DELTA),
            left=bool(left),
            right=bool(right),
            middle=bool(middle),
            wheel=self._clamp(float(wheel), -self._MAX_WHEEL, self._MAX_WHEEL),
        )

    def snapshot(self) -> VirtualInputSnapshot:
        return VirtualInputSnapshot(
            keys=frozenset(self._keys),
            mouse=self._mouse,
        )

    def clear(self) -> None:
        self._keys.clear()
        self._mouse = MouseState()

    @staticmethod
    def _normalize_key(key: str) -> str:
        value = str(key).strip().upper()
        if not value or len(value) > 32:
            raise ValueError("key must be a non-empty name of at most 32 characters")
        return value

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

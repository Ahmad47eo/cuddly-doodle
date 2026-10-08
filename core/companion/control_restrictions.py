"""Real-game control restrictions — the hard safety boundary.

Real Fortnite MUST remain observation / analysis / coaching only.
Every real-game control method here FAILS CLOSED and raises
`SafetyViolationError`. There is no "enable real-game control" switch.

The companion may analyze permitted observation data, but it must
NEVER convert that observation into Fortnite control commands.

Architecture tests verify this separation:
- `core/training` must not receive live Fortnite control input.
- `core/companion` must not import real control implementations that
  could reach the real client.
- The build should fail if forbidden control APIs are introduced.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.errors import SafetyViolationError

# ---------------------------------------------------------------------------
# What counts as a forbidden control category
# ---------------------------------------------------------------------------

_FORBIDDEN_CONTROL_CATEGORIES: tuple[str, ...] = (
    "keyboard_automation",
    "mouse_automation",
    "controller_automation",
    "os_input_simulation",
    "input_injection",
    "dll_injection",
    "process_manipulation",
    "game_memory_read",
    "game_memory_write",
    "packet_manipulation",
    "anticheat_bypass",
    "exploit_automation",
    "account_automation",
    "ranked_automation",
    "tournament_automation",
    "detection_evasion",
    "anticheat_evasion",
    "credential_collection",
    "login_automation",
    "client_modification",
    "press_key",
)


def _safety_violation(
    operation: str,
    *,
    detail: str = "",
) -> SafetyViolationError:
    msg = (
        f"FORBIDDEN: real-game control operation '{operation}' is not allowed. "
        "Real Fortnite remains observation / analysis / coaching only."
    )
    if detail:
        msg = f"{msg} {detail}"
    return SafetyViolationError(msg, attempted=operation)


# ---------------------------------------------------------------------------
# RealFortniteControlInterface — fail closed by design
# ---------------------------------------------------------------------------

class RealFortniteControlInterface:
    """Interface describing operations that WOULD control real Fortnite.

    Every method raises `SafetyViolationError` immediately. Implementations
    of real-game control must never exist in this codebase. If one appears,
    the architecture tests catch it and the build is treated as failed.
    """

    def move(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.move", detail="Keyboard/mouse/controller movement toward the real Fortnite client is forbidden.")

    def rotate_view(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.rotate_view", detail="View rotation toward the real Fortnite client is forbidden.")

    def aim(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.aim", detail="Aiming at the real Fortnite client is forbidden.")

    def fire(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.fire", detail="Firing toward the real Fortnite client is forbidden.")

    def reload(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.reload", detail="Reloading toward the real Fortnite client is forbidden.")

    def build(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.build", detail="Building toward the real Fortnite client is forbidden.")

    def edit(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.edit", detail="Editing toward the real Fortnite client is forbidden.")

    def drop_item(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.drop_item", detail="Item dropping toward the real Fortnite client is forbidden.")

    def use_item(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.use_item", detail="Item use toward the real Fortnite client is forbidden.")

    def interact(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.interact", detail="Interaction toward the real Fortnite client is forbidden.")

    def click(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.click", detail="Mouse/controller click toward the real Fortnite client is forbidden.")

    def press_key(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.press_key", detail="Keyboard press toward the real Fortnite client is forbidden.")

    def set_input_state(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.set_input_state", detail="Input injection toward the real Fortnite client is forbidden.")

    def read_game_memory(self, **kwargs: Any) -> Any:
        raise _safety_violation("real_fortnite_control.read_game_memory", detail="Game memory reading is forbidden.")

    def write_game_memory(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.write_game_memory", detail="Game memory writing is forbidden.")

    def hook_process(self, **kwargs: Any) -> Any:
        raise _safety_violation("real_fortnite_control.hook_process", detail="Process/DLL hooking is forbidden.")

    def send_packet(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.send_packet", detail="Packet manipulation is forbidden.")

    def capture_screen(self, **kwargs: Any) -> Any:
        raise _safety_violation("real_fortnite_control.capture_screen", detail="Direct in-game screen capture that implies control coupling is forbidden. Permitted observation uses user-provided or separately-authorized screenshots/videos only.")

    def login(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.login", detail="Login/account automation is forbidden.")

    def automate_account(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.automate_account", detail="Account automation / botting is forbidden.")

    def evade_detection(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.evade_detection", detail="Anti-cheat / detection evasion is forbidden.")

    def modify_client(self, **kwargs: Any) -> None:
        raise _safety_violation("real_fortnite_control.modify_client", detail="Client modification is forbidden.")


# ---------------------------------------------------------------------------
# Safety assertion helpers used by architecture tests
# ---------------------------------------------------------------------------

def assert_no_real_control_import() -> None:
    """Public marker for architecture/safety tests.

    The real enforcement lives in the test suite and the static scan, but
    this symbol gives the test runner a stable hook to assert the boundary
    is intact at runtime.
    """
    # Verifying the interface itself fails closed is the key invariant.
    ctrl = RealFortniteControlInterface()
    try:
        ctrl.move()
    except SafetyViolationError:
        return
    raise AssertionError("RealFortniteControlInterface.move() did not raise SafetyViolationError")


def list_forbidden_categories() -> tuple[str, ...]:
    """Return the categories that must never appear in real-game control."""
    return _FORBIDDEN_CONTROL_CATEGORIES

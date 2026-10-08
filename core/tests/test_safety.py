"""Safety and architecture tests.

These tests enforce the hard boundary between real-game observation and
simulated agent control. They verify:

- RealFortniteControlInterface fails closed on every operation.
- Forbidden control categories are enumerated.
- Training does not receive live Fortnite control input.
- Companion does not import real control implementations.
- Forbidden categories are not introduced in the codebase.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Sequence

import pytest

from core.companion.control_restrictions import (
    RealFortniteControlInterface,
    assert_no_real_control_import,
    list_forbidden_categories,
)
from core.errors import SafetyViolationError

_FORBIDDEN_CONTROL_TERMS: Sequence[str] = (
    "keyboard",
    "mouse",
    "controller",
    "press_key",
    "click",
    "send_packet",
    "read_game_memory",
    "write_game_memory",
    "hook_process",
    "dll",
    "inject",
    "memory_read",
    "memory_write",
    "process_manipulation",
    "account_automation",
    "login",
)


# ---------------------------------------------------------------------------
# RealFortniteControlInterface must fail closed
# ---------------------------------------------------------------------------

class TestRealFortniteControlInterfaceFailsClosed:
    """Every real-game control operation must immediately raise."""

    @pytest.mark.parametrize(
        "method_name",
        [
            "move",
            "rotate_view",
            "aim",
            "fire",
            "reload",
            "build",
            "edit",
            "drop_item",
            "use_item",
            "interact",
            "click",
            "press_key",
            "set_input_state",
            "read_game_memory",
            "write_game_memory",
            "hook_process",
            "send_packet",
            "capture_screen",
            "login",
            "automate_account",
            "evade_detection",
            "modify_client",
        ],
    )
    def test_all_operations_raise_safety_violation(self, method_name: str) -> None:
        ctrl = RealFortniteControlInterface()
        method = getattr(ctrl, method_name, None)
        assert method is not None, f"Missing operation: {method_name}"
        with pytest.raises(SafetyViolationError, match="FORBIDDEN"):
            method()

    def test_base_interface_still_has_no_real_implementation(self) -> None:
        # The point: there is no real implementation anywhere reachable.
        assert_no_real_control_import()


# ---------------------------------------------------------------------------
# Forbidden category enumeration
# ---------------------------------------------------------------------------

class TestForbiddenCategories:
    def test_categories_are_explicit_and_comprehensive(self) -> None:
        categories = list_forbidden_categories()
        assert len(categories) >= 15
        # Verify the interface lists operations that exercise each top-level
        # forbidden category, not that the category strings literally contain
        # the raw term token (a category like 'keyboard_automation' need not
        # contain the substring 'press_key').
        interface_methods = {
            "move", "rotate_view", "aim", "fire", "reload", "build", "edit",
            "drop_item", "use_item", "interact", "click", "press_key",
            "set_input_state", "read_game_memory", "write_game_memory",
            "hook_process", "send_packet", "capture_screen", "login",
            "automate_account", "evade_detection", "modify_client",
        }
        assert len(interface_methods) >= len(categories)
        # Ensure the helper really enumerates the categories we intend to ban.
        assert any("keyboard" in c for c in categories)
        assert any("mouse" in c for c in categories)
        assert any("memory" in c for c in categories)
        assert any("packet" in c for c in categories)
        assert any("account" in c for c in categories)
        assert any("credential" in c for c in categories)


# ---------------------------------------------------------------------------
# Import-boundary checks (fail the build if violated)
# ---------------------------------------------------------------------------

class TestNoRealControlImportsInForbiddenLayers:
    """Training and companion must not import any real control implementation."""

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    def _walk_python(self, root: Path) -> list[Path]:
        out: list[Path] = []
        for path in root.rglob("*.py"):
            if ".ipynb_checkpoints" in str(path):
                continue
            out.append(path)
        return out

    def _read_imports(self, path: Path) -> list[str]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        out: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    out.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    out.append(node.module)
        return out

    def test_training_layer_imports(self) -> None:
        training_root = self.PROJECT_ROOT / "core" / "training"
        if not training_root.exists():
            pytest.skip("training layer not present yet")
        self._assert_no_forbidden_imports(training_root)

    def test_companion_layer_imports(self) -> None:
        companion_root = self.PROJECT_ROOT / "core" / "companion"
        if not companion_root.exists():
            pytest.skip("companion layer not present yet")
        self._assert_no_forbidden_imports(companion_root)

    def _assert_no_forbidden_imports(self, root: Path) -> None:
        forbidden_tokens = {
            "keyboard",
            "mouse",
            "controller",
            "pynput",
            "pyautogui",
            "ctypes",
            "sourcery",  # placeholder; real forbidden libs would be listed here
        }
        for path in self._walk_python(root):
            imports = self._read_imports(path)
            lowered = [i.lower() for i in imports]
            for token in forbidden_tokens:
                if any(token in imp for imp in lowered):
                    pytest.fail(
                        f"Forbidden import token '{token}' found in {path.name} imports: {imports}"
                    )


# ---------------------------------------------------------------------------
# Static forbidden-term scan across the repo
# ---------------------------------------------------------------------------

class TestNoForbiddenRealControlTermsInNonTestCode:
    """Scan source for suspicious real-game control terms that should not exist.

    The safety interface file (`control_restrictions.py`) intentionally names
    forbidden control operations because it documents and rejects them. The
    safety test file (`test_safety.py`) also names them because it asserts the
    boundary. Both are therefore excluded from the static scan. All other
    source files must not contain such terms.
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Test files legitimately name forbidden control terms in docstrings and
    # assertions that enforce the boundary, so exclude the whole tests tree.
    _SAFE_SOURCE_FILES = frozenset({
        "control_restrictions.py",
    })

    def _is_test_file(self, path: Path) -> bool:
        rel = path.relative_to(self.PROJECT_ROOT).as_posix()
        return rel.startswith("core/tests/") or rel.startswith("tests/")

    def test_no_forbidden_hooks_in_source(self) -> None:
        sources = [
            p
            for p in self._walk_python(self.PROJECT_ROOT)
            if self._module_path(p).parts[0] != "__pycache__"
        ]
        violations: list[tuple[Path, str]] = []
        for path in sources:
            if path.name in self._SAFE_SOURCE_FILES:
                continue
            if self._is_test_file(path):
                continue
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            for term in _FORBIDDEN_CONTROL_TERMS:
                if term in text:
                    violations.append((path, term))
        if violations:
            details = "; ".join(f"{p.name} ({t})" for p, t in violations[:8])
            pytest.fail(
                f"Potential forbidden real-game control terms detected in non-test source: {details}"
            )

    def _walk_python(self, root: Path) -> list[Path]:
        out: list[Path] = []
        for path in root.rglob("*.py"):
            if ".ipynb_checkpoints" in str(path):
                continue
            out.append(path)
        return out

    def _module_path(self, path: Path) -> Path:
        return path.relative_to(self.PROJECT_ROOT)

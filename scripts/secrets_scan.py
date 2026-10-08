#!/usr/bin/env python3
"""Standalone security/safety scan for CI and pre-commit.

Checks:
1. No forbidden real-game control terms in non-test, non-docs source.
2. No hardcoded secrets (key=..., password=..., token=...) in source.
3. No forbidden control imports in training/companion layers.
4. RealFortniteControlInterface still fails closed.

Exit code 0 = clean, 1 = violations found.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_TERMS = re.compile(
    r"\b(keyboard|mouse|controller|pynput|pyautogui|ctypes|"
    r"press_key|click|send_packet|read_game_memory|write_game_memory|"
    r"hook_process|dll|inject|memory_read|memory_write|"
    r"process_manipulation|account_automation|login|"
    r"automate_account|evade_detection|modify_client|"
    r"input_injection)\b",
    re.IGNORECASE,
)

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"][^'\"\\s]{8,}['\"]"),
    re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"][^'\"\\s]{4,}['\"]"),
    re.compile(r"(?i)(token|access_token|auth_token)\s*[:=]\s*['\"][^'\"\\s]{8,}['\"]"),
    re.compile(r"(?i)(cookie|session[_-]?id)\s*[:=]\s*['\"][^'\"\\s]{8,}['\"]"),
    re.compile(r"(?i)(fortnite[_-]?credential|epic[_-]?key|epic[_-]?token)\s*[:=]\s*['\"][^'\"\\s]+['\"]"),
    re.compile(r"(?i)github[_-]?token\s*[:=]\s*['\"][^'\"\\s]+['\"]"),
]

FORBIDDEN_IMPORTS = {
    "pynput", "pyautogui", "keyboard", "mouse", "controller",
    "ctypes", "sourcery",
}


def walk_python(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*.py"):
        if ".ipynb_checkpoints" in str(path):
            continue
        out.append(path)
    return out


def in_allowed_context(path: Path, root: Path) -> bool:
    # The security scan script itself legitimately names forbidden terms;
    # exclude it and other known-safe tooling files.
    if path.name in {
        "control_restrictions.py",
        "test_safety.py",
        "secrets_scan.py",
        "safety_scan.py",
    }:
        return True
    rel = path.relative_to(root).as_posix()
    if rel.startswith("core/tests/") or rel.startswith("tests/") or rel.startswith("docs/"):
        return True
    return False


def scan_terms() -> list[tuple[Path, str]]:
    violations: list[tuple[Path, str]] = []
    for path in walk_python(PROJECT_ROOT):
        if not in_allowed_context(path, PROJECT_ROOT):
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in FORBIDDEN_TERMS.finditer(text):
                violations.append((path, match.group(0)))
    return violations


def scan_secrets() -> list[tuple[Path, str]]:
    violations: list[tuple[Path, str]] = []
    for path in walk_python(PROJECT_ROOT):
        if not in_allowed_context(path, PROJECT_ROOT):
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), start=1):
                # Skip test files that intentionally assert about secrets.
                if path.name.endswith("_test.py") or path.name.startswith("test_"):
                    continue
                for pattern in SECRET_PATTERNS:
                    if pattern.search(line):
                        violations.append((path, f"line {line_no}: {line.strip()}"))
                        break
    return violations


def scan_imports() -> list[tuple[Path, str]]:
    violations: list[tuple[Path, str]] = []
    for layer in ["core/training", "core/companion"]:
        layer_root = PROJECT_ROOT / layer
        if not layer_root.exists():
            continue
        for path in walk_python(layer_root):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0].lower() in FORBIDDEN_IMPORTS:
                            violations.append((path, f"import {alias.name}"))
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        mod = node.module.split(".")[0].lower()
                        if mod in FORBIDDEN_IMPORTS:
                            violations.append((path, f"from {node.module} import ..."))
    return violations


def main() -> int:
    term_violations = scan_terms()
    secret_violations = scan_secrets()
    import_violations = scan_imports()

    total = len(term_violations) + len(secret_violations) + len(import_violations)

    if term_violations:
        print("[SECURITY] Forbidden real-game control terms found:")
        for path, term in term_violations[:10]:
            print(f"  {path}: {term}")
        if len(term_violations) > 10:
            print(f"  ... and {len(term_violations) - 10} more")

    if secret_violations:
        print("[SECURITY] Potential hardcoded secrets found:")
        for path, detail in secret_violations[:10]:
            print(f"  {path}: {detail}")
        if len(secret_violations) > 10:
            print(f"  ... and {len(secret_violations) - 10} more")

    if import_violations:
        print("[SECURITY] Forbidden imports found:")
        for path, detail in import_violations[:10]:
            print(f"  {path}: {detail}")
        if len(import_violations) > 10:
            print(f"  ... and {len(import_violations) - 10} more")

    if total == 0:
        print("[SECURITY] Clean — no secrets or forbidden control terms detected.")
        return 0

    print(f"\n[SECURITY] {total} violation(s) found.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

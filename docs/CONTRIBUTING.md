# Contributing

##  Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # pytest

# Or use the project installer
pip install -e ".[dev]"
```

##  Running tests

```bash
# Full suite
python3 -m pytest core/tests/ -q

# Safety tests (must pass)
python3 -m pytest core/tests/test_safety.py -q

# Performance regression tests
python3 -m pytest core/tests/test_performance.py -q

# Specific layer
python3 -m pytest core/tests/test_simulator.py -q
python3 -m pytest core/tests/test_strategy.py -q
python3 -m pytest core/tests/test_policy.py -q
```

##  Safety rules

**Do not introduce any real-game control mechanism.** Real Fortnite must remain
observation / analysis / coaching only.

If you add a new module, run the security scan:

```bash
python3 scripts/secrets_scan.py
```

If you add control-related code, the architecture tests will fail the build.
That is intentional.

###  Forbidden terms

The security scan flags source files that contain terms like `keyboard`,
`mouse`, `controller`, `press_key`, `click`, `send_packet`, `read_game_memory`,
`write_game_memory`, `hook_process`, `dll`, `inject`, `account_automation`,
`login`, and similar. These are only allowed in:

- `core/companion/control_restrictions.py` (the fail-closed interface)
- Test files (`core/tests/`)
- Documentation (`docs/`)

##  Adding a new strategy mode

1. Add the mode to `StrategyMode` enum in `core/types.py`.
2. Add an `ActionFocus` preset in `core/strategy.py` (`_STRATEGY_PRESETS`).
3. Add tests in `core/tests/test_strategy.py`.
4. Update documentation.

##  Adding a new skill

Skills are modular capabilities the policy can combine. Each skill should define:

- Input definition (what it reads from the observation)
- Output definition (what action/decision it produces)
- Evaluation metrics
- Training tasks
- Tests where appropriate

Place skills under `core/training/skills/` (to be created).

##  Adding a new scenario

1. Add a `MapInfo` to `_DEFAULT_MAPS` in `core/training/simulator/core.py`.
2. Add scenario kind handling in the simulator.
3. Add tests in `core/tests/test_simulator.py`.

##  Adding a new model provider

1. Implement `ModelProvider` (capabilities, infer, load_state).
2. Wire through environment variables (never hard-code credentials).
3. Test with the router in `core/tests/test_model_routing.py`.

##  Code style

- Keep the reactive policy fast and deterministic.
- Avoid heavy dependencies.
- Keep memory bounded.
- Prefer small-model-first decisions.
- Validate configuration defensively.
- Never log secrets.

##  CI

GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

1. Install dependencies
2. Security scan (`scripts/secrets_scan.py`)
3. Full test suite (`python3 -m pytest core/tests/ -q`)
4. Performance regression tests

The build fails if:

- Any test fails
- The security scan finds forbidden terms or secrets
- The safety boundary is violated

##  Pull requests

Keep commits logical. Do not commit secrets, credentials, large datasets,
temporary files, local caches, or unnecessary build artifacts.

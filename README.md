# Fortnite AI Research — General-Purpose Simulated Gameplay AI Platform

FAST, GENERAL-PURPOSE gameplay AI research platform capable of learning many
different gameplay tasks inside a permitted training/simulation environment.

##  Safety First

Real Fortnite is **observation / analysis / coaching only**.

This platform NEVER controls the real Fortnite client. No keyboard, mouse,
controller, input injection, DLL injection, memory reading/writing, packet
manipulation, anti-cheat bypass, account automation, botting, or ranked /
tournament automation.

Full agent control exists ONLY inside the separate custom training/simulation
environment.

See [SAFETY.md](docs/SAFETY.md) and the `core/companion/control_restrictions.py`
hard boundary.

##  Project structure

```
core/
  companion/     # Real-game observation-only coaching
  training/      # Simulator, skills, policies, memory
  models/        # Model provider abstractions and router
  ui/            # Lightweight Tkinter dashboard
  tests/         # Project tests
tests/           # Top-level test runner aliases
docs/            # Documentation
scripts/         # Dev/build helper scripts
.github/         # CI workflows
```

##  Quick start

```bash
# Install deps (see requirements.txt / pyproject.toml)
pip install -r requirements.txt

# Run the full test suite
python -m pytest core/tests/ -q

# Launch the lightweight UI (local dev only)
python -m core.ui.main
```

##  Modes

The AI supports many configurable pro-style strategy modes:

`BALANCED`, `AGGRESSIVE`, `DEFENSIVE`, `COMPETITIVE`, `AIM_FOCUSED`,
`MOBILITY_FOCUSED`, `BUILD_EDIT_FOCUSED`, `TEAM_SUPPORT`, `RESOURCE_FOCUSED`,
`ENDGAME`, `SURVIVAL`, `OBJECTIVE_FOCUSED`, `ADAPTIVE`, `CUSTOM`.

New environments, skills, and strategies are added through configuration or
plugins — not by rewriting the architecture.

##  Hardware assumptions

Local development targets severely constrained hardware (Windows 11 Pro, i3-12100,
4 GB RAM, Intel UHD 730). Heavy training runs on cloud / rented GPU
infrastructure. Local operation focuses on development, testing, lightweight
inference, UI, dataset management, and small experiments.

`LOW_RAM_MODE` targets <= 150 MB steady-state; normal mode targets <= 250 MB.

##  Docs

- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [SAFETY.md](docs/SAFETY.md)
- [TRAINING.md](docs/TRAINING.md)
- [MODELS.md](docs/MODELS.md)
- [PERFORMANCE.md](docs/PERFORMANCE.md)
- [DEVELOPMENT_STATUS.md](docs/DEVELOPMENT_STATUS.md)
- [CONTRIBUTING.md](docs/CONTRIBUTING.md)

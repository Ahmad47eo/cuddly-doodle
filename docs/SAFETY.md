# Safety

##  The hard rule

Real Fortnite MUST remain observation / analysis / coaching only.

This platform NEVER controls the real Fortnite client.

##  What is forbidden

The following are NOT implemented anywhere in this codebase:

- Keyboard automation
- Mouse automation
- Controller automation
- OS-level input simulation
- Input injection
- DLL injection
- Process manipulation
- Game memory reading/writing
- Packet manipulation
- Anti-cheat bypass
- Exploits
- Client modification
- Account automation
- Botting
- Ranked automation
- Tournament automation
- Detection evasion
- Anti-cheat evasion
- Credential collection
- Login automation

##  How the boundary is enforced

### 1. Fail-closed control interface

`core/companion/control_restrictions.py` defines
`RealFortniteControlInterface`. Every method raises
`SafetyViolationError` immediately. There is no real implementation anywhere
reachable.

### 2. Architectural separation

- `core/training/` must not receive live Fortnite control input.
- `core/companion/` must not import real control implementations.
- The build fails if forbidden control APIs are introduced.

### 3. Architecture tests

`core/tests/test_safety.py` verifies:

- Every `RealFortniteControlInterface` operation raises `SafetyViolationError`.
- Forbidden control categories are enumerated.
- No forbidden imports in training/companion layers.
- No forbidden real-game control terms in non-test, non-docs source.

### 4. Static security scan

`scripts/secrets_scan.py` scans for:

- Forbidden real-game control terms in source.
- Hardcoded secrets (API keys, passwords, tokens, cookies, credentials).
- Forbidden control imports in training/companion layers.

##  Where agent control IS permitted

Full agent control exists ONLY inside the dedicated custom training/simulation
environment (`core/training/simulator/`). The simulator is a lightweight,
deterministic, Gymnasium-style environment used for research. It does NOT connect
to any real Fortnite client.

##  Coaching scope

The real-game companion (`core/companion/analysis.py`) may analyze permitted
observation inputs:

- User-provided screenshots
- User-provided videos
- Pre-recorded gameplay
- Structured telemetry the user explicitly supplied
- Simulator state (when used inside the simulator)

It may provide:

- Strategy suggestions
- Positioning advice
- Decision analysis
- Aim analysis
- Rotation analysis
- Post-game review
- Performance statistics
- Training recommendations

It must NOT control Fortnite.

##  Final safety scan

Before any release, run:

```bash
python3 scripts/secrets_scan.py
python3 -m pytest core/tests/test_safety.py -q
python3 -m pytest core/tests/ -q
```

All must pass with zero forbidden terms, zero secrets, and zero safety test
failures.

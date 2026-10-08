# Models

##  Provider-independent architecture

The AI is NOT hard-coded to one model provider. The system defines:

- `ModelProvider` — abstract contract for capability discovery, inference, and
  load state.
- `Capability` — what a model can do, with strengths, weaknesses, task hints,
  cost, and latency estimates.
- `ScoringContext` — task hint, latency budget, cost limit, small-first
  preference.
- `ModelRouter` — selects the best provider for a task.
- `LocalModelProvider` / `CloudModelProvider` / `FallbackProvider` —
  concrete provider wrappers.

##  Provider interface

```python
class ModelProvider:
    def capabilities(self, context: ScoringContext) -> list[Capability]: ...
    def infer(self, context, task_hint, payload) -> InferenceResult: ...
    def load_state(self) -> LoadState: ...
```

##  Model router

The router selects a provider based on:

- Task complexity / task hint
- Latency budget (soft + hard)
- Cost
- Hardware constraints (including LOW_RAM_MODE)
- Model availability / load state
- Context requirements
- Training vs inference mode

**Small-model-first**: the router prefers the smallest capable provider and
escalates only when necessary. If cloud inference is unavailable, it falls back
to a local method.

##  Routing decisions

Reasons reported by the router:

- `PERFECT_MATCH` — provider best matches the task
- `SMALL_MODEL_FIRST` — small provider is sufficient
- `BEST_AVAILABLE` — best available provider selected
- `FALLBACK` — primary unavailable, fell back to another
- `TIMEOUT` — no provider can meet the latency budget

##  Concrete providers (stub implementations)

For development and testing, the codebase includes:

- `DummyLocalProvider` — fast, small, CPU-only stub for local inference.
- `DummyCloudProvider` — slower, more capable stub for cloud inference.
- `FallbackProvider` — wraps primary + fallback, degrades gracefully.

These are placeholders to be replaced with real provider integrations (for
example ONNX/TensorFlow/PyTorch local inference, or an external cloud inference
endpoint). Real providers are wired through environment variables (never
hard-coded).

##  Latency budgets

The router respects `LatencyBudget(soft_ns, hard_ns)`. If the best provider
exceeds the hard budget, the router tries to find a smaller/faster provider.
If none exists, it reports `TIMEOUT`.

##  Cost guard

The router respects `max_cost_usd_per_1k_tokens`. Providers with cost above the
limit are penalized in scoring.

##  LOW_RAM_MODE interaction

In LOW_RAM_MODE, the router strongly prefers cheap/fast local providers for
simple tasks (steer, shoot, move, reload, use_item) over expensive/large ones.

See [PERFORMANCE.md](PERFORMANCE.md) for measured latencies.

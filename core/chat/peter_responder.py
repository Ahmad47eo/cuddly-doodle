"""Peter's lightweight text responder for the local phone-chat UI.

This is intentionally offline and dependency-free. It gives Peter a real,
usable chat personality while also exposing the existing local inference
provider's action classification. It never controls the OS or Fortnite.
"""

from __future__ import annotations

from typing import Any

from core.models.provider import LatencyBudget, ScoringContext
from core.models.providers.local import DummyLocalProvider


_PROVIDER = DummyLocalProvider(name="peter_local")


def _context(message: str) -> ScoringContext:
    hint = ""
    lower = message.lower()
    if any(x in lower for x in ("move", "walk", "run", "steer")):
        hint = "move"
    elif any(x in lower for x in ("shoot", "fire", "aim")):
        hint = "shoot"
    elif "reload" in lower:
        hint = "reload"
    elif any(x in lower for x in ("heal", "medkit", "item")):
        hint = "use_item"
    return ScoringContext(
        task_hint=hint,
        latency_budget=LatencyBudget(soft_ns=50_000, hard_ns=200_000),
        prefer_small_first=True,
    )


def respond(message: str) -> str:
    """Return Peter's reply to one phone-chat message."""
    text = " ".join(str(message).strip().split())
    if not text:
        return "I'm here. Send me a message."

    lower = text.lower()
    if lower in {"hi", "hello", "hey", "yo"}:
        return "Hey 😎 I'm Peter. I'm ready."
    if "who are you" in lower or "what are you" in lower:
        return "I'm Peter, your local AI teammate prototype. Chat is running on your PC."
    if "help" in lower:
        return "Yep. You can chat with me here, ask for strategy, or test simulator actions."
    if any(x in lower for x in ("bye", "goodbye", "gn")):
        return "See you later 👋"

    ctx = _context(text)
    result = _PROVIDER.infer(ctx, ctx.task_hint, {"message": text})

    if result.intended_action == "move":
        return "Got it. For the simulator, I'd prioritize movement and positioning. I can turn that into virtual input there."
    if result.intended_action == "fire":
        return "I'd focus on aim and target selection first. In the simulator, I can represent that as a virtual fire action."
    if result.intended_action == "reload":
        return "Reloading makes sense when the current engagement is safe. I'll keep it simulator-only."
    if result.intended_action == "use_item":
        return "I'd use the item when it's safe to do so. I can represent that in the simulator."
    return "I got you. I'm listening. Tell me what you want to plan or test."

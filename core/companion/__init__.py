"""Real-game observation-only companion (analysis / coaching)."""

from core.companion.analysis import GameplayCoach, Insight, PostGameReview
from core.companion.control_restrictions import (
    RealFortniteControlInterface,
    assert_no_real_control_import,
    list_forbidden_categories,
)

__all__ = [
    "GameplayCoach",
    "Insight",
    "PostGameReview",
    "RealFortniteControlInterface",
    "assert_no_real_control_import",
    "list_forbidden_categories",
]

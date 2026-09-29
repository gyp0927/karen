"""models.dev reasoning options -> Pi thinking-level maps.

Port of pi-ai's `scripts/models-dev-reasoning-options.ts`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

#: Pi's selectable thinking levels, in ascending order.
THINKING_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")

#: Either `{"type": "toggle"}`, `{"type": "effort", "values": [...]}` or
#: `{"type": "budget_tokens", "min": ..., "max": ...}`.
ReasoningOption = Mapping[str, Any]
ThinkingLevelMap = Dict[str, Optional[str]]


def get_effort_thinking_level_map(options: Optional[Sequence[ReasoningOption]]) -> Optional[ThinkingLevelMap]:
    """Convert models.dev verified effort values into Pi's selectable levels.

    Values without a Pi equivalent (`default` and JSON `null`) are intentionally
    omitted, so they map to `None` (unsupported) in the result.
    """
    effort_values: List[Any] = []
    for option in options or ():
        if option.get("type") == "effort":
            effort_values.extend(option.get("values") or ())

    if not effort_values:
        return None

    supported = set(effort_values)
    if not any(level in supported for level in THINKING_LEVELS) and "none" not in supported:
        return None

    result: ThinkingLevelMap = {"off": "none" if "none" in supported else None}
    for level in THINKING_LEVELS:
        result[level] = level if level in supported else None
    return result


def has_toggle(options: Optional[Iterable[ReasoningOption]]) -> bool:
    """Whether models.dev advertises an on/off toggle for this model."""
    return any(option.get("type") == "toggle" for option in options or ())


def has_effort(options: Optional[Iterable[ReasoningOption]]) -> bool:
    """Whether models.dev advertises discrete effort values for this model."""
    return any(option.get("type") == "effort" for option in options or ())


__all__ = ["THINKING_LEVELS", "ReasoningOption", "ThinkingLevelMap", "get_effort_thinking_level_map", "has_effort", "has_toggle"]

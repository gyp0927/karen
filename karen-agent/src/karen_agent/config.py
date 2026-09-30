"""Harness configuration defaults and validation (pi's `harness/config.ts`).

Note: karen-ai adapters retry transient HTTP errors via ``max_retries`` and the
loop does not consume ``RetryPolicy`` (pi uses it in `retryAssistantCall` and
the durable runtime, neither of which karen ports). The policy type and its
validation are ported for harness/app-level configuration parity.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

from karen_ai.types import KarenBase

from .compaction import CompactionSettings

__all__ = [
    "RetryPolicy",
    "DEFAULT_MAX_AGENT_RETRY_DELAY_MS",
    "DEFAULT_RETRY_POLICY",
    "validate_tool_names",
    "validate_retry_policy",
    "validate_compaction_settings",
]

#: pi-ai's retry.ts default cap for provider-requested retry delays.
DEFAULT_MAX_AGENT_RETRY_DELAY_MS = 60_000

_MAX_SAFE_INTEGER = 2**53 - 1


class RetryPolicy(KarenBase):
    """Assistant-call retry policy (pi-ai's `RetryPolicy`)."""

    enabled: bool
    max_retries: int
    base_delay_ms: int
    max_agent_delay_ms: Optional[int] = None


DEFAULT_RETRY_POLICY = RetryPolicy(
    enabled=True,
    max_retries=3,
    base_delay_ms=1_000,
    max_agent_delay_ms=DEFAULT_MAX_AGENT_RETRY_DELAY_MS,
)


def _is_safe_integer(value: Any) -> bool:
    """JS `Number.isSafeInteger`: booleans excluded; integral floats accepted."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER
    if isinstance(value, float):
        return value.is_integer() and -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER
    return False


def validate_tool_names(tools: List[Any]) -> None:
    """Reject duplicate tool names (pi throws `TypeError`)."""
    names = set()
    for tool in tools:
        name = tool.name
        if name in names:
            raise TypeError(f"Duplicate tool name: {json.dumps(name, ensure_ascii=False)}")
        names.add(name)


def validate_retry_policy(policy: RetryPolicy) -> None:
    """Reject non-safe-integer or negative retry values (pi throws `RangeError`)."""
    if (
        not _is_safe_integer(policy.max_retries)
        or policy.max_retries < 0
        or policy.max_retries == _MAX_SAFE_INTEGER
        or not _is_safe_integer(policy.base_delay_ms)
        or policy.base_delay_ms < 0
        or (
            policy.max_agent_delay_ms is not None
            and (not _is_safe_integer(policy.max_agent_delay_ms) or policy.max_agent_delay_ms < 0)
        )
    ):
        raise ValueError("Retry policy values must be finite non-negative safe integers")


def validate_compaction_settings(settings: CompactionSettings) -> None:
    """Reject non-safe-integer or negative token counts (pi throws `RangeError`)."""
    if (
        not _is_safe_integer(settings.reserve_tokens)
        or settings.reserve_tokens < 0
        or not _is_safe_integer(settings.keep_recent_tokens)
        or settings.keep_recent_tokens < 0
    ):
        raise ValueError("Compaction token counts must be finite non-negative safe integers")

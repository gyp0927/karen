"""Harness configuration defaults and validation (pi's `harness/config.ts`).

``RetryPolicy`` and ``DEFAULT_MAX_AGENT_RETRY_DELAY_MS`` are re-exported from
karen-ai (pi-ai's `utils/retry.ts`), where the classifier and the policy-driven
retry loop live; the harness app-level default (``DEFAULT_RETRY_POLICY``)
matches pi's `DEFAULT_RETRY_POLICY`. The summary calls in
``karen_agent.compaction`` consume the policy through
``karen_ai.utils.retry_assistant_call``; the agent loop itself leaves retries
to the application layer (pi coding-agent's `AgentSession`).
"""

from __future__ import annotations

import json
from typing import Any, List

from karen_ai.utils.retry import DEFAULT_MAX_AGENT_RETRY_DELAY_MS, RetryPolicy

from .compaction import CompactionSettings

__all__ = [
    "RetryPolicy",
    "DEFAULT_MAX_AGENT_RETRY_DELAY_MS",
    "DEFAULT_RETRY_POLICY",
    "validate_tool_names",
    "validate_retry_policy",
    "validate_compaction_settings",
]

_MAX_SAFE_INTEGER = 2**53 - 1


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

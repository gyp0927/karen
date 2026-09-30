"""Tests for `karen_agent.config` (pi's `harness/config.ts`)."""

import pytest

from karen_agent import (
    DEFAULT_MAX_AGENT_RETRY_DELAY_MS,
    DEFAULT_RETRY_POLICY,
    AgentTool,
    RetryPolicy,
    validate_compaction_settings,
    validate_retry_policy,
    validate_tool_names,
)
from karen_agent.compaction import CompactionSettings

_MAX_SAFE_INTEGER = 2**53 - 1


def _tool(name):
    async def execute(tool_call_id, params, signal, on_update):  # pragma: no cover - never executed
        raise AssertionError("not executed")

    return AgentTool(name=name, description="", label=name, parameters={"type": "object"}, execute=execute)


# ---------------------------------------------------------------------------
# validate_tool_names
# ---------------------------------------------------------------------------


def test_validate_tool_names_accepts_unique_names():
    validate_tool_names([_tool("read"), _tool("write")])


def test_validate_tool_names_rejects_duplicates():
    with pytest.raises(TypeError, match='Duplicate tool name: "read"'):
        validate_tool_names([_tool("read"), _tool("read")])


def test_validate_tool_names_quotes_non_ascii_like_js():
    with pytest.raises(TypeError, match='Duplicate tool name: "读"'):
        validate_tool_names([_tool("读"), _tool("读")])


# ---------------------------------------------------------------------------
# validate_retry_policy
# ---------------------------------------------------------------------------


def test_default_retry_policy_values():
    assert DEFAULT_RETRY_POLICY.enabled is True
    assert DEFAULT_RETRY_POLICY.max_retries == 3
    assert DEFAULT_RETRY_POLICY.base_delay_ms == 1_000
    assert DEFAULT_RETRY_POLICY.max_agent_delay_ms == DEFAULT_MAX_AGENT_RETRY_DELAY_MS == 60_000


def test_validate_retry_policy_accepts_valid_shapes():
    validate_retry_policy(DEFAULT_RETRY_POLICY)
    validate_retry_policy(RetryPolicy(enabled=False, max_retries=0, base_delay_ms=0, max_agent_delay_ms=None))
    # Integral floats are safe integers in JS too.
    validate_retry_policy(RetryPolicy(enabled=True, max_retries=3.0, base_delay_ms=0.0))


@pytest.mark.parametrize(
    "policy",
    [
        RetryPolicy(enabled=True, max_retries=-1, base_delay_ms=0),
        RetryPolicy(enabled=True, max_retries=_MAX_SAFE_INTEGER, base_delay_ms=0),
        RetryPolicy(enabled=True, max_retries=2**53, base_delay_ms=0),
        RetryPolicy(enabled=True, max_retries=1, base_delay_ms=-1),
        RetryPolicy(enabled=True, max_retries=1, base_delay_ms=1, max_agent_delay_ms=-5),
        # model_construct bypasses pydantic's int coercion so the validator sees raw floats.
        RetryPolicy.model_construct(enabled=True, max_retries=1.5, base_delay_ms=0),
        RetryPolicy.model_construct(enabled=True, max_retries=1, base_delay_ms=1, max_agent_delay_ms=1.25),
    ],
)
def test_validate_retry_policy_rejects_invalid_values(policy):
    with pytest.raises(ValueError, match="Retry policy values must be finite non-negative safe integers"):
        validate_retry_policy(policy)


# ---------------------------------------------------------------------------
# validate_compaction_settings
# ---------------------------------------------------------------------------


def test_validate_compaction_settings_accepts_defaults():
    validate_compaction_settings(CompactionSettings())


@pytest.mark.parametrize(
    "settings",
    [
        CompactionSettings(enabled=True, reserve_tokens=-1, keep_recent_tokens=0),
        CompactionSettings(enabled=True, reserve_tokens=0, keep_recent_tokens=-1),
        CompactionSettings(enabled=True, reserve_tokens=2**53, keep_recent_tokens=0),
        # model_construct bypasses pydantic's int coercion so the validator sees raw floats.
        CompactionSettings.model_construct(enabled=True, reserve_tokens=1.5, keep_recent_tokens=0),
    ],
)
def test_validate_compaction_settings_rejects_invalid_values(settings):
    with pytest.raises(ValueError, match="Compaction token counts must be finite non-negative safe integers"):
        validate_compaction_settings(settings)

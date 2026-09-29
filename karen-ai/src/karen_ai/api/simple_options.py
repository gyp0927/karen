"""Shared stream-simple option plumbing, mirroring pi-ai's api/simple-options.ts."""

from __future__ import annotations

from typing import Optional

from ..types import (
    Model,
    SimpleStreamOptions,
    StreamOptions,
    ThinkingBudgets,
    ThinkingLevel,
    TranscriptContext,
)
from ..utils.estimate import estimate_context_tokens

CONTEXT_SAFETY_TOKENS = 4096
MIN_MAX_TOKENS = 1


def clamp_max_tokens_to_context(model: Model, context: TranscriptContext, max_tokens: int) -> int:
    if model.context_window <= 0:
        return max(MIN_MAX_TOKENS, max_tokens)
    available = model.context_window - estimate_context_tokens(context) - CONTEXT_SAFETY_TOKENS
    return min(max_tokens, max(MIN_MAX_TOKENS, available))


def build_base_options(
    model: Model,
    context: TranscriptContext,
    options: Optional[SimpleStreamOptions],
    api_key: Optional[str] = None,
) -> StreamOptions:
    return StreamOptions(
        temperature=options.temperature if options else None,
        sampling_params=options.sampling_params if options else None,
        max_tokens=clamp_max_tokens_to_context(
            model, context, (options.max_tokens if options and options.max_tokens else model.max_tokens)
        ),
        signal=options.signal if options else None,
        telemetry_context=options.telemetry_context if options else None,
        api_key=api_key or (options.api_key if options else None),
        transport=options.transport if options else None,
        cache_retention=options.cache_retention if options else None,
        session_id=options.session_id if options else None,
        headers=options.headers if options else None,
        on_payload=options.on_payload if options else None,
        on_response=options.on_response if options else None,
        on_provider_stream_event=options.on_provider_stream_event if options else None,
        timeout_ms=options.timeout_ms if options else None,
        websocket_connect_timeout_ms=options.websocket_connect_timeout_ms if options else None,
        max_retries=options.max_retries if options else None,
        max_retry_delay_ms=options.max_retry_delay_ms if options else None,
        metadata=options.metadata if options else None,
        env=options.env if options else None,
    )


#: Tokens always left for the answer when a thinking budget shares the response ceiling.
MIN_ANSWER_TOKENS = 1024

DEFAULT_THINKING_BUDGETS = ThinkingBudgets(minimal=1024, low=2048, medium=8192, high=16384)


def clamp_reasoning(effort: Optional[ThinkingLevel]) -> Optional[ThinkingLevel]:
    return "high" if effort in ("xhigh", "max") else effort


def thinking_budget_for_level(reasoning_level: ThinkingLevel, custom_budgets: Optional[ThinkingBudgets] = None) -> int:
    level = clamp_reasoning(reasoning_level) or "high"
    custom = (custom_budgets.model_dump() if custom_budgets else {})
    default = DEFAULT_THINKING_BUDGETS.model_dump()
    budget = custom.get(level) or default.get(level)
    return int(budget)


def clamp_thinking_budget_to_answer_room(thinking_budget: int, ceiling: int) -> int:
    """Cap a thinking budget so at least MIN_ANSWER_TOKENS remain under a shared response ceiling."""
    return min(thinking_budget, max(0, ceiling - MIN_ANSWER_TOKENS))


def adjust_max_tokens_for_thinking(
    base_max_tokens: Optional[int],
    model_max_tokens: int,
    reasoning_level: ThinkingLevel,
    custom_budgets: Optional[ThinkingBudgets] = None,
) -> tuple[int, int]:
    """Returns (max_tokens, thinking_budget)."""
    thinking_budget = thinking_budget_for_level(reasoning_level, custom_budgets)
    max_tokens = model_max_tokens if base_max_tokens is None else min(base_max_tokens + thinking_budget, model_max_tokens)

    if max_tokens <= thinking_budget:
        thinking_budget = clamp_thinking_budget_to_answer_room(thinking_budget, max_tokens)

    return max_tokens, thinking_budget

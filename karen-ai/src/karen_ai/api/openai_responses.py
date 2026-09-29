"""OpenAI Responses API adapter, mirroring api/openai-responses.ts.

Used by OpenAI's modern models (GPT-5.x, o-series) and the Azure / Codex
variants, which share the conversion and stream machinery in
openai_responses_shared.py.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterable, Dict, List, Optional

import httpx

from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost, clamp_thinking_level
from ..transcript import get_declared_tools, resolve_transcript, resolve_transcript_tools
from ..types import (
    AssistantMessage,
    CacheRetention,
    DoneEvent,
    ErrorEvent,
    Model,
    OpenAIResponsesCompat,
    ProviderEnv,
    ProviderHeaders,
    ProviderResponse,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TranscriptContext,
    Usage,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.provider_env import get_provider_env_value
from ..utils.retry import ProviderHttpError, retry_provider_request
from ..utils.sse import iterate_sse_messages
from .constrained_sampling import create_grammar_tool_input_properties
from .github_copilot_headers import build_copilot_dynamic_headers, has_copilot_vision_input
from .openai_prompt_cache import clamp_openai_prompt_cache_key
from .openai_responses_shared import (
    ConvertResponsesMessagesOptions,
    ConvertResponsesToolsOptions,
    OpenAIResponsesStreamOptions,
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from .request_options import coerce_options
from .simple_options import build_base_options

KAREN_USER_AGENT = "karen-ai/0.1.0"

OPENAI_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode"}
# OpenAI Responses rejects max_output_tokens below 16:
# https://github.com/earendil-works/pi/issues/6265
OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16


class OpenAIResponsesOptions(StreamOptions):
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None  # "auto" | "detailed" | "concise"
    service_tier: Optional[str] = None
    tool_choice: Optional[Any] = None


# ---------------------------------------------------------------------------
# Compat resolution
# ---------------------------------------------------------------------------


def _detect_session_affinity_format(model: Model) -> str:
    return "openrouter" if model.provider == "openrouter" or "openrouter.ai" in model.base_url else "openai"


class _ResolvedCompat:
    """OpenAIResponsesCompat with all defaults applied."""

    _DEFAULTS: Dict[str, Any] = {}

    def __init__(self, model: Model):
        override = model.compat
        if override is not None and not isinstance(override, OpenAIResponsesCompat):
            override = OpenAIResponsesCompat.model_validate(override.model_dump())
        defaults = {
            "supports_developer_role": True,
            "supports_mid_convo_system_messages": False,
            "session_affinity_format": _detect_session_affinity_format(model),
            "supports_long_cache_retention": True,
            "supports_strict_mode": False,
            "supports_openai_grammar_tools": False,
            "supports_additional_tools": False,
            "supports_tool_search": False,
            "supports_explicit_prompt_cache_mode": False,
            "supports_max_output_tokens": True,
        }
        for field_name, default in defaults.items():
            value = getattr(override, field_name, None) if override else None
            setattr(self, field_name, value if value is not None else default)


def _resolve_cache_retention(cache_retention: Optional[CacheRetention], env: Optional[ProviderEnv]) -> CacheRetention:
    """Resolve cache retention preference.

    Defaults to "short" and uses PI_CACHE_RETENTION for backward compatibility.
    """
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


def _get_prompt_cache_retention(compat: _ResolvedCompat, cache_retention: CacheRetention) -> Optional[str]:
    if (
        cache_retention == "long"
        and compat.supports_long_cache_retention
        and not compat.supports_explicit_prompt_cache_mode
    ):
        return "24h"
    return None


def _get_prompt_cache_options(compat: _ResolvedCompat, cache_retention: CacheRetention) -> Optional[Dict[str, Any]]:
    if not compat.supports_explicit_prompt_cache_mode:
        return None
    if cache_retention == "none":
        return {"mode": "explicit"}
    if cache_retention == "long" and compat.supports_long_cache_retention:
        return {"ttl": "30m"}
    return None


# ---------------------------------------------------------------------------
# Auth / headers
# ---------------------------------------------------------------------------


def _has_header(headers: Optional[ProviderHeaders], name: str) -> bool:
    if not headers:
        return False
    expected = name.lower()
    return any(key.lower() == expected and value and value.strip() for key, value in headers.items())


def _get_client_api_key(provider: str, api_key: Optional[str], headers: Optional[ProviderHeaders]) -> str:
    if api_key:
        return api_key
    if _has_header(headers, "authorization") or _has_header(headers, "cf-aig-authorization"):
        return "unused"
    raise ValueError(f"No API key for provider: {provider}")


def _build_headers(
    model: Model,
    context: TranscriptContext,
    api_key: str,
    options_headers: Optional[ProviderHeaders],
    session_id: Optional[str],
    compat: _ResolvedCompat,
) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "User-Agent": KAREN_USER_AGENT,
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }
    for name, value in (model.headers or {}).items():
        if value is not None:
            headers[name] = value

    if model.provider == "github-copilot":
        has_images = has_copilot_vision_input(context.messages)
        headers.update(build_copilot_dynamic_headers(context.messages, has_images))

    if session_id:
        if compat.session_affinity_format == "openrouter":
            headers["x-session-id"] = session_id
        else:
            if compat.session_affinity_format == "openai":
                headers["session_id"] = session_id
            headers["x-client-request-id"] = session_id

    # Merge options headers last so they can override defaults; None suppresses.
    for name, value in (options_headers or {}).items():
        if value is None:
            for existing in list(headers.keys()):
                if existing.lower() == name.lower():
                    del headers[existing]
        else:
            headers[name] = value
    return headers


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def _build_params(
    model: Model,
    context: TranscriptContext,
    options: Optional[OpenAIResponsesOptions],
    compat: _ResolvedCompat,
    grammar_tool_input_properties: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            get_declared_tools(context.messages), compat.supports_openai_grammar_tools
        )
    transcript_tools = resolve_transcript_tools(
        context.messages,
        compat.supports_additional_tools or compat.supports_tool_search,
    )
    messages = convert_responses_messages(
        model,
        context,
        OPENAI_TOOL_CALL_PROVIDERS,
        ConvertResponsesMessagesOptions(
            grammar_tool_input_properties=grammar_tool_input_properties,
            supports_mid_convo_system_messages=compat.supports_mid_convo_system_messages,
            supports_additional_tools=compat.supports_additional_tools,
            supports_tool_search=compat.supports_tool_search,
            tool_options=ConvertResponsesToolsOptions(
                supports_strict_mode=compat.supports_strict_mode,
                supports_openai_grammar_tools=compat.supports_openai_grammar_tools,
            ),
        ),
    )

    cache_retention = _resolve_cache_retention(
        options.cache_retention if options else None, options.env if options else None
    )
    params: Dict[str, Any] = {"model": model.id, "input": messages, "stream": True, "store": False}
    if cache_retention != "none":
        cache_key = clamp_openai_prompt_cache_key(options.session_id if options else None)
        if cache_key is not None:
            params["prompt_cache_key"] = cache_key
    cache_retention_value = _get_prompt_cache_retention(compat, cache_retention)
    if cache_retention_value is not None:
        params["prompt_cache_retention"] = cache_retention_value
    cache_options = _get_prompt_cache_options(compat, cache_retention)
    if cache_options is not None:
        params["prompt_cache_options"] = cache_options

    if options and options.max_tokens and compat.supports_max_output_tokens:
        params["max_output_tokens"] = max(options.max_tokens, OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)

    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    if options and options.service_tier is not None:
        params["service_tier"] = options.service_tier

    if transcript_tools.request_tools:
        params["tools"] = convert_responses_tools(
            transcript_tools.request_tools,
            ConvertResponsesToolsOptions(
                supports_strict_mode=compat.supports_strict_mode,
                supports_openai_grammar_tools=compat.supports_openai_grammar_tools,
            ),
        )

    if options and options.tool_choice is not None:
        params["tool_choice"] = options.tool_choice

    if model.reasoning:
        reasoning_effort = options.reasoning_effort if options else None
        reasoning_summary = options.reasoning_summary if options else None
        level_map = model.thinking_level_map or {}
        if reasoning_effort or reasoning_summary:
            effort = (level_map.get(reasoning_effort) or reasoning_effort) if reasoning_effort else "medium"
            params["reasoning"] = {"effort": effort, "summary": reasoning_summary or "auto"}
            params["include"] = ["reasoning.encrypted_content"]
        elif model.provider != "github-copilot" and not ("off" in level_map and level_map["off"] is None):
            # off: None explicitly opts out of sending the reasoning parameter.
            params["reasoning"] = {"effort": level_map.get("off") or "none"}
        if model.provider == "xai":
            params["include"] = ["reasoning.encrypted_content"]

    # Last so custom keys override the named request fields. Per-request keys
    # override model defaults.
    if model.sampling_params:
        params.update(model.sampling_params)
    if options and options.sampling_params:
        params.update(options.sampling_params)

    return params


# ---------------------------------------------------------------------------
# Service tier pricing
# ---------------------------------------------------------------------------


def _get_service_tier_cost_multiplier(model: Model, service_tier: Optional[str]) -> float:
    if service_tier == "flex":
        return 0.5
    if service_tier in ("priority", "fast"):
        return 2.5 if model.id == "gpt-5.5" else 2
    return 1


def _apply_service_tier_pricing(usage: Usage, service_tier: Optional[str], model: Model) -> None:
    multiplier = _get_service_tier_cost_multiplier(model, service_tier)
    if multiplier == 1:
        return
    usage.cost.input *= multiplier
    usage.cost.output *= multiplier
    usage.cost.cache_read *= multiplier
    usage.cost.cache_write *= multiplier
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write


# ---------------------------------------------------------------------------
# SSE iteration
# ---------------------------------------------------------------------------


async def _iterate_responses_events(response: httpx.Response, signal=None) -> AsyncIterable[Dict[str, Any]]:
    async for sse in iterate_sse_messages(response.aiter_text(), signal):
        data = sse.data.strip()
        if not data:
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


# ---------------------------------------------------------------------------
# stream / stream_simple
# ---------------------------------------------------------------------------


def stream(
    model: Model,
    context: TranscriptContext,
    options: Optional[OpenAIResponsesOptions] = None,
) -> AssistantMessageEventStream:
    options = coerce_options(options, OpenAIResponsesOptions)
    event_stream = AssistantMessageEventStream()
    compat = _ResolvedCompat(model)
    normalized_context = resolve_transcript(context, compat.supports_mid_convo_system_messages)

    async def run() -> None:
        output = AssistantMessage(
            role="assistant",
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="pending",
            timestamp=int(time.time() * 1000),
        )

        try:
            api_key = _get_client_api_key(
                model.provider, options.api_key if options else None, options.headers if options else None
            )
            cache_retention = _resolve_cache_retention(
                options.cache_retention if options else None, options.env if options else None
            )
            cache_session_id = options.session_id if cache_retention != "none" and options else None
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                get_declared_tools(normalized_context.messages), compat.supports_openai_grammar_tools
            )

            params = _build_params(model, normalized_context, options, compat, grammar_tool_input_properties)
            if options and options.on_payload:
                next_params = options.on_payload(params, model)
                if asyncio.iscoroutine(next_params):
                    next_params = await next_params
                if next_params is not None:
                    params = next_params

            headers = _build_headers(
                model, normalized_context, api_key, options.headers if options else None, cache_session_id, compat
            )

            timeout = httpx.Timeout((options.timeout_ms or 600_000) / 1000, connect=60.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                url = model.base_url.rstrip("/") + "/responses"

                async def do_request() -> httpx.Response:
                    request = client.build_request("POST", url, json=params, headers=headers)
                    response = await client.send(request, stream=True)
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        await response.aclose()
                        raise ProviderHttpError(
                            f"OpenAI Responses API error {response.status_code}: {body}",
                            status=response.status_code,
                            headers=response.headers,
                        )
                    return response

                response = await retry_provider_request(
                    do_request,
                    max_retries=options.max_retries if options else None,
                    max_retry_delay_ms=options.max_retry_delay_ms if options else None,
                    signal=options.signal if options else None,
                )

                try:
                    if options and options.on_response:
                        maybe = options.on_response(
                            ProviderResponse(status=response.status_code, headers=dict(response.headers)), model
                        )
                        if asyncio.iscoroutine(maybe):
                            await maybe

                    event_stream.push(StartEvent(partial=output))

                    await process_responses_stream(
                        _iterate_responses_events(response, options.signal if options else None),
                        output,
                        event_stream,
                        model,
                        OpenAIResponsesStreamOptions(
                            on_provider_stream_event=options.on_provider_stream_event if options else None,
                            service_tier=options.service_tier if options else None,
                            grammar_tool_input_properties=grammar_tool_input_properties,
                            apply_service_tier_pricing=lambda usage, tier: _apply_service_tier_pricing(
                                usage, tier, model
                            ),
                        ),
                    )

                    if options and options.signal and options.signal.aborted:
                        raise RuntimeError("Request was aborted")
                    if output.stop_reason == "pending":
                        raise RuntimeError("OpenAI Responses stream ended without a stop reason")
                    if output.stop_reason in ("aborted", "error"):
                        raise RuntimeError(output.error_message or "An unknown error occurred")

                    event_stream.push(DoneEvent(reason=output.stop_reason, message=output))  # type: ignore[arg-type]
                    event_stream.end()
                finally:
                    await response.aclose()

        except Exception as error:
            output.stop_reason = "aborted" if (options and options.signal and options.signal.aborted) else "error"
            prefix = "OpenAI" if model.provider == "openai" else model.provider
            output.error_message = format_provider_error(normalize_provider_error(error), f"{prefix} API error")
            event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))  # type: ignore[arg-type]
            event_stream.end()

    asyncio.get_running_loop().create_task(run())
    return event_stream


def stream_simple(
    model: Model,
    context: TranscriptContext,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    _get_client_api_key(model.provider, options.api_key if options else None, options.headers if options else None)

    base = build_base_options(model, context, options, options.api_key if options else None)
    clamped_reasoning = clamp_thinking_level(model, options.reasoning) if options and options.reasoning else None
    reasoning_effort = None if clamped_reasoning == "off" else clamped_reasoning

    merged = OpenAIResponsesOptions(**base.model_dump(exclude_none=True))
    merged.tool_choice = options.tool_choice if options else None
    merged.reasoning_effort = reasoning_effort
    return stream(model, context, merged)

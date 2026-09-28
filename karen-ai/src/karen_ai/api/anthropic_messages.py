"""Anthropic Messages API adapter, mirroring pi-ai's api/anthropic-messages.ts.

Streams via plain httpx + SSE (pi-ai uses the official SDK only as an HTTP
layer and iterates the raw event stream itself; we do the same directly).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union

import httpx

from ..abort import AbortSignal
from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost
from ..transcript import (
    get_current_tools,
    get_declared_tools,
    get_initial_system_message,
    has_tool_redefinitions,
    resolve_transcript,
)
from ..types import (
    AnthropicMessagesCompat,
    AssistantMessage,
    CacheRetention,
    ContentBlock,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Model,
    ProviderEnv,
    ProviderHeaders,
    ProviderResponse,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    TranscriptContext,
    Usage,
)
from ..utils.json_parse import parse_json_with_repair, parse_streaming_json
from ..utils.provider_env import get_provider_env_value
from ..utils.retry import ProviderHttpError, retry_provider_request
from ..utils.sanitize import sanitize_surrogates
from ..utils.sse import iterate_sse_messages
from ..utils.text import get_system_message_text, render_system_message_update
from .simple_options import (
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_max_tokens_to_context,
)
from .transform_messages import transform_messages

KAREN_USER_AGENT = "karen-ai/0.1.0"

AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]
AnthropicThinkingDisplay = Literal["summarized", "omitted"]

FINE_GRAINED_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"

# Stealth mode: mimic Claude Code's tool naming for OAuth (subscription) tokens.
_CLAUDE_CODE_VERSION = "2.1.280"
_CLAUDE_CODE_TOOLS = [
    "Read", "Write", "Edit", "Bash", "Grep", "Glob", "AskUserQuestion",
    "EnterPlanMode", "ExitPlanMode", "KillShell", "NotebookEdit", "Skill",
    "Task", "TaskOutput", "TodoWrite", "WebFetch", "WebSearch",
]
_CC_TOOL_LOOKUP = {t.lower(): t for t in _CLAUDE_CODE_TOOLS}


def _to_claude_code_name(name: str) -> str:
    return _CC_TOOL_LOOKUP.get(name.lower(), name)


def _from_claude_code_name(name: str, tools: Optional[List[Tool]] = None) -> str:
    if tools:
        lower = name.lower()
        for tool in tools:
            if tool.name.lower() == lower:
                return tool.name
    return name


class AnthropicOptions(StreamOptions):
    thinking_enabled: Optional[bool] = None
    thinking_budget_tokens: Optional[int] = None
    effort: Optional[AnthropicEffort] = None
    thinking_display: Optional[AnthropicThinkingDisplay] = None
    interleaved_thinking: Optional[bool] = None
    tool_choice: Optional[Any] = None


def _get_anthropic_compat(model: Model) -> AnthropicMessagesCompat:
    raw = model.compat
    compat = raw if isinstance(raw, AnthropicMessagesCompat) else (
        AnthropicMessagesCompat.model_validate(raw.model_dump()) if raw else AnthropicMessagesCompat()
    )
    is_openrouter = model.provider == "openrouter" or "openrouter.ai" in model.base_url
    # Apply URL/provider-derived defaults (compat fields win when set).
    if compat.send_session_affinity_headers is None:
        object.__setattr__(compat, "send_session_affinity_headers", is_openrouter)
    if compat.session_affinity_format is None and is_openrouter:
        object.__setattr__(compat, "session_affinity_format", "openrouter")
    return compat


def _resolve_cache_retention(cache_retention: Optional[CacheRetention], env: Optional[ProviderEnv]) -> CacheRetention:
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


def _get_cache_control(
    model: Model,
    compat: AnthropicMessagesCompat,
    cache_retention: Optional[CacheRetention],
    env: Optional[ProviderEnv],
) -> tuple[CacheRetention, Optional[Dict[str, Any]]]:
    retention = _resolve_cache_retention(cache_retention, env)
    if retention == "none":
        return retention, None
    ttl = "1h" if retention == "long" and (compat.supports_long_cache_retention is not False) else None
    control: Dict[str, Any] = {"type": "ephemeral"}
    if ttl:
        control["ttl"] = ttl
    return retention, control


def _is_oauth_token(api_key: str) -> bool:
    return "sk-ant-oat" in api_key


def _merge_headers(*header_sources: Optional[ProviderHeaders]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for headers in header_sources:
        for name, value in (headers or {}).items():
            if value is None:
                # None suppresses a default header with the same name.
                for existing in list(merged.keys()):
                    if existing.lower() == name.lower():
                        del merged[existing]
            else:
                merged[name] = value
    return merged


def _has_header(headers: Optional[ProviderHeaders], name: str) -> bool:
    if not headers:
        return False
    expected = name.lower()
    return any(key.lower() == expected and value and value.strip() for key, value in headers.items())


def _assert_request_auth(provider: str, api_key: Optional[str], headers: Optional[ProviderHeaders]) -> None:
    if api_key:
        return
    if _has_header(headers, "authorization") or _has_header(headers, "x-api-key") or _has_header(headers, "cf-aig-authorization"):
        return
    raise ValueError(f"No API key for provider: {provider}")


_ANTHROPIC_MESSAGE_EVENTS = {
    "message_start",
    "message_delta",
    "message_stop",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
}


async def _iterate_anthropic_events(
    response: httpx.Response,
    signal: Optional[AbortSignal] = None,
) -> AsyncIterator[Dict[str, Any]]:
    saw_message_start = False
    saw_message_end = False

    async for sse in iterate_sse_messages(response.aiter_text(), signal):
        if sse.event == "error":
            raise ProviderHttpError(sse.data)
        if sse.event not in _ANTHROPIC_MESSAGE_EVENTS:
            continue
        try:
            event = parse_json_with_repair(sse.data)
        except Exception as error:
            raise ProviderHttpError(
                f"Could not parse Anthropic SSE event {sse.event}: {error}; data={sse.data}"
            ) from error
        if event.get("type") == "message_start":
            saw_message_start = True
        elif event.get("type") == "message_stop":
            saw_message_end = True
        yield event

    if saw_message_start and not saw_message_end:
        raise ProviderHttpError("Anthropic stream ended before message_stop")


def _build_headers(
    model: Model,
    compat: AnthropicMessagesCompat,
    api_key: Optional[str],
    options_headers: Optional[ProviderHeaders],
    is_oauth: bool,
    session_id: Optional[str],
    beta_features: List[str],
) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "accept": "application/json",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "User-Agent": KAREN_USER_AGENT,
    }
    if beta_features:
        headers["anthropic-beta"] = ",".join(beta_features)
    if is_oauth:
        headers["user-agent"] = f"claude-cli/{_CLAUDE_CODE_VERSION}"
        headers["x-app"] = "cli"
        headers["authorization"] = f"Bearer {api_key}"
    elif api_key:
        headers["x-api-key"] = api_key

    if session_id and compat.send_session_affinity_headers:
        header = "x-session-id" if compat.session_affinity_format == "openrouter" else "x-session-affinity"
        headers[header] = session_id

    return _merge_headers(headers, model.headers, options_headers)


def _get_beta_features(
    model: Model,
    compat: AnthropicMessagesCompat,
    is_oauth: bool,
    options: Optional[AnthropicOptions],
) -> List[str]:
    configured: Optional[str] = None
    for headers in (model.headers, options.headers if options else None):
        for name, value in (headers or {}).items():
            if name.lower() == "anthropic-beta":
                configured = value
    if configured is not None:
        return list(dict.fromkeys(f.strip() for f in configured.split(",") if f.strip()))

    features: List[str] = []
    if is_oauth:
        features.extend(["claude-code-20250219", "oauth-2025-04-20"])
    if (
        model.reasoning
        and options
        and options.thinking_enabled is True
        and (options.interleaved_thinking is None or options.interleaved_thinking)
        and compat.force_adaptive_thinking is not True
    ):
        features.append(INTERLEAVED_THINKING_BETA)
    return list(dict.fromkeys(features))


def _convert_content_blocks(content: List[Union[TextContent, ImageContent]]) -> Union[str, List[Dict[str, Any]]]:
    """Convert content blocks to Anthropic API format."""
    has_images = any(c.type == "image" for c in content)
    if not has_images:
        return sanitize_surrogates("\n".join(c.text for c in content if c.type == "text"))

    blocks: List[Dict[str, Any]] = []
    for block in content:
        if block.type == "text":
            blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
        else:
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": block.mime_type, "data": block.data},
                }
            )
    if not any(b["type"] == "text" for b in blocks):
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})
    return blocks


def _convert_tool_result(msg: ToolResultMessage) -> Dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": msg.tool_call_id,
        "content": _convert_content_blocks(msg.content),
        "is_error": msg.is_error,
    }


def _params_for_single(msg, is_oauth: bool, allow_empty_signature: bool) -> Optional[Dict[str, Any]]:
    """Convert one non-toolResult message (helper for _convert_messages)."""
    if msg.role == "system":
        # Only reached when the model accepts mid-conversation system messages;
        # otherwise the transcript was collapsed before conversion.
        text = render_system_message_update(msg)
        if text:
            return {"role": "system", "content": [{"type": "text", "text": sanitize_surrogates(text)}]}
        return None

    if msg.role == "user":
        if isinstance(msg.content, str):
            if msg.content.strip():
                return {"role": "user", "content": sanitize_surrogates(msg.content)}
            return None
        blocks = []
        for item in msg.content:
            if item.type == "text":
                blocks.append({"type": "text", "text": sanitize_surrogates(item.text)})
            else:
                blocks.append(
                    {"type": "image", "source": {"type": "base64", "media_type": item.mime_type, "data": item.data}}
                )
        blocks = [b for b in blocks if b["type"] != "text" or b["text"].strip()]
        return {"role": "user", "content": blocks} if blocks else None

    if msg.role == "assistant":
        blocks = []
        for block in msg.content:
            if block.type == "text":
                if not block.text.strip():
                    continue
                blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
            elif block.type == "thinking":
                if block.redacted:
                    blocks.append({"type": "redacted_thinking", "data": block.thinking_signature})
                    continue
                signature = block.thinking_signature
                has_signature = bool(signature and signature.strip())
                if not block.thinking.strip() and not has_signature:
                    continue
                if not has_signature:
                    # Missing signature (e.g. aborted stream): convert to plain text
                    # unless the provider accepts empty signatures.
                    if allow_empty_signature:
                        blocks.append(
                            {"type": "thinking", "thinking": sanitize_surrogates(block.thinking), "signature": ""}
                        )
                    else:
                        blocks.append({"type": "text", "text": sanitize_surrogates(block.thinking)})
                else:
                    blocks.append(
                        {"type": "thinking", "thinking": sanitize_surrogates(block.thinking), "signature": signature}
                    )
            elif block.type == "toolCall":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": _to_claude_code_name(block.name) if is_oauth else block.name,
                        "input": block.arguments or {},
                    }
                )
        return {"role": "assistant", "content": blocks} if blocks else None

    return None


def _convert_messages(
    transformed_messages: List,
    is_oauth: bool,
    cache_control: Optional[Dict[str, Any]],
    allow_empty_signature: bool = False,
) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []

    i = 0
    while i < len(transformed_messages):
        msg = transformed_messages[i]
        if msg.role == "toolResult":
            # Collect all consecutive toolResult messages into one user message.
            tool_results = []
            while i < len(transformed_messages) and transformed_messages[i].role == "toolResult":
                tool_results.append(_convert_tool_result(transformed_messages[i]))
                i += 1
            params.append({"role": "user", "content": tool_results})
            continue
        single = _params_for_single(msg, is_oauth, allow_empty_signature)
        if single is not None:
            params.append(single)
        i += 1

    # Add cache_control to the last user/system message to cache conversation history.
    if cache_control and params:
        last = params[-1]
        if last["role"] in ("user", "system"):
            content = last["content"]
            if isinstance(content, list) and content:
                last_block = content[-1]
                if last_block.get("type") in ("text", "image", "tool_result", "tool_addition", "tool_removal"):
                    last_block["cache_control"] = cache_control
            elif isinstance(content, str):
                last["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]

    return params
    if msg.role == "system":
        text = render_system_message_update(msg)
        if text:
            return {"role": "system", "content": [{"type": "text", "text": sanitize_surrogates(text)}]}
        return None

    if msg.role == "user":
        if isinstance(msg.content, str):
            if msg.content.strip():
                return {"role": "user", "content": sanitize_surrogates(msg.content)}
            return None
        blocks = []
        for item in msg.content:
            if item.type == "text":
                blocks.append({"type": "text", "text": sanitize_surrogates(item.text)})
            else:
                blocks.append(
                    {"type": "image", "source": {"type": "base64", "media_type": item.mime_type, "data": item.data}}
                )
        blocks = [b for b in blocks if b["type"] != "text" or b["text"].strip()]
        return {"role": "user", "content": blocks} if blocks else None

    if msg.role == "assistant":
        blocks = []
        for block in msg.content:
            if block.type == "text":
                if not block.text.strip():
                    continue
                blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
            elif block.type == "thinking":
                if block.redacted:
                    blocks.append({"type": "redacted_thinking", "data": block.thinking_signature})
                    continue
                signature = block.thinking_signature
                has_signature = bool(signature and signature.strip())
                if not block.thinking.strip() and not has_signature:
                    continue
                if not has_signature:
                    if allow_empty_signature:
                        blocks.append(
                            {"type": "thinking", "thinking": sanitize_surrogates(block.thinking), "signature": ""}
                        )
                    else:
                        blocks.append({"type": "text", "text": sanitize_surrogates(block.thinking)})
                else:
                    blocks.append(
                        {"type": "thinking", "thinking": sanitize_surrogates(block.thinking), "signature": signature}
                    )
            elif block.type == "toolCall":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": _to_claude_code_name(block.name) if is_oauth else block.name,
                        "input": block.arguments or {},
                    }
                )
        return {"role": "assistant", "content": blocks} if blocks else None

    return None


def _convert_tools(tools: List[Tool], is_oauth: bool, cache_control: Optional[Dict[str, Any]], compat: AnthropicMessagesCompat) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        entry: Dict[str, Any] = {
            "name": _to_claude_code_name(tool.name) if is_oauth else tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
        converted.append(entry)
    if cache_control and converted and compat.supports_cache_control_on_tools is not False:
        converted[-1]["cache_control"] = cache_control
    return converted


def _build_params(
    model: Model,
    context: TranscriptContext,
    compat: AnthropicMessagesCompat,
    is_oauth: bool,
    options: Optional[AnthropicOptions],
) -> Dict[str, Any]:
    _, cache_control = _get_cache_control(
        model, compat, options.cache_retention if options else None, options.env if options else None
    )
    initial_system_message = get_initial_system_message(context.messages)
    initial_system_text = get_system_message_text(initial_system_message) if initial_system_message else ""
    transformed = transform_messages(context.messages, model, _normalize_tool_call_id)
    conversation_messages = transformed[1:] if initial_system_message else transformed

    params: Dict[str, Any] = {
        "model": model.id,
        "messages": _convert_messages(
            conversation_messages,
            is_oauth,
            cache_control,
            compat.allow_empty_signature is True,
        ),
        "max_tokens": (options.max_tokens if options and options.max_tokens else model.max_tokens),
        "stream": True,
    }

    # For OAuth tokens, we MUST include the Claude Code identity preamble.
    if is_oauth:
        system: List[Dict[str, Any]] = [
            {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."}
        ]
        if cache_control:
            system[0]["cache_control"] = cache_control
        if initial_system_text:
            block: Dict[str, Any] = {"type": "text", "text": sanitize_surrogates(initial_system_text)}
            if cache_control:
                block["cache_control"] = cache_control
            system.append(block)
        params["system"] = system
    elif initial_system_text:
        block = {"type": "text", "text": sanitize_surrogates(initial_system_text)}
        if cache_control:
            block["cache_control"] = cache_control
        params["system"] = [block]

    # Temperature is incompatible with extended thinking and unsupported on some models.
    if (
        options
        and options.temperature is not None
        and not options.thinking_enabled
        and compat.supports_temperature is not False
    ):
        params["temperature"] = options.temperature

    tools = get_current_tools(context.messages)
    if tools:
        params["tools"] = _convert_tools(tools, is_oauth, cache_control, compat)

    if model.reasoning:
        if options and options.thinking_enabled:
            display = options.thinking_display or "summarized"
            if compat.force_adaptive_thinking is True:
                params["thinking"] = {"type": "adaptive", "display": display}
                if options.effort:
                    params["output_config"] = {"effort": options.effort}
            else:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": options.thinking_budget_tokens or 1024,
                    "display": display,
                }
        elif options and options.thinking_enabled is False and (model.thinking_level_map or {}).get("off", "__missing__") is not None:
            params["thinking"] = {"type": "disabled"}

    if options and options.metadata:
        user_id = options.metadata.get("user_id")
        if isinstance(user_id, str):
            params["metadata"] = {"user_id": user_id}

    if options and options.tool_choice:
        if isinstance(options.tool_choice, str):
            params["tool_choice"] = {"type": options.tool_choice}
        else:
            params["tool_choice"] = options.tool_choice

    return params


def _normalize_tool_call_id(id: str) -> str:
    """Normalize tool call IDs to Anthropic's required pattern and length."""
    import re

    return re.sub(r"[^a-zA-Z0-9_-]", "_", id)[:64]


def _map_stop_reason(reason: Optional[str], stop_details: Optional[Dict[str, Any]]) -> tuple[str, Optional[str]]:
    if reason is None:
        return "stop", None
    mapping = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "toolUse",
        "refusal": "error",
        "pause_turn": "stop",
    }
    if reason in mapping:
        stop_reason = mapping[reason]
        if reason == "refusal":
            return stop_reason, "Anthropic refusal stop"
        return stop_reason, None
    return "error", f"Anthropic stop reason: {reason}"


def stream(
    model: Model,
    context: TranscriptContext,
    options: Optional[AnthropicOptions] = None,
) -> AssistantMessageEventStream:
    """Stream a normalized transcript through the Anthropic Messages API."""
    event_stream = AssistantMessageEventStream()
    compat = _get_anthropic_compat(model)
    normalized_context = resolve_transcript(context, compat.supports_mid_convo_system_messages)
    current_tools = get_current_tools(normalized_context.messages)

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
            api_key = options.api_key if options else None
            _assert_request_auth(model.provider, api_key, options.headers if options else None)
            is_oauth = bool(api_key and _is_oauth_token(api_key))

            cache_retention = _resolve_cache_retention(
                options.cache_retention if options else None, options.env if options else None
            )
            cache_session_id = options.session_id if cache_retention != "none" and options else None

            beta_features = _get_beta_features(model, compat, is_oauth, options)
            params = _build_params(model, normalized_context, compat, is_oauth, options)

            if options and options.on_payload:
                next_params = options.on_payload(params, model)
                if asyncio.iscoroutine(next_params):
                    next_params = await next_params
                if next_params is not None:
                    params = {**next_params, "stream": True}

            headers = _build_headers(model, compat, api_key, options.headers if options else None, is_oauth, cache_session_id, beta_features)

            timeout = httpx.Timeout(
                (options.timeout_ms or 600_000) / 1000,
                connect=60.0,
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                url = model.base_url.rstrip("/") + "/v1/messages"

                async def do_request() -> httpx.Response:
                    request = client.build_request("POST", url, json=params, headers=headers)
                    response = await client.send(request, stream=True)
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        await response.aclose()
                        raise ProviderHttpError(
                            f"Anthropic API error {response.status_code}: {body}",
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

                    # Streaming state: blocks carry a scratch `_wire_index` (Anthropic's
                    # content-block index) mapped into output.content positions.
                    blocks: List[ContentBlock] = output.content
                    wire_index_to_block: Dict[int, int] = {}

                    async for event in _iterate_anthropic_events(response, options.signal if options else None):
                        if options and options.on_provider_stream_event:
                            maybe = options.on_provider_stream_event(event, model)
                            if asyncio.iscoroutine(maybe):
                                await maybe

                        event_type = event.get("type")

                        if event_type == "message_start":
                            message = event.get("message", {})
                            output.response_id = message.get("id")
                            response_model = message.get("model")
                            if response_model and response_model != model.id:
                                output.response_model = response_model
                            usage = message.get("usage") or {}
                            output.usage.input = usage.get("input_tokens") or 0
                            output.usage.output = usage.get("output_tokens") or 0
                            output.usage.cache_read = usage.get("cache_read_input_tokens") or 0
                            output.usage.cache_write = usage.get("cache_creation_input_tokens") or 0
                            cache_creation = usage.get("cache_creation") or {}
                            output.usage.cache_write1h = cache_creation.get("ephemeral_1h_input_tokens") or 0
                            output.usage.total_tokens = (
                                output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write
                            )
                            calculate_cost(model, output.usage)

                        elif event_type == "content_block_start":
                            content_block = event.get("content_block", {})
                            wire_index = event.get("index", 0)
                            block_type = content_block.get("type")

                            if block_type == "text":
                                block = TextContent(text=content_block.get("text") or "")
                                blocks.append(block)
                                wire_index_to_block[wire_index] = len(blocks) - 1
                                event_stream.push(TextStartEvent(content_index=len(blocks) - 1, partial=output))
                            elif block_type == "thinking":
                                block = ThinkingContent(
                                    thinking=content_block.get("thinking") or "",
                                    thinking_signature=content_block.get("signature") or "",
                                )
                                blocks.append(block)
                                wire_index_to_block[wire_index] = len(blocks) - 1
                                event_stream.push(ThinkingStartEvent(content_index=len(blocks) - 1, partial=output))
                            elif block_type == "redacted_thinking":
                                block = ThinkingContent(
                                    thinking="[Reasoning redacted]",
                                    thinking_signature=content_block.get("data"),
                                    redacted=True,
                                )
                                blocks.append(block)
                                wire_index_to_block[wire_index] = len(blocks) - 1
                                event_stream.push(ThinkingStartEvent(content_index=len(blocks) - 1, partial=output))
                            elif block_type == "tool_use":
                                name = content_block.get("name", "")
                                block = ToolCall(
                                    id=content_block.get("id", ""),
                                    name=_from_claude_code_name(name, current_tools) if is_oauth else name,
                                    arguments=content_block.get("input") or {},
                                )
                                block.partial_json = ""
                                blocks.append(block)
                                wire_index_to_block[wire_index] = len(blocks) - 1
                                event_stream.push(ToolCallStartEvent(content_index=len(blocks) - 1, partial=output))

                        elif event_type == "content_block_delta":
                            wire_index = event.get("index", 0)
                            delta = event.get("delta", {})
                            content_index = wire_index_to_block.get(wire_index)
                            if content_index is None:
                                continue
                            block = blocks[content_index]
                            delta_type = delta.get("type")

                            if delta_type == "text_delta" and isinstance(block, TextContent):
                                block.text += delta.get("text", "")
                                event_stream.push(
                                    TextDeltaEvent(content_index=content_index, delta=delta.get("text", ""), partial=output)
                                )
                            elif delta_type == "thinking_delta" and isinstance(block, ThinkingContent):
                                block.thinking += delta.get("thinking", "")
                                event_stream.push(
                                    ThinkingDeltaEvent(
                                        content_index=content_index, delta=delta.get("thinking", ""), partial=output
                                    )
                                )
                            elif delta_type == "input_json_delta" and isinstance(block, ToolCall):
                                block.partial_json = (block.partial_json or "") + delta.get("partial_json", "")
                                block.arguments = parse_streaming_json(block.partial_json)
                                event_stream.push(
                                    ToolCallDeltaEvent(
                                        content_index=content_index,
                                        delta=delta.get("partial_json", ""),
                                        partial=output,
                                    )
                                )
                            elif delta_type == "signature_delta" and isinstance(block, ThinkingContent):
                                block.thinking_signature = (block.thinking_signature or "") + delta.get("signature", "")

                        elif event_type == "content_block_stop":
                            wire_index = event.get("index", 0)
                            content_index = wire_index_to_block.pop(wire_index, None)
                            if content_index is None:
                                continue
                            block = blocks[content_index]
                            if isinstance(block, TextContent):
                                event_stream.push(TextEndEvent(content_index=content_index, content=block.text, partial=output))
                            elif isinstance(block, ThinkingContent):
                                event_stream.push(
                                    ThinkingEndEvent(content_index=content_index, content=block.thinking, partial=output)
                                )
                            elif isinstance(block, ToolCall):
                                block.arguments = parse_streaming_json(block.partial_json)
                                block.partial_json = None
                                event_stream.push(
                                    ToolCallEndEvent(content_index=content_index, tool_call=block, partial=output)
                                )

                        elif event_type == "message_delta":
                            delta = event.get("delta", {})
                            if delta.get("stop_reason"):
                                output.raw_stop_reason = delta["stop_reason"]
                                stop_reason, error_message = _map_stop_reason(delta["stop_reason"], delta.get("stop_details"))
                                output.stop_reason = stop_reason  # type: ignore[assignment]
                                if error_message:
                                    output.error_message = error_message
                            usage = event.get("usage")
                            if usage:
                                if usage.get("input_tokens") is not None:
                                    output.usage.input = usage["input_tokens"]
                                if usage.get("output_tokens") is not None:
                                    output.usage.output = usage["output_tokens"]
                                if usage.get("cache_read_input_tokens") is not None:
                                    output.usage.cache_read = usage["cache_read_input_tokens"]
                                if usage.get("cache_creation_input_tokens") is not None:
                                    output.usage.cache_write = usage["cache_creation_input_tokens"]
                                cache_creation = usage.get("cache_creation") or {}
                                if cache_creation.get("ephemeral_1h_input_tokens") is not None:
                                    output.usage.cache_write1h = cache_creation["ephemeral_1h_input_tokens"]
                                details = usage.get("output_tokens_details") or {}
                                if details.get("thinking_tokens") is not None:
                                    output.usage.reasoning = details["thinking_tokens"]
                            output.usage.total_tokens = (
                                output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write
                            )
                            calculate_cost(model, output.usage)

                    if options and options.signal and options.signal.aborted:
                        raise RuntimeError("Request was aborted")
                    if output.stop_reason == "pending":
                        raise RuntimeError("Anthropic stream ended without a stop reason")
                    if output.stop_reason in ("aborted", "error"):
                        raise RuntimeError(output.error_message or "An unknown error occurred")

                    event_stream.push(DoneEvent(reason=output.stop_reason, message=output))  # type: ignore[arg-type]
                    event_stream.end()
                finally:
                    await response.aclose()

        except Exception as error:
            output.stop_reason = "aborted" if (options and options.signal and options.signal.aborted) else "error"
            output.error_message = str(error)
            event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))  # type: ignore[arg-type]
            event_stream.end()

    asyncio.get_running_loop().create_task(run())
    return event_stream


def _map_thinking_level_to_effort(model: Model, level: Optional[str]) -> AnthropicEffort:
    mapped = (model.thinking_level_map or {}).get(level) if level else None
    if isinstance(mapped, str):
        return mapped  # type: ignore[return-value]
    if level in ("minimal", "low"):
        return "low"
    if level == "medium":
        return "medium"
    return "high"


def stream_simple(
    model: Model,
    context: TranscriptContext,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    _assert_request_auth(model.provider, options.api_key if options else None, options.headers if options else None)

    base = build_base_options(model, context, options, options.api_key if options else None)
    base_options = AnthropicOptions(**base.model_dump(exclude_none=True))
    base_options.tool_choice = options.tool_choice if options else None

    compat = _get_anthropic_compat(model)

    if not options or not options.reasoning:
        base_options.thinking_enabled = False
        return stream(model, context, base_options)

    # Adaptive thinking models use an effort level; older models use budget-based thinking.
    if compat.force_adaptive_thinking is True:
        base_options.thinking_enabled = True
        base_options.effort = _map_thinking_level_to_effort(model, options.reasoning)
        return stream(model, context, base_options)

    max_tokens, thinking_budget = adjust_max_tokens_for_thinking(
        base.max_tokens,
        model.max_tokens,
        options.reasoning,
        options.thinking_budgets if options else None,
    )
    clamped_max = clamp_max_tokens_to_context(model, context, max_tokens)

    base_options.max_tokens = clamped_max
    base_options.thinking_enabled = True
    base_options.thinking_budget_tokens = min(thinking_budget, max(0, clamped_max - 1024))
    return stream(model, context, base_options)

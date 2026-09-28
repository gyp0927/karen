"""OpenAI Chat Completions API adapter, mirroring pi-ai's api/openai-completions.ts.

The most universal adapter: works with OpenAI itself and the many
OpenAI-compatible endpoints (DeepSeek, Moonshot, OpenRouter, vLLM, ...),
driven by per-model compat settings auto-detected from provider/base URL.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Union

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
    OpenAICompletionsCompat,
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
    TranscriptContext,
    Usage,
)
from ..utils.json_parse import parse_streaming_json
from ..utils.provider_env import get_provider_env_value
from ..utils.retry import ProviderHttpError, retry_provider_request
from ..utils.sanitize import sanitize_surrogates
from ..utils.sse import iterate_sse_messages
from ..utils.text import get_system_message_text, render_system_message_update
from .simple_options import (
    build_base_options,
    clamp_thinking_budget_to_answer_room,
    thinking_budget_for_level,
)
from .transform_messages import transform_messages

KAREN_USER_AGENT = "karen-ai/0.1.0"


class OpenAICompletionsOptions(StreamOptions):
    tool_choice: Optional[Any] = None
    reasoning_effort: Optional[str] = None
    thinking_budgets: Optional[Any] = None


# ---------------------------------------------------------------------------
# Compat resolution
# ---------------------------------------------------------------------------


class _ResolvedCompat:
    """Concrete compat with all defaults applied (mirrors ResolvedOpenAICompletionsCompat)."""

    def __init__(self, model: Model):
        base = _detect_compat(model)
        override = model.compat
        if override is not None and not isinstance(override, OpenAICompletionsCompat):
            override = OpenAICompletionsCompat.model_validate(override.model_dump())
        for field_name in OpenAICompletionsCompat.model_fields:
            value = getattr(override, field_name, None) if override else None
            if value is not None:
                setattr(self, field_name, value)
            else:
                setattr(self, field_name, getattr(base, field_name))


def _detect_compat(model: Model) -> OpenAICompletionsCompat:
    """Auto-detect compatibility settings from provider name and base URL."""
    provider = model.provider
    base_url = model.base_url

    is_zai = provider in ("zai", "zai-coding-cn") or "api.z.ai" in base_url or "open.bigmodel.cn" in base_url
    is_together = provider == "together" or "api.together.ai" in base_url or "api.together.xyz" in base_url
    is_moonshot = provider in ("moonshotai", "moonshotai-cn") or "api.moonshot." in base_url
    is_openrouter = provider == "openrouter" or "openrouter.ai" in base_url
    is_cloudflare_workers = provider == "cloudflare-workers-ai" or "api.cloudflare.com" in base_url
    is_cloudflare_gateway = provider == "cloudflare-ai-gateway" or "gateway.ai.cloudflare.com" in base_url
    is_nvidia = provider == "nvidia" or "integrate.api.nvidia.com" in base_url
    is_ant_ling = provider == "ant-ling" or "api.ant-ling.com" in base_url
    is_cerebras = provider == "cerebras" or "cerebras.ai" in base_url
    is_deepseek = provider == "deepseek" or "deepseek.com" in base_url.lower()
    is_grok = provider == "xai" or "api.x.ai" in base_url

    is_non_standard = (
        is_nvidia
        or is_cerebras
        or is_grok
        or is_together
        or "chutes.ai" in base_url
        or is_deepseek
        or is_zai
        or is_moonshot
        or provider == "opencode"
        or "opencode.ai" in base_url
        or is_cloudflare_workers
        or is_cloudflare_gateway
        or is_ant_ling
    )

    use_max_tokens = (
        "chutes.ai" in base_url
        or is_deepseek
        or is_moonshot
        or is_cloudflare_gateway
        or is_together
        or is_nvidia
        or is_ant_ling
        or is_zai
    )

    is_openrouter_developer_role_model = is_openrouter and (
        model.id.startswith("anthropic/") or model.id.startswith("openai/")
    )
    cache_control_format = "anthropic" if (provider == "openrouter" and model.id.startswith("anthropic/")) else None

    if is_deepseek:
        thinking_format = "deepseek"
    elif is_zai:
        thinking_format = "zai"
    elif is_moonshot:
        thinking_format = "openai"
    elif is_openrouter:
        thinking_format = "openrouter"
    elif is_together:
        thinking_format = "together"
    elif is_ant_ling:
        thinking_format = "ant-ling"
    else:
        thinking_format = "openai"

    return OpenAICompletionsCompat(
        supports_store=not is_non_standard,
        supports_developer_role=is_openrouter_developer_role_model or (not is_non_standard and not is_openrouter),
        supports_reasoning_effort=not (
            is_grok or is_zai or is_moonshot or is_together or is_cloudflare_gateway or is_nvidia or is_ant_ling
        ),
        supports_usage_in_streaming=True,
        supports_finish_reason=True,
        max_tokens_field="max_tokens" if use_max_tokens else "max_completion_tokens",
        requires_tool_result_name=False,
        requires_assistant_after_tool_result=False,
        requires_thinking_as_text=False,
        requires_reasoning_content_on_assistant_messages=is_deepseek,
        thinking_format=thinking_format,
        supports_openai_grammar_tools=False,
        supports_mid_convo_system_messages=False,
        supports_mid_convo_tool_additions=False,
        supports_strict_mode=False,
        cache_control_format=cache_control_format,
        send_session_affinity_headers=is_openrouter,
        session_affinity_format="openrouter" if is_openrouter else None,
        supports_long_cache_retention=True,
    )


def _resolve_cache_retention(cache_retention: Optional[CacheRetention], env: Optional[ProviderEnv]) -> CacheRetention:
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


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
        headers[name] = value

    if session_id and compat.send_session_affinity_headers:
        if compat.session_affinity_format == "openrouter":
            headers["x-session-id"] = session_id
        else:
            if compat.session_affinity_format == "openai":
                headers["session_id"] = session_id
            headers["x-client-request-id"] = session_id
            headers["x-session-affinity"] = session_id

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


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _convert_tools(tools: List[Tool], compat: _ResolvedCompat) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        function: Dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if compat.supports_strict_mode is not False:
            function["strict"] = False
        converted.append({"type": "function", "function": function})
    return converted


def _convert_messages(
    model: Model,
    context: TranscriptContext,
    compat: _ResolvedCompat,
) -> List[Dict[str, Any]]:
    normalized_context = resolve_transcript(context, compat.supports_mid_convo_system_messages)
    params: List[Dict[str, Any]] = []

    def normalize_tool_call_id(id: str) -> str:
        # Handle pipe-separated IDs from OpenAI Responses-style APIs.
        if "|" in id:
            call_id, _, item_id = id.partition("|")
            call_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in call_id)
            item_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in item_id)
            combined = f"{call_id}_{item_id}" if item_id else call_id
            if len(combined) <= 40:
                return combined
            hash_part = _short_hash(id)[:8]
            prefix = call_id[: max(1, 40 - len(hash_part) - 1)]
            return f"{prefix}_{hash_part}"
        if model.provider == "openai":
            return id[:40]
        return id

    transformed = transform_messages(normalized_context.messages, model, normalize_tool_call_id)
    transcript_tools = resolve_transcript_tools(
        normalized_context.messages,
        compat.supports_mid_convo_system_messages is True and compat.supports_mid_convo_tool_additions is True,
    )
    instruction_role = "developer" if (model.reasoning and compat.supports_developer_role) else "system"

    last_role: Optional[str] = None
    i = 0
    while i < len(transformed):
        msg = transformed[i]

        # Some providers don't allow user messages directly after tool results.
        if compat.requires_assistant_after_tool_result and last_role == "toolResult" and msg.role == "user":
            params.append({"role": "assistant", "content": "I have processed the tool results."})

        if msg.role == "system":
            added_tools = msg.tools_added if (i > 0 and transcript_tools.anchors_additions) else []
            if added_tools:
                params.append({"role": "system", "tools": _convert_tools(added_tools, compat)})
            text = get_system_message_text(msg) if i == 0 else render_system_message_update(msg)
            if text:
                params.append({"role": instruction_role, "content": sanitize_surrogates(text)})

        elif msg.role == "user":
            if isinstance(msg.content, str):
                params.append({"role": "user", "content": sanitize_surrogates(msg.content)})
            else:
                content = []
                for item in msg.content:
                    if item.type == "text":
                        if item.text:
                            content.append({"type": "text", "text": sanitize_surrogates(item.text)})
                    else:
                        content.append(
                            {"type": "image_url", "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"}}
                        )
                if not content:
                    i += 1
                    last_role = msg.role
                    continue
                params.append({"role": "user", "content": content})

        elif msg.role == "assistant":
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": "" if compat.requires_assistant_after_tool_result else None,
            }

            text_parts = [b for b in msg.content if b.type == "text" and b.text.strip()]
            assistant_text = "".join(sanitize_surrogates(b.text) for b in text_parts)
            thinking_blocks = [b for b in msg.content if b.type == "thinking"]
            tool_calls = [b for b in msg.content if b.type == "toolCall"]

            non_empty_thinking = [b for b in thinking_blocks if b.thinking.strip()]
            if non_empty_thinking:
                if compat.requires_thinking_as_text:
                    thinking_text = "\n\n".join(sanitize_surrogates(b.thinking) for b in non_empty_thinking)
                    assistant_msg["content"] = [{"type": "text", "text": thinking_text}] + [
                        {"type": "text", "text": sanitize_surrogates(b.text)} for b in text_parts
                    ]
                else:
                    # Always send assistant content as a plain string (Chat Completions
                    # standard); content-block arrays make some models mirror the structure.
                    if assistant_text:
                        assistant_msg["content"] = assistant_text
                    signature = non_empty_thinking[0].thinking_signature
                    if signature in ("reasoning", "reasoning_content", "reasoning_text"):
                        assistant_msg[signature] = "\n".join(b.thinking for b in non_empty_thinking)
            elif assistant_text:
                assistant_msg["content"] = assistant_text

            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in tool_calls
                ]

            if (
                compat.requires_reasoning_content_on_assistant_messages
                and model.reasoning
                and "reasoning_content" not in assistant_msg
            ):
                assistant_msg["reasoning_content"] = ""

            content = assistant_msg["content"]
            has_content = content is not None and len(content) > 0
            if not has_content and "tool_calls" not in assistant_msg:
                i += 1
                last_role = msg.role
                continue
            params.append(assistant_msg)

        elif msg.role == "toolResult":
            image_blocks: List[Dict[str, Any]] = []
            while i < len(transformed) and transformed[i].role == "toolResult":
                tool_msg = transformed[i]
                text_result = "\n".join(b.text for b in tool_msg.content if b.type == "text")
                has_images = any(c.type == "image" for c in tool_msg.content)
                tool_result_text = text_result or ("(see attached image)" if has_images else "(no tool output)")
                tool_result_msg: Dict[str, Any] = {
                    "role": "tool",
                    "content": sanitize_surrogates(tool_result_text),
                    "tool_call_id": tool_msg.tool_call_id,
                }
                if compat.requires_tool_result_name and tool_msg.tool_name:
                    tool_result_msg["name"] = tool_msg.tool_name
                params.append(tool_result_msg)

                if has_images and "image" in model.input:
                    for block in tool_msg.content:
                        if block.type == "image":
                            image_blocks.append(
                                {"type": "image_url", "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"}}
                            )
                i += 1

            if image_blocks:
                if compat.requires_assistant_after_tool_result:
                    params.append({"role": "assistant", "content": "I have processed the tool results."})
                params.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Attached image(s) from tool result:"}, *image_blocks],
                    }
                )
                last_role = "user"
            else:
                last_role = "toolResult"
            continue

        last_role = msg.role
        i += 1

    return params


def _resolve_thinking_token_budget_field(compat: _ResolvedCompat) -> Optional[str]:
    if compat.thinking_token_budget_field:
        return compat.thinking_token_budget_field
    if compat.supports_thinking_token_budget:
        return "thinking_token_budget"
    return None


def _build_params(
    model: Model,
    context: TranscriptContext,
    options: Optional[OpenAICompletionsOptions],
    compat: _ResolvedCompat,
    cache_retention: CacheRetention,
) -> Dict[str, Any]:
    transcript_tools = resolve_transcript_tools(
        context.messages,
        compat.supports_mid_convo_system_messages is True and compat.supports_mid_convo_tool_additions is True,
    )
    messages = _convert_messages(model, context, compat)

    params: Dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": True,
    }

    if ("api.openai.com" in model.base_url and cache_retention != "none") or (
        cache_retention == "long" and compat.supports_long_cache_retention
    ):
        if options and options.session_id:
            params["prompt_cache_key"] = options.session_id[:64]
    if cache_retention == "long" and compat.supports_long_cache_retention:
        params["prompt_cache_retention"] = "24h"

    if compat.supports_usage_in_streaming is not False:
        params["stream_options"] = {"include_usage": True}
    if compat.supports_store:
        params["store"] = False

    if options and options.max_tokens:
        if compat.max_tokens_field == "max_tokens":
            params["max_tokens"] = options.max_tokens
        else:
            params["max_completion_tokens"] = options.max_tokens

    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    if transcript_tools.request_tools:
        params["tools"] = _convert_tools(transcript_tools.request_tools, compat)
        if compat.zai_tool_stream:
            params["tool_stream"] = True
    elif _has_tool_history(context.messages):
        # Some proxies require the tools param when messages include tool history.
        params["tools"] = []

    if options and options.tool_choice:
        params["tool_choice"] = options.tool_choice

    if compat.vllm_priority is not None:
        params["priority"] = compat.vllm_priority

    _apply_thinking_params(params, model, options, compat)

    # OpenRouter / Vercel AI Gateway routing preferences.
    if compat.open_router_routing:
        params["provider"] = compat.open_router_routing.model_dump(by_alias=True, exclude_none=True)
    if compat.vercel_gateway_routing:
        routing = compat.vercel_gateway_routing
        gateway_options = {}
        if routing.only:
            gateway_options["only"] = routing.only
        if routing.order:
            gateway_options["order"] = routing.order
        if gateway_options:
            params["providerOptions"] = {"gateway": gateway_options}

    # Last so custom keys override the named request fields.
    if model.sampling_params:
        params.update(model.sampling_params)
    if options and options.sampling_params:
        params.update(options.sampling_params)

    return params


def _apply_thinking_params(
    params: Dict[str, Any],
    model: Model,
    options: Optional[OpenAICompletionsOptions],
    compat: _ResolvedCompat,
) -> None:
    reasoning_effort = options.reasoning_effort if options else None
    thinking_format = compat.thinking_format or "openai"
    level_map = model.thinking_level_map or {}

    # Token budget for top-level budget fields (shared-ceiling providers).
    thinking_budget: Optional[int] = None
    if reasoning_effort and model.reasoning:
        ceiling = params.get("max_tokens") or params.get("max_completion_tokens") or model.max_tokens
        budget = clamp_thinking_budget_to_answer_room(
            thinking_budget_for_level(reasoning_effort, options.thinking_budgets if options else None),
            ceiling,
        )
        thinking_budget = budget if budget > 0 else None

    if thinking_format == "zai" and model.reasoning:
        params["thinking"] = {"type": "enabled", "clear_thinking": False} if reasoning_effort else {"type": "disabled"}
        if reasoning_effort and compat.supports_reasoning_effort:
            effort = level_map.get(reasoning_effort, reasoning_effort)
            if isinstance(effort, str):
                params["reasoning_effort"] = effort
    elif thinking_format == "qwen" and model.reasoning:
        params["enable_thinking"] = bool(reasoning_effort)
        if reasoning_effort and compat.supports_reasoning_effort:
            effort = level_map.get(reasoning_effort, reasoning_effort)
            if isinstance(effort, str):
                params["reasoning_effort"] = effort
    elif thinking_format == "qwen-chat-template" and model.reasoning:
        params["chat_template_kwargs"] = {"enable_thinking": bool(reasoning_effort), "preserve_thinking": True}
    elif thinking_format == "chat-template" and model.reasoning:
        kwargs = _build_chat_template_values(model, options, compat.chat_template_kwargs or {}, thinking_budget)
        if kwargs:
            params["chat_template_kwargs"] = kwargs
    elif thinking_format == "baseten" and model.reasoning:
        args = _build_chat_template_values(model, options, compat.chat_template_args or {}, thinking_budget)
        if args:
            params["chat_template_args"] = args
        if compat.supports_reasoning_effort:
            effort = level_map.get(reasoning_effort) if reasoning_effort else level_map.get("off")
            effort = reasoning_effort if effort is None else effort
            if isinstance(effort, str):
                params["reasoning_effort"] = effort
    elif thinking_format == "deepseek" and model.reasoning:
        if reasoning_effort:
            params["thinking"] = {"type": "enabled"}
        elif level_map.get("off", "__missing__") is not None:
            params["thinking"] = {"type": "disabled"}
        if reasoning_effort and compat.supports_reasoning_effort:
            params["reasoning_effort"] = level_map.get(reasoning_effort, reasoning_effort)
    elif thinking_format == "openrouter" and model.reasoning:
        if reasoning_effort:
            params["reasoning"] = {"effort": level_map.get(reasoning_effort, reasoning_effort)}
        elif level_map.get("off", "__missing__") is not None:
            params["reasoning"] = {"effort": level_map.get("off") or "none"}
    elif thinking_format == "ant-ling" and model.reasoning and reasoning_effort:
        effort = level_map.get(reasoning_effort)
        if isinstance(effort, str):
            params["reasoning"] = {"effort": effort}
    elif thinking_format == "together" and model.reasoning:
        params["reasoning"] = {"enabled": bool(reasoning_effort)}
        if reasoning_effort and compat.supports_reasoning_effort:
            params["reasoning_effort"] = level_map.get(reasoning_effort, reasoning_effort)
    elif thinking_format == "string-thinking" and model.reasoning:
        if reasoning_effort:
            params["thinking"] = level_map.get(reasoning_effort, reasoning_effort)
        elif level_map.get("off", "__missing__") is not None:
            params["thinking"] = level_map.get("off") or "none"
    elif reasoning_effort and model.reasoning and compat.supports_reasoning_effort:
        params["reasoning_effort"] = level_map.get(reasoning_effort, reasoning_effort)
    elif not reasoning_effort and model.reasoning and compat.supports_reasoning_effort:
        off_value = level_map.get("off")
        if isinstance(off_value, str):
            params["reasoning_effort"] = off_value

    budget_field = _resolve_thinking_token_budget_field(compat)
    if budget_field and thinking_budget is not None:
        params[budget_field] = thinking_budget


def _resolve_chat_template_kwarg_value(
    model: Model,
    options: Optional[OpenAICompletionsOptions],
    value: Any,
    thinking_budget: Optional[int],
) -> Any:
    from ..types import ChatTemplateKwargVar

    if not isinstance(value, ChatTemplateKwargVar):
        if isinstance(value, dict) and "$var" in value:
            value = ChatTemplateKwargVar.model_validate(value)
        else:
            return value

    reasoning_effort = options.reasoning_effort if options else None
    if not reasoning_effort and value.omit_when_off:
        return _OMIT
    if value.var == "thinking.enabled":
        return bool(reasoning_effort)
    if value.var == "thinking.budget":
        return thinking_budget

    level_map = model.thinking_level_map or {}
    mapped = level_map.get(reasoning_effort) if reasoning_effort else level_map.get("off")
    if mapped is None:
        return reasoning_effort if mapped is None and reasoning_effort else _OMIT
    return mapped


_OMIT = object()


def _build_chat_template_values(
    model: Model,
    options: Optional[OpenAICompletionsOptions],
    values: Dict[str, Any],
    thinking_budget: Optional[int],
) -> Optional[Dict[str, Any]]:
    resolved: Dict[str, Any] = {}
    for key, value in values.items():
        result = _resolve_chat_template_kwarg_value(model, options, value, thinking_budget)
        if result is not _OMIT:
            resolved[key] = result
    return resolved or None


def _has_tool_history(messages) -> bool:
    for msg in messages:
        if msg.role == "toolResult":
            return True
        if msg.role == "assistant" and any(b.type == "toolCall" for b in msg.content):
            return True
    return False


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_chunk_usage(raw_usage: Dict[str, Any], model: Model) -> Usage:
    prompt_tokens = raw_usage.get("prompt_tokens") or 0
    prompt_details = raw_usage.get("prompt_tokens_details") or {}
    cache_read_tokens = (
        prompt_details.get("cached_tokens")
        or raw_usage.get("prompt_cache_hit_tokens")
        or raw_usage.get("cached_tokens")
        or 0
    )
    cache_write_tokens = prompt_details.get("cache_write_tokens") or 0

    # Documented OpenAI/OpenRouter semantics: cached_tokens is cache-read hits.
    input_tokens = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)
    output_tokens = raw_usage.get("completion_tokens") or 0
    completion_details = raw_usage.get("completion_tokens_details") or {}
    usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read_tokens,
        cache_write=cache_write_tokens,
        reasoning=completion_details.get("reasoning_tokens") or 0,
        total_tokens=input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
    )
    calculate_cost(model, usage)
    return usage


def _map_stop_reason(reason: Optional[str]) -> tuple[str, Optional[str]]:
    if reason is None:
        return "stop", None
    mapping = {
        "stop": "stop",
        "end": "stop",
        "length": "length",
        "function_call": "toolUse",
        "tool_calls": "toolUse",
    }
    if reason in mapping:
        return mapping[reason], None
    if reason == "content_filter":
        return "error", "Provider finish_reason: content_filter"
    if reason == "network_error":
        return "error", "Provider finish_reason: network_error"
    return "error", f"Provider finish_reason: {reason}"


_REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


async def _iterate_openai_chunks(response: httpx.Response, signal=None):
    async for sse in iterate_sse_messages(response.aiter_text(), signal):
        data = sse.data.strip()
        if not data:
            continue
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


# ---------------------------------------------------------------------------
# stream / stream_simple
# ---------------------------------------------------------------------------


def stream(
    model: Model,
    context: TranscriptContext,
    options: Optional[OpenAICompletionsOptions] = None,
) -> AssistantMessageEventStream:
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
            api_key = _get_client_api_key(model.provider, options.api_key if options else None, options.headers if options else None)
            cache_retention = _resolve_cache_retention(
                options.cache_retention if options else None, options.env if options else None
            )
            cache_session_id = options.session_id if cache_retention != "none" and options else None

            params = _build_params(model, normalized_context, options, compat, cache_retention)
            if options and options.on_payload:
                next_params = options.on_payload(params, model)
                if asyncio.iscoroutine(next_params):
                    next_params = await next_params
                if next_params is not None:
                    params = next_params

            headers = _build_headers(model, api_key, options.headers if options else None, cache_session_id, compat)

            timeout = httpx.Timeout((options.timeout_ms or 600_000) / 1000, connect=60.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                url = model.base_url.rstrip("/") + "/chat/completions"

                async def do_request() -> httpx.Response:
                    request = client.build_request("POST", url, json=params, headers=headers)
                    response = await client.send(request, stream=True)
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        await response.aclose()
                        raise ProviderHttpError(
                            f"OpenAI-compatible API error {response.status_code}: {body}",
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

                    blocks = output.content
                    text_block: Optional[TextContent] = None
                    thinking_block: Optional[ThinkingContent] = None
                    has_finish_reason = False
                    tool_call_blocks_by_index: Dict[int, ToolCall] = {}
                    tool_call_blocks_by_id: Dict[str, ToolCall] = {}

                    def ensure_text_block() -> TextContent:
                        nonlocal text_block
                        if text_block is None:
                            text_block = TextContent(text="")
                            blocks.append(text_block)
                            event_stream.push(TextStartEvent(content_index=len(blocks) - 1, partial=output))
                        return text_block

                    def ensure_thinking_block(signature: str) -> ThinkingContent:
                        nonlocal thinking_block
                        if thinking_block is None:
                            thinking_block = ThinkingContent(thinking="", thinking_signature=signature)
                            blocks.append(thinking_block)
                            event_stream.push(ThinkingStartEvent(content_index=len(blocks) - 1, partial=output))
                        return thinking_block

                    def ensure_tool_call_block(tool_call: Dict[str, Any]) -> ToolCall:
                        stream_index = tool_call.get("index") if isinstance(tool_call.get("index"), int) else None
                        function = tool_call.get("function") or {}
                        name = function.get("name") or ""
                        block: Optional[ToolCall] = None
                        if stream_index is not None:
                            block = tool_call_blocks_by_index.get(stream_index)
                        if block is None and tool_call.get("id"):
                            block = tool_call_blocks_by_id.get(tool_call["id"])
                        if block is None:
                            block = ToolCall(id=tool_call.get("id") or "", name=name, arguments={})
                            block.partial_args = ""
                            block.stream_index = stream_index
                            if stream_index is not None:
                                tool_call_blocks_by_index[stream_index] = block
                            if tool_call.get("id"):
                                tool_call_blocks_by_id[tool_call["id"]] = block
                            blocks.append(block)
                            event_stream.push(ToolCallStartEvent(content_index=len(blocks) - 1, partial=output))
                        if stream_index is not None and block.stream_index is None:
                            block.stream_index = stream_index
                            tool_call_blocks_by_index[stream_index] = block
                        if tool_call.get("id"):
                            tool_call_blocks_by_id[tool_call["id"]] = block
                        if not block.name and name:
                            block.name = name
                        return block

                    def finish_block(block) -> None:
                        content_index = blocks.index(block)
                        if isinstance(block, TextContent):
                            event_stream.push(TextEndEvent(content_index=content_index, content=block.text, partial=output))
                        elif isinstance(block, ThinkingContent):
                            event_stream.push(
                                ThinkingEndEvent(content_index=content_index, content=block.thinking, partial=output)
                            )
                        elif isinstance(block, ToolCall):
                            block.arguments = parse_streaming_json(block.partial_args)
                            block.partial_args = None
                            block.stream_index = None
                            event_stream.push(ToolCallEndEvent(content_index=content_index, tool_call=block, partial=output))

                    async for chunk in _iterate_openai_chunks(response, options.signal if options else None):
                        if options and options.on_provider_stream_event:
                            maybe = options.on_provider_stream_event(chunk, model)
                            if asyncio.iscoroutine(maybe):
                                await maybe
                        if not isinstance(chunk, dict):
                            continue

                        if chunk.get("id") and not output.response_id:
                            output.response_id = chunk["id"]
                        chunk_model = chunk.get("model")
                        if isinstance(chunk_model, str) and chunk_model and chunk_model != model.id and not output.response_model:
                            output.response_model = chunk_model
                        if chunk.get("usage"):
                            output.usage = _parse_chunk_usage(chunk["usage"], model)

                        choices = chunk.get("choices")
                        choice = choices[0] if isinstance(choices, list) and choices else None
                        if not choice:
                            continue

                        # Fallback: some providers (e.g. Moonshot) return usage on the choice.
                        if not chunk.get("usage") and choice.get("usage"):
                            output.usage = _parse_chunk_usage(choice["usage"], model)

                        if choice.get("finish_reason"):
                            output.raw_stop_reason = choice["finish_reason"]
                            stop_reason, error_message = _map_stop_reason(choice["finish_reason"])
                            output.stop_reason = stop_reason  # type: ignore[assignment]
                            if error_message:
                                output.error_message = error_message
                            has_finish_reason = True

                        delta = choice.get("delta")
                        if not delta:
                            continue

                        if delta.get("content"):
                            block = ensure_text_block()
                            block.text += delta["content"]
                            event_stream.push(
                                TextDeltaEvent(
                                    content_index=blocks.index(block), delta=delta["content"], partial=output
                                )
                            )

                        # Reasoning may arrive in reasoning_content (llama.cpp), reasoning
                        # (other OpenAI-compatible endpoints), or reasoning_text. Use the
                        # first non-empty field to avoid duplication.
                        for field in _REASONING_FIELDS:
                            value = delta.get(field)
                            if isinstance(value, str) and value:
                                block = ensure_thinking_block(field)
                                block.thinking += value
                                event_stream.push(
                                    ThinkingDeltaEvent(content_index=blocks.index(block), delta=value, partial=output)
                                )
                                break

                        if delta.get("tool_calls"):
                            for tool_call in delta["tool_calls"]:
                                block = ensure_tool_call_block(tool_call)
                                if not block.id and tool_call.get("id"):
                                    block.id = tool_call["id"]
                                    tool_call_blocks_by_id[tool_call["id"]] = block
                                function = tool_call.get("function") or {}
                                if not block.name and function.get("name"):
                                    block.name = function["name"]
                                delta_args = ""
                                if function.get("arguments"):
                                    delta_args = function["arguments"]
                                    block.partial_args = (block.partial_args or "") + delta_args
                                    block.arguments = parse_streaming_json(block.partial_args)
                                event_stream.push(
                                    ToolCallDeltaEvent(
                                        content_index=blocks.index(block), delta=delta_args, partial=output
                                    )
                                )

                    for block in list(blocks):
                        finish_block(block)

                    if options and options.signal and options.signal.aborted:
                        raise RuntimeError("Request was aborted")
                    if output.stop_reason == "aborted":
                        raise RuntimeError("Request was aborted")
                    if not has_finish_reason and not compat.supports_finish_reason:
                        output.stop_reason = "toolUse" if any(isinstance(b, ToolCall) for b in blocks) else "stop"
                    if output.stop_reason == "error":
                        raise RuntimeError(output.error_message or "Provider returned an error stop reason")
                    if (compat.supports_finish_reason and not has_finish_reason) or output.stop_reason == "pending":
                        raise RuntimeError("Stream ended without finish_reason")

                    event_stream.push(DoneEvent(reason=output.stop_reason, message=output))  # type: ignore[arg-type]
                    event_stream.end()
                finally:
                    await response.aclose()

        except Exception as error:
            for block in output.content:
                if isinstance(block, ToolCall):
                    block.partial_args = None
                    block.stream_index = None
            output.stop_reason = "aborted" if (options and options.signal and options.signal.aborted) else "error"
            output.error_message = str(error)
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

    merged = OpenAICompletionsOptions(**base.model_dump(exclude_none=True))
    merged.tool_choice = options.tool_choice if options else None
    merged.reasoning_effort = reasoning_effort
    merged.thinking_budgets = options.thinking_budgets if options else None
    return stream(model, context, merged)

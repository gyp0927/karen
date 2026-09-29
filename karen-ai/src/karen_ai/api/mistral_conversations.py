"""Mistral Conversations adapter, mirroring api/mistral-conversations.ts.

Native Mistral Chat Completions endpoint ({base}/v1/chat/completions) with
Mistral's content-chunk extensions (thinking chunks) and 9-char alphanumeric
tool call ID requirement.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterable, Dict, List, Optional, Union

import httpx

from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost, clamp_thinking_level
from ..transcript import get_current_tools, resolve_transcript
from ..types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ProviderHeaders,
    ProviderResponse,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    TranscriptContext,
    Usage,
)
from ..utils.error_body import safe_json_stringify, truncate_error_text
from ..utils.hash import short_hash
from ..utils.json_parse import parse_streaming_json
from ..utils.sanitize import sanitize_surrogates
from ..utils.sse import iterate_sse_messages
from ..utils.text import get_system_message_text, render_system_message_update
from .constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from .request_options import coerce_options
from .simple_options import build_base_options
from .transform_messages import transform_messages

KAREN_USER_AGENT = "karen-ai/0.1.0"

MISTRAL_TOOL_CALL_ID_LENGTH = 9
MAX_MISTRAL_ERROR_BODY_CHARS = 4000

MistralReasoningEffort = str  # "none" | "high"


class MistralOptions(StreamOptions):
    tool_choice: Optional[Any] = None  # "auto"|"none"|"any"|"required"|{"type":"function","function":{"name":...}}
    prompt_mode: Optional[str] = None  # "reasoning"
    reasoning_effort: Optional[MistralReasoningEffort] = None


# ---------------------------------------------------------------------------
# Tool call ID normalization (Mistral requires 9-char alphanumeric IDs)
# ---------------------------------------------------------------------------


def derive_mistral_tool_call_id(call_id: str, attempt: int) -> str:
    normalized = "".join(ch for ch in call_id if ch.isascii() and ch.isalnum())
    if attempt == 0 and len(normalized) == MISTRAL_TOOL_CALL_ID_LENGTH:
        return normalized
    seed_base = normalized or call_id
    seed = seed_base if attempt == 0 else f"{seed_base}:{attempt}"
    return "".join(ch for ch in short_hash(seed) if ch.isalnum())[:MISTRAL_TOOL_CALL_ID_LENGTH]


class MistralToolCallIdNormalizer:
    """Stable original→Mistral ID mapping with collision handling."""

    def __init__(self) -> None:
        self._id_map: Dict[str, str] = {}
        self._reverse_map: Dict[str, str] = {}

    def __call__(self, call_id: str, *_args) -> str:
        existing = self._id_map.get(call_id)
        if existing is not None:
            return existing
        attempt = 0
        while True:
            candidate = derive_mistral_tool_call_id(call_id, attempt)
            owner = self._reverse_map.get(candidate)
            if owner is None or owner == call_id:
                self._id_map[call_id] = candidate
                self._reverse_map[candidate] = call_id
                return candidate
            attempt += 1


# ---------------------------------------------------------------------------
# Payload building (camelCase payload, remapped to snake_case on the wire)
# ---------------------------------------------------------------------------


def _to_function_tools(tools: List[Tool]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        strict = resolve_json_schema_strict_sampling(tool, True)
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": get_json_schema_tool_parameters(tool, strict),
                    "strict": strict if strict is not None else False,
                },
            }
        )
    return converted


def _build_tool_result_text(text: str, has_images: bool, supports_images: bool, is_error: bool) -> str:
    trimmed = text.strip()
    error_prefix = "[tool error] " if is_error else ""

    if trimmed:
        image_suffix = "\n[tool image omitted: model does not support images]" if has_images and not supports_images else ""
        return f"{error_prefix}{trimmed}{image_suffix}"

    if has_images:
        if supports_images:
            return "[tool error] (see attached image)" if is_error else "(see attached image)"
        return (
            "[tool error] (image omitted: model does not support images)"
            if is_error
            else "(image omitted: model does not support images)"
        )

    return "[tool error] (no tool output)" if is_error else "(no tool output)"


def _to_chat_messages(messages: List, supports_images: bool) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []

    for index, msg in enumerate(messages):
        if msg.role == "system":
            text = get_system_message_text(msg) if index == 0 else render_system_message_update(msg)
            if text:
                result.append({"role": "system", "content": sanitize_surrogates(text)})
            continue

        if msg.role == "user":
            if isinstance(msg.content, str):
                result.append({"role": "user", "content": sanitize_surrogates(msg.content)})
                continue
            had_images = any(item.type == "image" for item in msg.content)
            content: List[Dict[str, Any]] = []
            for item in msg.content:
                if item.type == "text":
                    content.append({"type": "text", "text": sanitize_surrogates(item.text)})
                elif supports_images:
                    content.append(
                        {"type": "image_url", "imageUrl": f"data:{item.mime_type};base64,{item.data}"}
                    )
            if content:
                result.append({"role": "user", "content": content})
            elif had_images and not supports_images:
                result.append({"role": "user", "content": "(image omitted: model does not support images)"})
            continue

        if msg.role == "assistant":
            content_parts: List[Dict[str, Any]] = []
            tool_calls: List[Dict[str, Any]] = []
            for block in msg.content:
                if block.type == "text":
                    if block.text.strip():
                        content_parts.append({"type": "text", "text": sanitize_surrogates(block.text)})
                elif block.type == "thinking":
                    if block.thinking.strip():
                        content_parts.append(
                            {
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": sanitize_surrogates(block.thinking)}],
                            }
                        )
                else:
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {"name": block.name, "arguments": json.dumps(block.arguments or {})},
                            "index": 0,
                        }
                    )
            assistant_message: Dict[str, Any] = {"role": "assistant", "prefix": False}
            if content_parts:
                assistant_message["content"] = content_parts
            if tool_calls:
                assistant_message["toolCalls"] = tool_calls
            if content_parts or tool_calls:
                result.append(assistant_message)
            continue

        # toolResult
        tool_content: List[Dict[str, Any]] = []
        text_result = "\n".join(sanitize_surrogates(part.text) for part in msg.content if part.type == "text")
        has_images = any(part.type == "image" for part in msg.content)
        tool_text = _build_tool_result_text(text_result, has_images, supports_images, bool(msg.is_error))
        tool_content.append({"type": "text", "text": tool_text})
        if supports_images:
            for part in msg.content:
                if part.type == "image":
                    tool_content.append(
                        {"type": "image_url", "imageUrl": f"data:{part.mime_type};base64,{part.data}"}
                    )
        result.append(
            {
                "role": "tool",
                "toolCallId": msg.tool_call_id,
                "name": msg.tool_name,
                "content": tool_content,
            }
        )

    return result


def _should_use_prompt_caching(options: Optional[MistralOptions]) -> bool:
    return bool(
        options and options.cache_retention != "none" and options.session_id
    )


def _map_tool_choice(choice: Any) -> Any:
    if not choice:
        return None
    if choice in ("auto", "none", "any", "required"):
        return choice
    if isinstance(choice, dict) and choice.get("type") == "function":
        return {"type": "function", "function": {"name": choice["function"]["name"]}}
    return choice


def build_chat_payload(
    model: Model,
    context: TranscriptContext,
    messages: List,
    options: Optional[MistralOptions],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model.id,
        "stream": True,
        "messages": _to_chat_messages(messages, "image" in model.input),
    }

    current_tools = get_current_tools(context.messages)
    if current_tools:
        payload["tools"] = _to_function_tools(current_tools)
    if options and options.temperature is not None:
        payload["temperature"] = options.temperature
    if options and options.max_tokens is not None:
        payload["maxTokens"] = options.max_tokens
    if options and options.tool_choice:
        payload["toolChoice"] = _map_tool_choice(options.tool_choice)
    if options and options.prompt_mode:
        payload["promptMode"] = options.prompt_mode
    if options and options.reasoning_effort:
        payload["reasoningEffort"] = options.reasoning_effort
    if _should_use_prompt_caching(options):
        payload["promptCacheKey"] = options.session_id

    return payload


# ---------------------------------------------------------------------------
# Wire format remapping (camelCase → snake_case)
# ---------------------------------------------------------------------------

_WIRE_KEY_MAP = (
    ("topP", "top_p"),
    ("maxTokens", "max_tokens"),
    ("randomSeed", "random_seed"),
    ("responseFormat", "response_format"),
    ("toolChoice", "tool_choice"),
    ("presencePenalty", "presence_penalty"),
    ("frequencyPenalty", "frequency_penalty"),
    ("parallelToolCalls", "parallel_tool_calls"),
    ("reasoningEffort", "reasoning_effort"),
    ("promptMode", "prompt_mode"),
    ("promptCacheKey", "prompt_cache_key"),
    ("safePrompt", "safe_prompt"),
)

_CHUNK_KEY_MAP = (
    ("imageUrl", "image_url"),
    ("documentUrl", "document_url"),
    ("documentName", "document_name"),
    ("fileId", "file_id"),
    ("referenceIds", "reference_ids"),
    ("inputAudio", "input_audio"),
)


def _remap(record: Dict[str, Any], source: str, target: str) -> None:
    if source not in record:
        return
    record[target] = record.pop(source)


def _to_wire_message(message: Dict[str, Any]) -> Dict[str, Any]:
    wire = dict(message)
    _remap(wire, "toolCalls", "tool_calls")
    _remap(wire, "toolCallId", "tool_call_id")
    content = message.get("content")
    if isinstance(content, list):
        wire_content = []
        for chunk in content:
            wire_chunk = dict(chunk)
            for source, target in _CHUNK_KEY_MAP:
                _remap(wire_chunk, source, target)
            wire_content.append(wire_chunk)
        wire["content"] = wire_content
    return wire


def to_mistral_wire_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    wire: Dict[str, Any] = dict(payload)
    for source, target in _WIRE_KEY_MAP:
        _remap(wire, source, target)
    wire["messages"] = [_to_wire_message(message) for message in payload["messages"]]

    response_format = wire.get("response_format")
    if isinstance(response_format, dict):
        wire_response_format = dict(response_format)
        _remap(wire_response_format, "jsonSchema", "json_schema")
        json_schema = wire_response_format.get("json_schema")
        if isinstance(json_schema, dict):
            wire_json_schema = dict(json_schema)
            _remap(wire_json_schema, "schemaDefinition", "schema")
            wire_response_format["json_schema"] = wire_json_schema
        wire["response_format"] = wire_response_format

    return wire


# ---------------------------------------------------------------------------
# Headers / errors
# ---------------------------------------------------------------------------


def _has_header_override(overrides: Optional[ProviderHeaders], target: str) -> bool:
    return bool(overrides) and any(name.lower() == target for name in overrides)


def _build_headers(model: Model, api_key: str, options: Optional[MistralOptions]) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "User-Agent": KAREN_USER_AGENT,
        "accept": "text/event-stream",
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    for overrides in (model.headers, options.headers if options else None):
        for name, value in (overrides or {}).items():
            if value is None:
                for existing in list(headers.keys()):
                    if existing.lower() == name.lower():
                        del headers[existing]
            else:
                headers[name] = value

    has_explicit_affinity = _has_header_override(model.headers, "x-affinity") or _has_header_override(
        options.headers if options else None, "x-affinity"
    )
    if _should_use_prompt_caching(options) and not has_explicit_affinity:
        headers["x-affinity"] = options.session_id
    return headers


def _format_mistral_error(error: Any) -> str:
    if isinstance(error, Exception):
        status_code = getattr(error, "status", None) or getattr(error, "status_code", None)
        if not isinstance(status_code, int):
            status_code = None
        body_text = getattr(error, "body", None)
        body_text = body_text.strip() if isinstance(body_text, str) else None
        message = str(error)
        if status_code is not None and body_text:
            return f"Mistral API error ({status_code}): {truncate_error_text(body_text, MAX_MISTRAL_ERROR_BODY_CHARS)}"
        if status_code is not None:
            return f"Mistral API error ({status_code}): {message}"
        return message or type(error).__name__
    return safe_json_stringify(error)


class MistralHttpError(Exception):
    def __init__(self, status_code: int, body: str, status_text: str):
        super().__init__(status_text or f"Request failed with status {status_code}")
        self.status_code = status_code
        self.body = body


# ---------------------------------------------------------------------------
# Stream reading
# ---------------------------------------------------------------------------


async def _read_mistral_events(response: httpx.Response, signal=None) -> AsyncIterable[Dict[str, Any]]:
    async for sse in iterate_sse_messages(response.aiter_text(), signal):
        data = sse.data.strip()
        if not data:
            continue
        if data == "[DONE]":
            return
        parsed = json.loads(data)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("choices"), list):
            raise RuntimeError("Invalid Mistral streaming event")
        yield parsed


def _get_cached_prompt_tokens(usage: Dict[str, Any], prompt_tokens: int) -> int:
    raw_cached = 0
    for container_key, nested_key in (
        ("promptTokensDetails", "cachedTokens"),
        ("prompt_tokens_details", "cached_tokens"),
        ("promptTokenDetails", "cachedTokens"),
        ("prompt_token_details", "cached_tokens"),
    ):
        container = usage.get(container_key)
        if isinstance(container, dict) and isinstance(container.get(nested_key), (int, float)):
            raw_cached = container[nested_key]
            break
    else:
        for flat_key in ("numCachedTokens", "num_cached_tokens"):
            if isinstance(usage.get(flat_key), (int, float)):
                raw_cached = usage[flat_key]
                break
    cached_tokens = raw_cached if isinstance(raw_cached, (int, float)) else 0
    return min(prompt_tokens, max(0, int(cached_tokens)))


def _map_chat_stop_reason(reason: Optional[str]) -> tuple[str, Optional[str]]:
    if reason is None or reason == "stop":
        return "stop", None
    if reason in ("length", "model_length"):
        return "length", None
    if reason == "tool_calls":
        return "toolUse", None
    if reason == "error":
        return "error", "Provider stopped with: error"
    return "error", f"Provider stopped with: {reason}"


async def _consume_chat_stream(
    model: Model,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    mistral_stream: AsyncIterable[Dict[str, Any]],
    on_provider_stream_event=None,
) -> None:
    from ..types import (
        TextDeltaEvent,
        TextEndEvent,
        TextStartEvent,
        ThinkingDeltaEvent,
        ThinkingEndEvent,
        ThinkingStartEvent,
        ToolCallDeltaEvent,
        ToolCallEndEvent,
        ToolCallStartEvent,
    )

    current_block: Optional[Union[TextContent, ThinkingContent]] = None
    blocks = output.content
    tool_blocks_by_key: Dict[Any, int] = {}

    def block_index() -> int:
        return len(blocks) - 1

    def finish_current_block() -> None:
        nonlocal current_block
        if current_block is None:
            return
        if current_block.type == "text":
            stream.push(TextEndEvent(content_index=block_index(), content=current_block.text, partial=output))
        elif current_block.type == "thinking":
            stream.push(
                ThinkingEndEvent(content_index=block_index(), content=current_block.thinking, partial=output)
            )
        current_block = None

    async for chunk in mistral_stream:
        if on_provider_stream_event:
            maybe = on_provider_stream_event(chunk, model)
            if asyncio.iscoroutine(maybe):
                await maybe
        # Mistral's streamed CompletionChunk carries an id; keep the first one.
        if not output.response_id and chunk.get("id"):
            output.response_id = chunk["id"]

        usage = chunk.get("usage")
        if usage:
            prompt_tokens = usage.get("prompt_tokens") or 0
            cached_prompt_tokens = _get_cached_prompt_tokens(usage, prompt_tokens)
            output.usage.input = max(0, prompt_tokens - cached_prompt_tokens)
            output.usage.output = usage.get("completion_tokens") or 0
            output.usage.cache_read = cached_prompt_tokens
            output.usage.cache_write = 0
            output.usage.total_tokens = usage.get("total_tokens") or (
                output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write
            )
            calculate_cost(model, output.usage)

        choice = chunk["choices"][0] if chunk["choices"] else None
        if not choice:
            continue

        if choice.get("finish_reason"):
            output.raw_stop_reason = choice["finish_reason"]
            stop_reason, error_message = _map_chat_stop_reason(choice["finish_reason"])
            output.stop_reason = stop_reason  # type: ignore[assignment]
            if error_message:
                output.error_message = error_message

        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content is not None:
            content_items = [content] if isinstance(content, str) else content
            for item in content_items:
                if isinstance(item, str):
                    text_delta = sanitize_surrogates(item)
                    # GLM models on Mistral send empty content deltas around
                    # thinking and tool calls; opening blocks for them splits
                    # thinking, which Mistral rejects on replay.
                    if not text_delta:
                        continue
                    if current_block is None or current_block.type != "text":
                        finish_current_block()
                        current_block = TextContent(text="")
                        blocks.append(current_block)
                        stream.push(TextStartEvent(content_index=block_index(), partial=output))
                    current_block.text += text_delta
                    stream.push(TextDeltaEvent(content_index=block_index(), delta=text_delta, partial=output))
                    continue

                if item.get("type") == "thinking":
                    thinking_delta = sanitize_surrogates(
                        "".join(
                            part.get("text") or "" for part in (item.get("thinking") or []) if part.get("text")
                        )
                    )
                    if not thinking_delta:
                        continue
                    if current_block is None or current_block.type != "thinking":
                        finish_current_block()
                        current_block = ThinkingContent(thinking="")
                        blocks.append(current_block)
                        stream.push(ThinkingStartEvent(content_index=block_index(), partial=output))
                    current_block.thinking += thinking_delta
                    stream.push(ThinkingDeltaEvent(content_index=block_index(), delta=thinking_delta, partial=output))
                    continue

                if item.get("type") == "text":
                    text_delta = sanitize_surrogates(item.get("text") or "")
                    if not text_delta:
                        continue
                    if current_block is None or current_block.type != "text":
                        finish_current_block()
                        current_block = TextContent(text="")
                        blocks.append(current_block)
                        stream.push(TextStartEvent(content_index=block_index(), partial=output))
                    current_block.text += text_delta
                    stream.push(TextDeltaEvent(content_index=block_index(), delta=text_delta, partial=output))

        for tool_call in delta.get("tool_calls") or []:
            finish_current_block()
            raw_id = tool_call.get("id")
            call_id = (
                raw_id
                if raw_id and raw_id != "null"
                else derive_mistral_tool_call_id(f"toolcall:{tool_call.get('index') or 0}", 0)
            )
            key = tool_call.get("index") if tool_call.get("index") is not None else call_id
            block: Optional[ToolCall] = None
            existing_index = tool_blocks_by_key.get(key)
            if existing_index is not None and blocks[existing_index].type == "toolCall":
                block = blocks[existing_index]

            if block is None:
                block = ToolCall(
                    id=call_id,
                    name=(tool_call.get("function") or {}).get("name") or "",
                    arguments={},
                )
                block.partial_args = ""
                blocks.append(block)
                tool_blocks_by_key[key] = len(blocks) - 1
                stream.push(ToolCallStartEvent(content_index=len(blocks) - 1, partial=output))

            function = tool_call.get("function") or {}
            raw_arguments = function.get("arguments")
            args_delta = (
                raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments or {})
            )
            block.partial_args = (block.partial_args or "") + args_delta
            block.arguments = parse_streaming_json(block.partial_args)
            stream.push(
                ToolCallDeltaEvent(content_index=tool_blocks_by_key[key], delta=args_delta, partial=output)
            )

    finish_current_block()
    for index in tool_blocks_by_key.values():
        block = blocks[index]
        if block.type != "toolCall":
            continue
        block.arguments = parse_streaming_json(block.partial_args)
        # Finalize in-place and strip the scratch buffer so replay only
        # carries parsed arguments.
        block.partial_args = None
        stream.push(ToolCallEndEvent(content_index=index, tool_call=block, partial=output))


# ---------------------------------------------------------------------------
# Reasoning helpers
# ---------------------------------------------------------------------------


def _uses_reasoning_effort(model: Model) -> bool:
    return (
        model.id == "mistral-small-2603"
        or model.id == "mistral-small-latest"
        or model.id.startswith("mistral-medium-")
        or model.id == "zai-glm-5-2"
    )


def _uses_prompt_mode_reasoning(model: Model) -> bool:
    return model.reasoning and not _uses_reasoning_effort(model)


def _map_reasoning_effort(model: Model, level: str) -> MistralReasoningEffort:
    return (model.thinking_level_map or {}).get(level) or "high"


# ---------------------------------------------------------------------------
# stream / stream_simple
# ---------------------------------------------------------------------------


def stream(
    model: Model,
    context: TranscriptContext,
    options: Optional[MistralOptions] = None,
) -> AssistantMessageEventStream:
    options = coerce_options(options, MistralOptions)
    event_stream = AssistantMessageEventStream()
    compat_supports_mid_convo = getattr(model.compat, "supports_mid_convo_system_messages", None)
    normalized_context = resolve_transcript(context, compat_supports_mid_convo is True)

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
            if not api_key:
                raise ValueError(f"No API key for provider: {model.provider}")

            normalize_id = MistralToolCallIdNormalizer()
            transformed_messages = transform_messages(normalized_context.messages, model, normalize_id)

            payload = build_chat_payload(model, normalized_context, transformed_messages, options)
            if options and options.on_payload:
                next_payload = options.on_payload(payload, model)
                if asyncio.iscoroutine(next_payload):
                    next_payload = await next_payload
                if next_payload is not None:
                    payload = next_payload

            headers = _build_headers(model, api_key, options)
            wire_payload = to_mistral_wire_payload(payload)

            # new URL("v1/chat/completions", baseUrl + "/") semantics.
            base = model.base_url.rstrip("/") + "/"
            url = f"{base}v1/chat/completions"

            timeout = httpx.Timeout((options.timeout_ms or 60_000) / 1000, connect=60.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                # pi-ai's Mistral adapter performs a single request (no provider
                # retry policy), so options.max_retries does not apply here.
                request = client.build_request("POST", url, json=wire_payload, headers=headers)
                response = await client.send(request, stream=True)
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    await response.aclose()
                    raise MistralHttpError(
                        response.status_code, body, httpx.Response(response.status_code).reason_phrase
                    )

                try:
                    if options and options.on_response:
                        maybe = options.on_response(
                            ProviderResponse(status=response.status_code, headers=dict(response.headers)), model
                        )
                        if asyncio.iscoroutine(maybe):
                            await maybe

                    event_stream.push(StartEvent(partial=output))

                    await _consume_chat_stream(
                        model,
                        output,
                        event_stream,
                        _read_mistral_events(response, options.signal if options else None),
                        options.on_provider_stream_event if options else None,
                    )

                    if options and options.signal and options.signal.aborted:
                        raise RuntimeError("Request was aborted")
                    if output.stop_reason == "pending":
                        raise RuntimeError("Mistral stream ended without a finish reason")
                    if output.stop_reason in ("aborted", "error"):
                        raise RuntimeError(output.error_message or "An unknown error occurred")

                    event_stream.push(DoneEvent(reason=output.stop_reason, message=output))  # type: ignore[arg-type]
                    event_stream.end()
                finally:
                    await response.aclose()

        except Exception as error:
            for block in output.content:
                if isinstance(block, ToolCall):
                    # partial_args is only a streaming scratch buffer; never persist it.
                    block.partial_args = None
            output.stop_reason = "aborted" if (options and options.signal and options.signal.aborted) else "error"
            output.error_message = _format_mistral_error(error)
            event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))  # type: ignore[arg-type]
            event_stream.end()

    asyncio.get_running_loop().create_task(run())
    return event_stream


def stream_simple(
    model: Model,
    context: TranscriptContext,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    api_key = options.api_key if options else None
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, context, options, api_key)
    clamped_reasoning = clamp_thinking_level(model, options.reasoning) if options and options.reasoning else None
    reasoning = None if clamped_reasoning == "off" else clamped_reasoning
    should_use_reasoning = model.reasoning and reasoning is not None

    merged = MistralOptions(**base.model_dump(exclude_none=True))
    merged.tool_choice = options.tool_choice if options else None
    merged.prompt_mode = "reasoning" if should_use_reasoning and _uses_prompt_mode_reasoning(model) else None
    merged.reasoning_effort = (
        _map_reasoning_effort(model, reasoning) if should_use_reasoning and _uses_reasoning_effort(model) else None
    )
    return stream(model, context, merged)

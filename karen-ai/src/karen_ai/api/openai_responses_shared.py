"""Shared OpenAI Responses API machinery, mirroring api/openai-responses-shared.ts.

Used by the openai-responses adapter and its Azure / Codex variants:
message + tool conversion to the Responses wire format and processing of the
Responses SSE event stream into the karen-ai event protocol.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, Awaitable, Callable, Dict, List, Optional, Set, Union

from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost
from ..transcript import resolve_transcript, resolve_transcript_tools
from ..types import (
    AssistantMessage,
    ImageContent,
    Model,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    TranscriptContext,
    Usage,
)
from ..utils.hash import short_hash
from ..utils.json_parse import parse_streaming_json
from ..utils.sanitize import sanitize_surrogates
from ..utils.text import get_system_message_text, render_system_message_update
from .constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
    get_grammar_tool_input,
    get_json_schema_tool_parameters,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from .transform_messages import transform_messages

# ---------------------------------------------------------------------------
# Text signatures
# ---------------------------------------------------------------------------


def encode_text_signature_v1(message_id: str, phase: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {"v": 1, "id": message_id}
    if phase:
        payload["phase"] = phase
    return json.dumps(payload, separators=(",", ":"))


def parse_text_signature(signature: Optional[str]) -> Optional[Dict[str, Any]]:
    """Returns {"id": str, "phase"?: str} or None."""
    if not signature:
        return None
    if signature.startswith("{"):
        try:
            parsed = json.loads(signature)
            if isinstance(parsed, dict) and parsed.get("v") == 1 and isinstance(parsed.get("id"), str):
                result: Dict[str, Any] = {"id": parsed["id"]}
                if parsed.get("phase") in ("commentary", "final_answer"):
                    result["phase"] = parsed["phase"]
                return result
        except (json.JSONDecodeError, ValueError):
            pass  # Fall through to legacy plain-string handling.
    return {"id": signature}


# ---------------------------------------------------------------------------
# Tool result output
# ---------------------------------------------------------------------------


def convert_tool_result_output(
    model: Model,
    content: List[Union[TextContent, ImageContent]],
) -> Union[str, List[Dict[str, Any]]]:
    text_result = "\n".join(c.text for c in content if c.type == "text")
    images = [c for c in content if c.type == "image"]
    has_text = len(text_result) > 0

    if not images or "image" not in model.input:
        if has_text:
            return sanitize_surrogates(text_result)
        return "(see attached image)" if images else "(no tool output)"

    output: List[Dict[str, Any]] = []
    if has_text:
        output.append({"type": "input_text", "text": sanitize_surrogates(text_result)})
    for image in images:
        output.append(
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:{image.mime_type};base64,{image.data}",
            }
        )
    return output


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass
class ConvertResponsesToolsOptions:
    strict: Optional[bool] = None
    supports_strict_mode: bool = True
    supports_openai_grammar_tools: bool = False
    tool_search_result: bool = False


@dataclass
class ConvertResponsesMessagesOptions:
    include_system_prompt: bool = True
    grammar_tool_input_properties: Optional[Dict[str, str]] = None
    #: Whether later system messages are sent in place; otherwise they are
    #: folded into the leading prompt.
    supports_mid_convo_system_messages: bool = False
    supports_additional_tools: bool = False
    supports_tool_search: bool = False
    tool_options: Optional[ConvertResponsesToolsOptions] = None


@dataclass
class OpenAIResponsesStreamOptions:
    on_provider_stream_event: Optional[Callable[[Any, Model], Any]] = None
    service_tier: Optional[str] = None
    grammar_tool_input_properties: Optional[Dict[str, str]] = None
    resolve_service_tier: Optional[Callable[[Optional[str], Optional[str]], Optional[str]]] = None
    apply_service_tier_pricing: Optional[Callable[[Usage, Optional[str]], None]] = None


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def _normalize_id_part(part: str) -> str:
    sanitized = "".join(ch if (ch.isascii() and (ch.isalnum() or ch in "-_")) else "_" for ch in part)
    normalized = sanitized[:64] if len(sanitized) > 64 else sanitized
    return normalized.rstrip("_")


def _build_foreign_responses_item_id(item_id: str) -> str:
    normalized = f"fc_{short_hash(item_id)}"
    return normalized[:64] if len(normalized) > 64 else normalized


def convert_responses_messages(
    model: Model,
    context: TranscriptContext,
    allowed_tool_call_providers: Set[str],
    options: Optional[ConvertResponsesMessagesOptions] = None,
) -> List[Dict[str, Any]]:
    options = options or ConvertResponsesMessagesOptions()
    normalized_context = resolve_transcript(context, options.supports_mid_convo_system_messages)
    messages: List[Dict[str, Any]] = []

    def normalize_tool_call_id(call_id: str, _target_model: Model, source: AssistantMessage) -> str:
        if model.provider not in allowed_tool_call_providers:
            return _normalize_id_part(call_id)
        if "|" not in call_id:
            return _normalize_id_part(call_id)
        raw_call_id, _, item_id = call_id.partition("|")
        normalized_call_id = _normalize_id_part(raw_call_id)
        is_foreign_tool_call = source.provider != model.provider or source.api != model.api
        normalized_item_id = (
            _build_foreign_responses_item_id(item_id) if is_foreign_tool_call else _normalize_id_part(item_id)
        )
        # OpenAI Responses API requires item ids to start with "fc".
        if not normalized_item_id.startswith("fc_"):
            normalized_item_id = _normalize_id_part(f"fc_{normalized_item_id}")
        return f"{normalized_call_id}|{normalized_item_id}"

    transformed_messages = transform_messages(normalized_context.messages, model, normalize_tool_call_id)
    transcript_tools = resolve_transcript_tools(
        normalized_context.messages,
        options.supports_additional_tools or options.supports_tool_search,
    )

    def append_system_tool_additions(message, seed: str) -> None:
        tools = (message.tools_added or []) if transcript_tools.anchors_additions else []
        if not tools:
            return
        if options.supports_additional_tools:
            messages.append(
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": convert_responses_tools(tools, options.tool_options),
                }
            )
            return
        if not options.supports_tool_search:
            return
        names = [tool.name for tool in tools]
        names_joined = ",".join(names)
        call_id = f"pi_tool_load_{short_hash(f'{seed}:{names_joined}')}"
        messages.append(
            {
                "type": "tool_search_call",
                "call_id": call_id,
                "execution": "client",
                "status": "completed",
                "arguments": {"query": " ".join(names), "limit": len(names)},
            }
        )
        messages.append(
            {
                "type": "tool_search_output",
                "call_id": call_id,
                "execution": "client",
                "status": "completed",
                "tools": convert_responses_tools(
                    tools,
                    ConvertResponsesToolsOptions(
                        strict=options.tool_options.strict if options.tool_options else None,
                        supports_strict_mode=options.tool_options.supports_strict_mode
                        if options.tool_options
                        else True,
                        supports_openai_grammar_tools=options.tool_options.supports_openai_grammar_tools
                        if options.tool_options
                        else False,
                        tool_search_result=True,
                    ),
                ),
            }
        )

    include_initial_system_message = options.include_system_prompt
    compat_supports_developer = getattr(model.compat, "supports_developer_role", None)
    instruction_role = "developer" if (model.reasoning and compat_supports_developer is not False) else "system"

    msg_index = 0
    source_index = 0
    for msg in transformed_messages:
        is_leading_system_message = source_index == 0 and msg.role == "system"
        source_index += 1

        if msg.role == "system":
            if not is_leading_system_message:
                append_system_tool_additions(msg, f"system:{msg_index}")
            if not is_leading_system_message or include_initial_system_message:
                text = get_system_message_text(msg) if is_leading_system_message else render_system_message_update(msg)
                if text:
                    messages.append({"role": instruction_role, "content": sanitize_surrogates(text)})

        elif msg.role == "user":
            if isinstance(msg.content, str):
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": sanitize_surrogates(msg.content)}],
                    }
                )
            else:
                content: List[Dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        content.append({"type": "input_text", "text": sanitize_surrogates(item.text)})
                    else:
                        content.append(
                            {
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": f"data:{item.mime_type};base64,{item.data}",
                            }
                        )
                if not content:
                    continue
                messages.append({"role": "user", "content": content})

        elif msg.role == "assistant":
            output: List[Dict[str, Any]] = []
            assistant_msg: AssistantMessage = msg  # type: ignore[assignment]
            is_same_provider_and_api = assistant_msg.provider == model.provider and assistant_msg.api == model.api
            is_same_model = is_same_provider_and_api and assistant_msg.model == model.id
            is_different_model = is_same_provider_and_api and assistant_msg.model != model.id
            text_block_index = 0

            for block in assistant_msg.content:
                if block.type == "thinking":
                    if block.thinking_signature:
                        try:
                            reasoning_item = json.loads(block.thinking_signature)
                            if isinstance(reasoning_item, dict):
                                output.append(reasoning_item)
                        except (json.JSONDecodeError, ValueError):
                            pass

                elif block.type == "text":
                    text_block: TextContent = block  # type: ignore[assignment]
                    parsed_signature = parse_text_signature(text_block.text_signature)
                    fallback_message_id = (
                        f"msg_pi_{msg_index}" if text_block_index == 0 else f"msg_pi_{msg_index}_{text_block_index}"
                    )
                    text_block_index += 1
                    # OpenAI requires ids to be max 64 characters.
                    msg_id = parsed_signature["id"] if parsed_signature else fallback_message_id
                    if len(msg_id) > 64:
                        msg_id = f"msg_{short_hash(msg_id)}"
                    message_item: Dict[str, Any] = {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": sanitize_surrogates(text_block.text), "annotations": []}
                        ],
                        "status": "completed",
                        "id": msg_id,
                    }
                    if parsed_signature and parsed_signature.get("phase"):
                        message_item["phase"] = parsed_signature["phase"]
                    output.append(message_item)

                elif block.type == "toolCall":
                    tool_call: ToolCall = block  # type: ignore[assignment]
                    call_id, _, item_id_raw = tool_call.id.partition("|")
                    grammar_props = options.grammar_tool_input_properties or {}
                    custom_input_property = grammar_props.get(tool_call.name)
                    item_id: Optional[str] = item_id_raw or None

                    # For different-model messages, drop the item id to avoid
                    # OpenAI's fc_xxx/rs_xxx pairing validation. Custom-tool calls
                    # replayed as function_call also drop non-fc_* ids such as
                    # ctc_* custom-tool ids.
                    if (is_different_model and item_id and item_id.startswith("fc_")) or (
                        custom_input_property is None and not (item_id or "").startswith("fc_")
                    ):
                        item_id = None

                    include_namespace = is_same_model and tool_call.namespace is not None
                    if custom_input_property is not None:
                        item: Dict[str, Any] = {
                            "type": "custom_tool_call",
                            "call_id": call_id,
                            "name": tool_call.name,
                            "input": sanitize_surrogates(
                                get_grammar_tool_input(tool_call.name, tool_call.arguments, custom_input_property)
                            ),
                        }
                        if item_id is not None:
                            item["id"] = item_id
                        if include_namespace:
                            item["namespace"] = tool_call.namespace
                        output.append(item)
                    else:
                        item = {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        }
                        if item_id is not None:
                            item["id"] = item_id
                        if include_namespace:
                            item["namespace"] = tool_call.namespace
                        output.append(item)

            if not output:
                continue
            messages.extend(output)

        elif msg.role == "toolResult":
            call_id = msg.tool_call_id.split("|")[0]
            result_output = convert_tool_result_output(model, msg.content)
            grammar_props = options.grammar_tool_input_properties or {}
            if msg.tool_name in grammar_props:
                messages.append(
                    {
                        "type": "custom_tool_call_output",
                        "call_id": call_id,
                        "output": result_output,
                    }
                )
            else:
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result_output,
                    }
                )

        if not is_leading_system_message:
            msg_index += 1

    return messages


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


def convert_responses_tools(
    tools: List[Tool],
    options: Optional[ConvertResponsesToolsOptions] = None,
) -> List[Dict[str, Any]]:
    default_strict = False if options is None or options.strict is None else options.strict
    supports_strict_mode = True if options is None else options.supports_strict_mode
    supports_grammar = False if options is None else options.supports_openai_grammar_tools
    tool_search_result = False if options is None else options.tool_search_result

    converted: List[Dict[str, Any]] = []
    for tool in tools:
        grammar = resolve_grammar_constrained_sampling(tool, supports_grammar)
        if grammar:
            grammar_tool: Dict[str, Any] = {
                "type": "custom",
                "name": tool.name,
                "description": tool.description,
                "format": {
                    "type": "grammar",
                    "syntax": grammar.format,
                    "definition": grammar.definition,
                },
            }
            if tool_search_result:
                grammar_tool["defer_loading"] = True
            converted.append(grammar_tool)
            continue

        constrained_strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        strict = constrained_strict if constrained_strict is not None else default_strict
        function_tool: Dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": get_json_schema_tool_parameters(tool, strict is True),
        }
        if tool_search_result:
            function_tool["defer_loading"] = True
        if supports_strict_mode:
            function_tool["strict"] = strict
        converted.append(function_tool)
    return converted


# ---------------------------------------------------------------------------
# Stream processing
# ---------------------------------------------------------------------------


@dataclass
class _OutputSlot:
    type: str  # "thinking" | "text" | "toolCall"
    block: Any
    content_index: int
    #: Scratch JSON buffer for streamed function_call arguments.
    partial_json: Optional[str] = None
    #: Scratch state for streamed custom_tool_call input.
    custom_property: Optional[str] = None
    custom_buffer: Optional[GrammarToolInputJsonBuffer] = None


async def process_responses_stream(
    openai_stream: AsyncIterable[Dict[str, Any]],
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    options: Optional[OpenAIResponsesStreamOptions] = None,
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

    options = options or OpenAIResponsesStreamOptions()
    saw_terminal_response_event = False
    output_slots: Dict[int, _OutputSlot] = {}
    reasoning_blocks_by_id: Dict[str, ThinkingContent] = {}

    def apply_message_phase_stop_reason(item: Dict[str, Any]) -> None:
        if item.get("type") == "message" and item.get("phase") == "final_answer":
            output.stop_reason = "stop"

    def get_slot(output_index: int, slot_type: str) -> Optional[_OutputSlot]:
        slot = output_slots.get(output_index)
        return slot if slot is not None and slot.type == slot_type else None

    def push_tool_call_delta(slot: _OutputSlot, delta: Optional[str]) -> None:
        if delta is None:
            return
        stream.push(
            ToolCallDeltaEvent(content_index=slot.content_index, delta=delta, partial=output)
        )

    def get_custom_tool_call_input(slot: _OutputSlot) -> str:
        if slot.custom_property is None:
            return ""
        value = slot.block.arguments.get(slot.custom_property)
        return value if isinstance(value, str) else ""

    def append_custom_tool_call_input(slot: _OutputSlot, next_input: str, close: bool) -> Optional[str]:
        if slot.custom_property is None or slot.custom_buffer is None:
            return None
        delta = append_grammar_tool_input_json_delta(slot.custom_buffer, slot.custom_property, next_input, close)
        slot.block.arguments = {slot.custom_property: next_input}
        return delta

    def create_slot(output_index: int, item: Dict[str, Any]) -> Optional[_OutputSlot]:
        item_type = item.get("type")
        if item_type == "reasoning":
            block = ThinkingContent(thinking="")
            output.content.append(block)
            slot = _OutputSlot("thinking", block, len(output.content) - 1)
            output_slots[output_index] = slot
            stream.push(ThinkingStartEvent(content_index=slot.content_index, partial=output))
            return slot
        if item_type == "message":
            apply_message_phase_stop_reason(item)
            block = TextContent(text="")
            output.content.append(block)
            slot = _OutputSlot("text", block, len(output.content) - 1)
            output_slots[output_index] = slot
            stream.push(TextStartEvent(content_index=slot.content_index, partial=output))
            return slot
        if item_type == "function_call":
            block = ToolCall(
                id=f"{item.get('call_id', '')}|{item.get('id', '')}",
                name=item.get("name") or "",
                arguments={},
            )
            if item.get("namespace") is not None:
                block.namespace = item["namespace"]
            output.content.append(block)
            slot = _OutputSlot(
                "toolCall", block, len(output.content) - 1, partial_json=item.get("arguments") or ""
            )
            output_slots[output_index] = slot
            stream.push(ToolCallStartEvent(content_index=slot.content_index, partial=output))
            return slot
        if item_type == "custom_tool_call":
            grammar_props = options.grammar_tool_input_properties or {}
            input_property = grammar_props.get(item.get("name") or "", "input")
            raw_input = item.get("input") or ""
            block = ToolCall(
                id=f"{item.get('call_id', '')}|{item.get('id', '')}",
                name=item.get("name") or "",
                arguments={input_property: raw_input},
            )
            if item.get("namespace") is not None:
                block.namespace = item["namespace"]
            output.content.append(block)
            slot = _OutputSlot(
                "toolCall",
                block,
                len(output.content) - 1,
                custom_property=input_property,
                custom_buffer=GrammarToolInputJsonBuffer(),
            )
            output_slots[output_index] = slot
            stream.push(ToolCallStartEvent(content_index=slot.content_index, partial=output))
            return slot
        return None

    def get_or_create_slot(output_index: int, item: Dict[str, Any]) -> Optional[_OutputSlot]:
        return output_slots.get(output_index) or create_slot(output_index, item)

    # Azure OpenAI can omit reasoning.encrypted_content from
    # response.output_item.done and provide it only in
    # response.completed.response.output. Backfill the persisted reasoning
    # signature from the terminal response to keep store:false multi-turn
    # replay stateless. See https://github.com/earendil-works/pi/issues/6409.
    def backfill_reasoning_signatures(response_output: List[Dict[str, Any]]) -> None:
        for item in response_output:
            if item.get("type") != "reasoning" or not item.get("encrypted_content"):
                continue
            block = reasoning_blocks_by_id.get(item.get("id", ""))
            if block is None or not block.thinking_signature:
                continue
            try:
                stored_item = json.loads(block.thinking_signature)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(stored_item, dict) or stored_item.get("encrypted_content"):
                continue
            stored_item["encrypted_content"] = item["encrypted_content"]
            block.thinking_signature = json.dumps(stored_item, separators=(",", ":"))

    def finalize_response(response: Dict[str, Any]) -> None:
        nonlocal saw_terminal_response_event
        saw_terminal_response_event = True
        backfill_reasoning_signatures(response.get("output") or [])
        if response.get("id"):
            output.response_id = response["id"]
        raw_usage = response.get("usage")
        if raw_usage:
            input_details = raw_usage.get("input_tokens_details") or {}
            cached_tokens = input_details.get("cached_tokens") or 0
            cache_write_tokens = input_details.get("cache_write_tokens") or 0
            output_details = raw_usage.get("output_tokens_details") or {}
            output.usage = Usage(
                # OpenAI includes cached and cache-write tokens in input_tokens.
                input=max(0, (raw_usage.get("input_tokens") or 0) - cached_tokens - cache_write_tokens),
                output=raw_usage.get("output_tokens") or 0,
                cache_read=cached_tokens,
                cache_write=cache_write_tokens,
                reasoning=output_details.get("reasoning_tokens") or 0,
                total_tokens=raw_usage.get("total_tokens") or 0,
            )
        calculate_cost(model, output.usage)
        if options.apply_service_tier_pricing:
            service_tier = (
                options.resolve_service_tier(response.get("service_tier"), options.service_tier)
                if options.resolve_service_tier
                else (response.get("service_tier") or options.service_tier)
            )
            options.apply_service_tier_pricing(output.usage, service_tier)
        # Map status to stop reason. For incomplete responses, retain the
        # provider's specific reason so max-output truncation and content
        # filtering stay distinct.
        status = response.get("status")
        incomplete_details = response.get("incomplete_details") or {}
        incomplete_reason = incomplete_details.get("reason")
        if not isinstance(incomplete_reason, str):
            incomplete_reason = None
        output.raw_stop_reason = f"{status}.{incomplete_reason}" if incomplete_reason else status
        mapped_stop, error_message = map_stop_reason(status, incomplete_reason)
        output.stop_reason = mapped_stop  # type: ignore[assignment]
        output.error_message = error_message
        if any(b.type == "toolCall" for b in output.content) and output.stop_reason == "stop":
            output.stop_reason = "toolUse"

    async for event in openai_stream:
        if options.on_provider_stream_event:
            maybe = options.on_provider_stream_event(event, model)
            if asyncio.iscoroutine(maybe):
                await maybe
        event_type = event.get("type")

        if event_type == "response.created":
            response = event.get("response") or {}
            output.response_id = response.get("id")

        elif event_type == "response.output_item.added":
            create_slot(event.get("output_index", 0), event.get("item") or {})

        elif event_type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            slot = get_slot(event.get("output_index", 0), "thinking")
            if slot is None:
                continue
            slot.block.thinking += event.get("delta", "")
            stream.push(
                ThinkingDeltaEvent(content_index=slot.content_index, delta=event.get("delta", ""), partial=output)
            )

        elif event_type == "response.reasoning_summary_part.done":
            slot = get_slot(event.get("output_index", 0), "thinking")
            if slot is None:
                continue
            slot.block.thinking += "\n\n"
            stream.push(ThinkingDeltaEvent(content_index=slot.content_index, delta="\n\n", partial=output))

        elif event_type in ("response.output_text.delta", "response.refusal.delta"):
            slot = get_slot(event.get("output_index", 0), "text")
            if slot is None:
                continue
            slot.block.text += event.get("delta", "")
            stream.push(TextDeltaEvent(content_index=slot.content_index, delta=event.get("delta", ""), partial=output))

        elif event_type == "response.function_call_arguments.delta":
            slot = get_slot(event.get("output_index", 0), "toolCall")
            if slot is None or slot.partial_json is None:
                continue
            slot.partial_json += event.get("delta", "")
            slot.block.arguments = parse_streaming_json(slot.partial_json)
            push_tool_call_delta(slot, event.get("delta", ""))

        elif event_type == "response.function_call_arguments.done":
            slot = get_slot(event.get("output_index", 0), "toolCall")
            if slot is None or slot.partial_json is None:
                continue
            previous_partial_json = slot.partial_json
            slot.partial_json = event.get("arguments", "")
            slot.block.arguments = parse_streaming_json(slot.partial_json)
            if slot.partial_json.startswith(previous_partial_json):
                delta = slot.partial_json[len(previous_partial_json):]
                if delta:
                    push_tool_call_delta(slot, delta)

        elif event_type == "response.custom_tool_call_input.delta":
            slot = get_slot(event.get("output_index", 0), "toolCall")
            if slot is None or slot.custom_buffer is None:
                continue
            push_tool_call_delta(
                slot,
                append_custom_tool_call_input(slot, get_custom_tool_call_input(slot) + event.get("delta", ""), False),
            )

        elif event_type == "response.custom_tool_call_input.done":
            slot = get_slot(event.get("output_index", 0), "toolCall")
            if slot is None or slot.custom_buffer is None:
                continue
            push_tool_call_delta(slot, append_custom_tool_call_input(slot, event.get("input", ""), True))

        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            apply_message_phase_stop_reason(item)
            slot = get_or_create_slot(event.get("output_index", 0), item)
            item_type = item.get("type")

            if item_type == "reasoning" and slot is not None and slot.type == "thinking":
                summary_text = "\n\n".join(s.get("text", "") for s in (item.get("summary") or []))
                content_text = "\n\n".join(c.get("text", "") for c in (item.get("content") or []))
                slot.block.thinking = summary_text or content_text or slot.block.thinking
                slot.block.thinking_signature = json.dumps(item, separators=(",", ":"))
                if item.get("id"):
                    reasoning_blocks_by_id[item["id"]] = slot.block
                stream.push(
                    ThinkingEndEvent(
                        content_index=slot.content_index, content=slot.block.thinking, partial=output
                    )
                )
                output_slots.pop(event.get("output_index", 0), None)

            elif item_type == "message" and slot is not None and slot.type == "text":
                slot.block.text = "".join(
                    c.get("text", "") if c.get("type") == "output_text" else c.get("refusal", "")
                    for c in (item.get("content") or [])
                )
                slot.block.text_signature = encode_text_signature_v1(item.get("id", ""), item.get("phase"))
                stream.push(
                    TextEndEvent(content_index=slot.content_index, content=slot.block.text, partial=output)
                )
                output_slots.pop(event.get("output_index", 0), None)

            elif item_type == "function_call" and slot is not None and slot.type == "toolCall":
                if slot.partial_json is not None:
                    slot.block.arguments = parse_streaming_json(item.get("arguments") or slot.partial_json or "{}")
                    slot.partial_json = None
                if item.get("namespace") is not None:
                    slot.block.namespace = item["namespace"]
                stream.push(ToolCallEndEvent(content_index=slot.content_index, tool_call=slot.block, partial=output))
                output_slots.pop(event.get("output_index", 0), None)

            elif item_type == "custom_tool_call" and slot is not None and slot.type == "toolCall":
                if slot.custom_buffer is not None:
                    push_tool_call_delta(
                        slot,
                        append_custom_tool_call_input(
                            slot, item.get("input") or get_custom_tool_call_input(slot), True
                        ),
                    )
                    slot.custom_buffer = None
                if item.get("namespace") is not None:
                    slot.block.namespace = item["namespace"]
                stream.push(ToolCallEndEvent(content_index=slot.content_index, tool_call=slot.block, partial=output))
                output_slots.pop(event.get("output_index", 0), None)

        elif event_type in ("response.completed", "response.incomplete"):
            finalize_response(event.get("response") or {})

        elif event_type == "error":
            raise RuntimeError(f"Error Code {event.get('code')}: {event.get('message')}" or "Unknown error")

        elif event_type == "response.failed":
            saw_terminal_response_event = True
            response = event.get("response") or {}
            output.raw_stop_reason = response.get("status")
            error = response.get("error")
            details = response.get("incomplete_details")
            if error:
                message = f"{error.get('code') or 'unknown'}: {error.get('message') or 'no message'}"
            elif details and details.get("reason"):
                message = f"incomplete: {details['reason']}"
            else:
                message = "Unknown error (no error details in response)"
            raise RuntimeError(message)

    if not saw_terminal_response_event:
        raise RuntimeError("OpenAI Responses stream ended before a terminal response event")


def map_stop_reason(status: Optional[str], incomplete_reason: Optional[str] = None) -> tuple[str, Optional[str]]:
    if not status:
        return "stop", None
    if status == "completed":
        return "stop", None
    if status == "incomplete":
        if incomplete_reason == "max_output_tokens":
            return "length", None
        return (
            "error",
            f"Response incomplete: {incomplete_reason}"
            if incomplete_reason
            else "Response incomplete without a provider reason",
        )
    if status in ("failed", "cancelled"):
        return "error", None
    # These two are wonky ...
    if status in ("in_progress", "queued"):
        return "stop", None
    raise RuntimeError(f"Unhandled stop reason: {status}")

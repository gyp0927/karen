"""Shared utilities for Google Generative AI and Google Vertex providers.

Mirrors api/google-shared.ts. Wire format notes:
- `thought: true` marks thinking parts; `thoughtSignature` is an encrypted
  context-replay blob that can appear on ANY part type and must be echoed back.
- Gemini 3+ models require explicit tool call IDs and support multimodal
  function responses.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncIterable, Callable, Dict, List, Optional

from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost, clamp_thinking_level
from ..transcript import collapse_system_messages, without_initial_system_message
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
from ..utils.sanitize import sanitize_surrogates
from .constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from .transform_messages import transform_messages

# Gemini's discrete thinkingLevel control values (Gemini 3 models).
GOOGLE_API_THINKING_LEVELS = ("THINKING_LEVEL_UNSPECIFIED", "MINIMAL", "LOW", "MEDIUM", "HIGH")


class GoogleThinkingConfig:
    """Request-side thinking selection for Google adapters."""

    def __init__(
        self,
        enabled: bool,
        budget_tokens: Optional[int] = None,
        level: Optional[str] = None,
    ):
        self.enabled = enabled
        #: -1 for dynamic, 0 to disable.
        self.budget_tokens = budget_tokens
        #: One of GOOGLE_API_THINKING_LEVELS.
        self.level = level


# ---------------------------------------------------------------------------
# Thinking level helpers
# ---------------------------------------------------------------------------


def resolve_google_thinking_level(model: Model, level: str) -> str:
    """Resolve a supported pi level or model-specific Google mapping to a standard Google level."""
    mapped = (model.thinking_level_map or {}).get(level)
    resolved_level = mapped.lower() if isinstance(mapped, str) else level
    if resolved_level in ("minimal", "low", "medium", "high"):
        return resolved_level
    raise ValueError(
        f"Unsupported Google thinking level mapping for {model.provider}/{model.id}: {level} -> {mapped}"
    )


def uses_google_thinking_level(model: Model) -> bool:
    """Whether this model uses Gemini's discrete `thinkingLevel` control instead
    of the token-based `thinkingBudget` control."""
    model_id = model.id.lower()
    return (
        re.search(r"gemini-3(?:\.\d+)?-(?:pro|flash)", model_id) is not None
        or model_id == "gemini-flash-latest"
        or model_id == "gemini-flash-lite-latest"
        # Match both hosted Gemma 4 naming forms: gemma-4-* and gemma4-*.
        or re.search(r"gemma-?4", model_id) is not None
    )


def to_google_thinking_level(level: str) -> str:
    return {
        "minimal": "MINIMAL",
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
    }[level]


def get_disabled_google_thinking_config(model: Model) -> Dict[str, Any]:
    if not uses_google_thinking_level(model):
        return {"thinkingBudget": 0}
    fallback = clamp_thinking_level(model, "off")
    if fallback == "off":
        return {"thinkingBudget": 0}
    resolved_level = resolve_google_thinking_level(model, fallback)
    return {"thinkingLevel": to_google_thinking_level(resolved_level)}


def is_thinking_part(part: Dict[str, Any]) -> bool:
    """`thought: true` is the definitive marker for thinking content.

    `thoughtSignature` can appear on ANY part type — it does NOT mark the part
    itself as thinking content.
    """
    return part.get("thought") is True


def retain_thought_signature(existing: Optional[str], incoming: Optional[str]) -> Optional[str]:
    """Some backends only send `thoughtSignature` on the first delta of a block;
    preserve the last non-empty signature within the block."""
    if isinstance(incoming, str) and len(incoming) > 0:
        return incoming
    return existing


# Thought signatures must be base64 for Google APIs (TYPE_BYTES).
_BASE64_SIGNATURE_PATTERN = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _is_valid_thought_signature(signature: Optional[str]) -> bool:
    if not signature or len(signature) % 4 != 0:
        return False
    return _BASE64_SIGNATURE_PATTERN.match(signature) is not None


def _resolve_thought_signature(is_same_provider_and_model: bool, signature: Optional[str]) -> Optional[str]:
    """Only keep signatures from the same provider/model and with valid base64."""
    return signature if (is_same_provider_and_model and _is_valid_thought_signature(signature)) else None


# ---------------------------------------------------------------------------
# Model capabilities
# ---------------------------------------------------------------------------


def _get_gemini_major_version(model_id: str) -> Optional[int]:
    match = re.match(r"gemini(?:-live)?-(\d+)", model_id.lower())
    if not match:
        return None
    return int(match.group(1))


def requires_tool_call_id(model_id: str) -> bool:
    """Models via Google APIs that require explicit tool call IDs in function
    calls/responses."""
    major = _get_gemini_major_version(model_id)
    return model_id.startswith("claude-") or model_id.startswith("gpt-oss-") or (major is not None and major >= 3)


def _supports_multimodal_function_response(model_id: str) -> bool:
    major = _get_gemini_major_version(model_id)
    if major is not None:
        return major >= 3
    return True


def supports_google_strict_tool_sampling(model_id: str) -> bool:
    """Gemini 3+ enforces required function parameters in validated tool-calling modes."""
    major = _get_gemini_major_version(model_id)
    return major is not None and major >= 3


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def convert_messages(model: Model, context: TranscriptContext) -> List[Dict[str, Any]]:
    """Convert internal messages to Gemini Content[] format."""
    # Gemini has no mid-conversation system messages; the leading prompt is
    # sent as systemInstruction.
    conversation = without_initial_system_message(collapse_system_messages(context).messages)
    contents: List[Dict[str, Any]] = []

    def normalize_tool_call_id(call_id: str, _target_model: Model = model, _source=None) -> str:
        if not requires_tool_call_id(model.id):
            return call_id
        return "".join(ch if (ch.isascii() and (ch.isalnum() or ch in "-_")) else "_" for ch in call_id)[:64]

    transformed_messages = transform_messages(conversation, model, normalize_tool_call_id)

    for msg in transformed_messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                contents.append({"role": "user", "parts": [{"text": sanitize_surrogates(msg.content)}]})
            else:
                parts: List[Dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        parts.append({"text": sanitize_surrogates(item.text)})
                    else:
                        parts.append({"inlineData": {"mimeType": item.mime_type, "data": item.data}})
                if not parts:
                    continue
                contents.append({"role": "user", "parts": parts})

        elif msg.role == "assistant":
            parts = []
            # Only same provider AND same model messages keep thinking blocks.
            is_same_provider_and_model = msg.provider == model.provider and msg.model == model.id

            for block in msg.content:
                if block.type == "text":
                    thought_signature = _resolve_thought_signature(is_same_provider_and_model, block.text_signature)
                    # Skip empty text blocks — unless they carry a thought
                    # signature. Gemini can attach the signature to a part whose
                    # visible text is empty and requires it echoed back;
                    # dropping it breaks the reasoning chain.
                    if (not block.text or block.text.strip() == "") and not thought_signature:
                        continue
                    part: Dict[str, Any] = {"text": sanitize_surrogates(block.text)}
                    if thought_signature:
                        part["thoughtSignature"] = thought_signature
                    parts.append(part)

                elif block.type == "thinking":
                    if is_same_provider_and_model:
                        thought_signature = _resolve_thought_signature(True, block.thinking_signature)
                        if (not block.thinking or block.thinking.strip() == "") and not thought_signature:
                            continue
                        part = {"thought": True, "text": sanitize_surrogates(block.thinking)}
                        if thought_signature:
                            part["thoughtSignature"] = thought_signature
                        parts.append(part)
                    else:
                        # Cross-provider/model: signature unusable, plain text
                        # (no tags to avoid model mimicking them).
                        if not block.thinking or block.thinking.strip() == "":
                            continue
                        parts.append({"text": sanitize_surrogates(block.thinking)})

                elif block.type == "toolCall":
                    tool_call: ToolCall = block  # type: ignore[assignment]
                    thought_signature = _resolve_thought_signature(
                        is_same_provider_and_model, tool_call.thought_signature
                    )
                    function_call: Dict[str, Any] = {"name": tool_call.name, "args": tool_call.arguments or {}}
                    if requires_tool_call_id(model.id):
                        function_call["id"] = tool_call.id
                    part = {"functionCall": function_call}
                    if thought_signature:
                        part["thoughtSignature"] = thought_signature
                    parts.append(part)

            if not parts:
                continue
            contents.append({"role": "model", "parts": parts})

        elif msg.role == "toolResult":
            text_result = "\n".join(c.text for c in msg.content if c.type == "text")
            images: List[ImageContent] = (
                [c for c in msg.content if c.type == "image"] if "image" in model.input else []
            )
            has_text = len(text_result) > 0
            has_images = len(images) > 0

            # Gemini 3+ models support multimodal function responses with images
            # nested inside functionResponse.parts. Claude and other non-Gemini
            # models behind Cloud Code Assist / Gemini < 3 still need a separate
            # user image turn.
            model_supports_multimodal = _supports_multimodal_function_response(model.id)

            # "output" key for success, "error" key for errors per SDK docs.
            response_value = sanitize_surrogates(text_result) if has_text else "(see attached image)" if has_images else ""
            image_parts = [{"inlineData": {"mimeType": image.mime_type, "data": image.data}} for image in images]

            function_response: Dict[str, Any] = {
                "name": msg.tool_name,
                "response": {"error": response_value} if msg.is_error else {"output": response_value},
            }
            if has_images and model_supports_multimodal:
                function_response["parts"] = image_parts
            if requires_tool_call_id(model.id):
                function_response["id"] = msg.tool_call_id

            function_response_part = {"functionResponse": function_response}

            # Cloud Code Assist API requires all function responses in a single
            # user turn; merge with the last user turn when possible.
            last_content = contents[-1] if contents else None
            if last_content is not None and last_content.get("role") == "user" and any(
                "functionResponse" in p for p in last_content.get("parts", [])
            ):
                last_content["parts"].append(function_response_part)
            else:
                contents.append({"role": "user", "parts": [function_response_part]})

            # For Gemini < 3, add images in a separate user message.
            if has_images and not model_supports_multimodal:
                contents.append({"role": "user", "parts": [{"text": "Tool result image:"}, *image_parts]})

    return contents


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------

_JSON_SCHEMA_META_DECLARATIONS = {
    "$schema",
    "$id",
    "$anchor",
    "$dynamicAnchor",
    "$vocabulary",
    "$comment",
    "$defs",
    "definitions",  # pre-draft-2019-09 equivalent of $defs
}


def sanitize_for_openapi(schema: Any) -> Any:
    """Strip meta-declarations from a schema object."""
    if not isinstance(schema, dict):
        return schema
    return {key: sanitize_for_openapi(value) for key, value in schema.items() if key not in _JSON_SCHEMA_META_DECLARATIONS}


def convert_tools(
    tools: List[Tool],
    use_parameters: bool = False,
    supports_strict_mode: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    """Convert tools to Gemini function declarations format.

    By default uses `parametersJsonSchema` which supports full JSON Schema.
    Set `use_parameters` to use the legacy `parameters` field (OpenAPI 3.03
    Schema) — needed for Cloud Code Assist with Claude models, where the API
    translates `parameters` into Anthropic's `input_schema`.
    """
    if not tools:
        return None
    declarations: List[Dict[str, Any]] = []
    for tool in tools:
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        parameters = get_json_schema_tool_parameters(tool, strict)
        declaration: Dict[str, Any] = {"name": tool.name, "description": tool.description}
        if use_parameters:
            declaration["parameters"] = sanitize_for_openapi(parameters)
        else:
            declaration["parametersJsonSchema"] = parameters
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}]


def map_tool_choice(choice: str) -> str:
    """Map tool choice string to Gemini FunctionCallingConfigMode."""
    if choice in ("auto", "none", "any"):
        return choice.upper()
    return "AUTO"


def resolve_google_function_calling_mode(
    tools: List[Tool],
    tool_choice: Optional[str],
    supports_strict_mode: bool,
) -> Optional[str]:
    use_strict_mode = any(
        resolve_json_schema_strict_sampling(tool, supports_strict_mode) is True for tool in tools
    )
    if tool_choice in ("none", "any"):
        return map_tool_choice(tool_choice)
    if use_strict_mode:
        return "VALIDATED"
    return map_tool_choice(tool_choice) if tool_choice else None


# ---------------------------------------------------------------------------
# Stop reason / usage
# ---------------------------------------------------------------------------


def map_stop_reason_string(reason: str) -> str:
    """Map a raw Gemini finishReason string to our StopReason."""
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    return "error"


# ---------------------------------------------------------------------------
# Shared stream processing
# ---------------------------------------------------------------------------

_tool_call_counter = 0


def _next_tool_call_id(name: str) -> str:
    global _tool_call_counter
    _tool_call_counter += 1
    return f"{name}_{int(time.time() * 1000)}_{_tool_call_counter}"


async def process_google_stream(
    chunks: AsyncIterable[Dict[str, Any]],
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    on_provider_stream_event: Optional[Callable[[Any, Model], Any]] = None,
) -> None:
    """The shared GenerateContentResponse stream loop used by both Google adapters."""
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

    current_block: Optional[Any] = None  # TextContent | ThinkingContent
    blocks = output.content

    def block_index() -> int:
        return len(blocks) - 1

    def close_current_block() -> None:
        nonlocal current_block
        if current_block is None:
            return
        if current_block.type == "text":
            stream.push(TextEndEvent(content_index=block_index(), content=current_block.text, partial=output))
        else:
            stream.push(
                ThinkingEndEvent(content_index=block_index(), content=current_block.thinking, partial=output)
            )
        current_block = None

    async for chunk in chunks:
        if on_provider_stream_event:
            maybe = on_provider_stream_event(chunk, model)
            if asyncio.iscoroutine(maybe):
                await maybe
        # GenerateContentResponse.responseId is an output-only identifier; keep
        # the first non-empty one from the stream.
        if not output.response_id and chunk.get("responseId"):
            output.response_id = chunk["responseId"]

        candidates = chunk.get("candidates") or []
        candidate = candidates[0] if candidates else None
        parts = ((candidate or {}).get("content") or {}).get("parts") or []
        for part in parts:
            if part.get("text") is not None:
                thinking_part = is_thinking_part(part)
                if (
                    current_block is None
                    or (thinking_part and current_block.type != "thinking")
                    or (not thinking_part and current_block.type != "text")
                ):
                    close_current_block()
                    if thinking_part:
                        current_block = ThinkingContent(thinking="")
                        blocks.append(current_block)
                        stream.push(ThinkingStartEvent(content_index=block_index(), partial=output))
                    else:
                        current_block = TextContent(text="")
                        blocks.append(current_block)
                        stream.push(TextStartEvent(content_index=block_index(), partial=output))

                if current_block.type == "thinking":
                    current_block.thinking += part["text"]
                    current_block.thinking_signature = retain_thought_signature(
                        current_block.thinking_signature, part.get("thoughtSignature")
                    )
                    stream.push(
                        ThinkingDeltaEvent(content_index=block_index(), delta=part["text"], partial=output)
                    )
                else:
                    current_block.text += part["text"]
                    current_block.text_signature = retain_thought_signature(
                        current_block.text_signature, part.get("thoughtSignature")
                    )
                    stream.push(TextDeltaEvent(content_index=block_index(), delta=part["text"], partial=output))

            if part.get("functionCall"):
                close_current_block()
                function_call = part["functionCall"]

                # Generate a unique ID if not provided or if it's a duplicate.
                provided_id = function_call.get("id")
                needs_new_id = not provided_id or any(
                    b.type == "toolCall" and b.id == provided_id for b in output.content
                )
                tool_call_id = provided_id if not needs_new_id else _next_tool_call_id(function_call.get("name") or "")

                tool_call = ToolCall(
                    id=tool_call_id,
                    name=function_call.get("name") or "",
                    arguments=function_call.get("args") or {},
                )
                if part.get("thoughtSignature"):
                    tool_call.thought_signature = part["thoughtSignature"]

                output.content.append(tool_call)
                stream.push(ToolCallStartEvent(content_index=block_index(), partial=output))
                stream.push(
                    ToolCallDeltaEvent(
                        content_index=block_index(),
                        delta=json.dumps(tool_call.arguments),
                        partial=output,
                    )
                )
                stream.push(ToolCallEndEvent(content_index=block_index(), tool_call=tool_call, partial=output))

        if candidate and candidate.get("finishReason"):
            output.raw_stop_reason = candidate["finishReason"]
            output.stop_reason = map_stop_reason_string(candidate["finishReason"])  # type: ignore[assignment]
            if any(b.type == "toolCall" for b in output.content) and output.stop_reason == "stop":
                output.stop_reason = "toolUse"

        usage_metadata = chunk.get("usageMetadata")
        if usage_metadata:
            output.usage = Usage(
                input=(usage_metadata.get("promptTokenCount") or 0)
                - (usage_metadata.get("cachedContentTokenCount") or 0),
                output=(usage_metadata.get("candidatesTokenCount") or 0)
                + (usage_metadata.get("thoughtsTokenCount") or 0),
                cache_read=usage_metadata.get("cachedContentTokenCount") or 0,
                cache_write=0,
                reasoning=usage_metadata.get("thoughtsTokenCount") or 0,
                total_tokens=usage_metadata.get("totalTokenCount") or 0,
            )
            calculate_cost(model, output.usage)

    close_current_block()


def get_google_budget(model: Model, level: str, custom_budgets=None) -> int:
    """Token budget for a thinking level on budget-based Gemini models. -1 = dynamic."""
    if custom_budgets is not None:
        value = getattr(custom_budgets, level, None)
        if value is not None:
            return value

    if "2.5-pro" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}[level]
    if "2.5-flash-lite" in model.id:
        return {"minimal": 512, "low": 2048, "medium": 8192, "high": 24576}[level]
    if "2.5-flash" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}[level]
    return -1


def build_google_config(
    model: Model,
    context: TranscriptContext,
    options,
) -> Dict[str, Any]:
    """The shared request-config builder (REST body minus contents/model)."""
    from ..transcript import get_current_tools, get_initial_system_message
    from ..utils.text import get_system_message_text

    initial_system_message = get_initial_system_message(context.messages)
    current_tools = get_current_tools(context.messages)

    generation_config: Dict[str, Any] = {}
    if options and options.temperature is not None:
        generation_config["temperature"] = options.temperature
    if options and options.max_tokens is not None:
        generation_config["maxOutputTokens"] = options.max_tokens

    supports_strict_mode = supports_google_strict_tool_sampling(model.id)
    function_calling_mode = (
        resolve_google_function_calling_mode(current_tools, options.tool_choice if options else None, supports_strict_mode)
        if current_tools
        else None
    )
    system_instruction = get_system_message_text(initial_system_message) if initial_system_message else ""

    body: Dict[str, Any] = {}
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": sanitize_surrogates(system_instruction)}]}
    if current_tools:
        body["tools"] = convert_tools(current_tools, False, supports_strict_mode)
    if function_calling_mode is not None:
        body["toolConfig"] = {"functionCallingConfig": {"mode": function_calling_mode}}

    thinking = getattr(options, "thinking", None) if options else None
    if thinking is not None and thinking.enabled and model.reasoning:
        thinking_config: Dict[str, Any] = {"includeThoughts": True}
        if thinking.level is not None:
            thinking_config["thinkingLevel"] = thinking.level
        elif thinking.budget_tokens is not None:
            thinking_config["thinkingBudget"] = thinking.budget_tokens
        generation_config["thinkingConfig"] = thinking_config
    elif model.reasoning and thinking is not None and not thinking.enabled:
        generation_config["thinkingConfig"] = get_disabled_google_thinking_config(model)

    if generation_config:
        body["generationConfig"] = generation_config

    return body


def build_google_stream_simple_thinking(model: Model, options) -> GoogleThinkingConfig:
    """Resolve SimpleStreamOptions reasoning into a GoogleThinkingConfig."""
    if not options or not options.reasoning:
        return GoogleThinkingConfig(enabled=False)
    clamped_reasoning = clamp_thinking_level(model, options.reasoning)
    if clamped_reasoning == "off":
        return GoogleThinkingConfig(enabled=False)
    resolved_level = resolve_google_thinking_level(model, clamped_reasoning)
    if uses_google_thinking_level(model):
        return GoogleThinkingConfig(enabled=True, level=to_google_thinking_level(resolved_level))
    return GoogleThinkingConfig(
        enabled=True,
        budget_tokens=get_google_budget(model, resolved_level, options.thinking_budgets),
    )

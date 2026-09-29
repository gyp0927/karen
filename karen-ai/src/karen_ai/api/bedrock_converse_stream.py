"""Amazon Bedrock ConverseStream adapter, mirroring api/bedrock-converse-stream.ts.

pi-ai drives this through @aws-sdk/client-bedrock-runtime; karen-ai speaks the
documented REST protocol directly instead (consistent with the other adapters):
a SigV4-signed (or bearer-token) JSON POST to
`{endpoint}/model/{modelId}/converse-stream`, whose response body is an
`application/vnd.amazon.eventstream` binary event stream.

Credential chain: env vars (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/
AWS_SESSION_TOKEN), then the shared credentials/config files for the active
profile. SSO/IMDS/container credential providers are not supported; bearer
tokens (AWS_BEARER_TOKEN_BEDROCK) bypass SigV4 entirely.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import quote, urlsplit

import httpx

from ..abort import AbortSignal
from ..errors import AbortError
from ..models import calculate_cost
from ..transcript import (
    TranscriptContext,
    collapse_system_messages,
    get_current_tools,
    get_initial_system_message,
    without_initial_system_message,
)
from ..event_stream import AssistantMessageEventStream
from ..types import (
    AssistantMessage,
    AssistantMessageDiagnostic,
    CacheRetention,
    DoneEvent,
    ErrorEvent,
    Model,
    ProviderResponse,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
)
from ..types import SimpleStreamOptions, StreamOptions, ThinkingBudgets, ThinkingLevel, Tool
from ..utils.aws_eventstream import EventStreamDecoder, EventStreamMessage
from ..utils.aws_sigv4 import AwsCredentials, resolve_aws_credentials, resolve_aws_profile_region, sign_request
from ..utils.diagnostics import append_assistant_message_diagnostic
from ..utils.headers import provider_headers_to_record
from ..utils.json_parse import parse_streaming_json
from ..utils.provider_env import get_provider_env_value
from ..utils.retry import ProviderHttpError, retry_provider_request
from ..utils.sanitize import sanitize_surrogates
from ..utils.text import get_system_message_text
from .constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from .simple_options import adjust_max_tokens_for_thinking, build_base_options, clamp_max_tokens_to_context, clamp_reasoning
from .transform_messages import transform_messages

BedrockThinkingDisplay = Literal["summarized", "omitted"]
BedrockToolChoice = Union[Literal["auto", "any", "none"], Dict[str, str]]


class BedrockOptions(StreamOptions):
    region: Optional[str] = None
    profile: Optional[str] = None
    tool_choice: Optional[BedrockToolChoice] = None
    #: See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-reasoning.html
    reasoning: Optional[ThinkingLevel] = None
    thinking_budgets: Optional[ThinkingBudgets] = None
    interleaved_thinking: Optional[bool] = None
    thinking_display: Optional[BedrockThinkingDisplay] = None
    request_metadata: Optional[Dict[str, str]] = None
    #: Bearer token for Bedrock API key authentication (AWS_BEARER_TOKEN_BEDROCK).
    bearer_token: Optional[str] = None


EMPTY_TEXT_PLACEHOLDER = "<empty>"
# Matches the placeholder the Anthropic API path uses for redacted thinking.
REDACTED_THINKING_PLACEHOLDER = "[Reasoning redacted]"

_BEDROCK_SERVICE = "bedrock"
DEFAULT_REGION = "us-east-1"

# Human-readable prefixes for Bedrock exception names; downstream retry logic
# matches patterns like `server.?error`, so keep the legacy prefix format.
BEDROCK_ERROR_PREFIXES = {
    "InternalServerException": "Internal server error",
    "ModelStreamErrorException": "Model stream error",
    "ValidationException": "Validation error",
    "ThrottlingException": "Throttling error",
    "ServiceUnavailableException": "Service unavailable",
}
BEDROCK_DATA_RETENTION_DOCS_URL = "https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html"

MAX_BEDROCK_DIAGNOSTIC_VALUE_CHARS = 200


class BedrockServiceException(Exception):
    """A modeled Bedrock error (HTTP-level or delivered as a stream exception event)."""

    def __init__(self, name: str, message: str, status: Optional[int] = None, request_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.name = name
        self.status = status
        self.request_id = request_id


def _format_bedrock_error(error: BaseException) -> str:
    if isinstance(error, BedrockServiceException):
        core = f"{error.status}: {error}" if error.status is not None else str(error)
        hint = (
            f" See {BEDROCK_DATA_RETENTION_DOCS_URL} for supported data retention modes."
            if re.search(r"data retention mode", core, re.IGNORECASE)
            else ""
        )
        prefix = BEDROCK_ERROR_PREFIXES.get(error.name, error.name)
        return f"{prefix}: {core}{hint}"
    core = str(error) or type(error).__name__
    hint = (
        f" See {BEDROCK_DATA_RETENTION_DOCS_URL} for supported data retention modes."
        if re.search(r"data retention mode", core, re.IGNORECASE)
        else ""
    )
    return f"{core}{hint}"


def _normalize_diagnostic_value(value: Any) -> Optional[str]:
    """Over-long header values are dropped rather than truncated."""
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed if 0 < len(trimmed) <= MAX_BEDROCK_DIAGNOSTIC_VALUE_CHARS else None


def _append_bedrock_failure_diagnostic(
    output: AssistantMessage,
    error: BaseException,
    fallback_request_id: Optional[str],
) -> None:
    details: Dict[str, Any] = {}
    if isinstance(error, BedrockServiceException):
        if error.status is not None:
            details["status"] = error.status
        if error.name.endswith("Exception"):
            code = _normalize_diagnostic_value(error.name)
            if code:
                details["errorCode"] = code
        request_id = _normalize_diagnostic_value(error.request_id) or fallback_request_id
    elif isinstance(error, ProviderHttpError):
        details["status"] = error.status
        request_id = fallback_request_id
    else:
        request_id = fallback_request_id
    if request_id:
        details["requestId"] = request_id
    if not details:
        return
    append_assistant_message_diagnostic(
        output,
        AssistantMessageDiagnostic(
            type="bedrock_response_failure", timestamp=int(time.time() * 1000), details=details
        ),
    )


# -- Model predicates ----------------------------------------------------------


def _get_model_match_candidates(model_id: str, model_name: Optional[str] = None) -> List[str]:
    values = [model_id, model_name] if model_name else [model_id]
    candidates: List[str] = []
    for value in values:
        lower = value.lower()
        candidates.append(lower)
        candidates.append(re.sub(r"[\s_.:]+", "-", lower))
    return candidates


def supports_adaptive_thinking(model_id: str, model_name: Optional[str] = None) -> bool:
    candidates = _get_model_match_candidates(model_id, model_name)
    return any(
        marker in candidate
        for candidate in candidates
        for marker in ("opus-4-6", "opus-4-7", "opus-4-8", "opus-5", "sonnet-4-6", "sonnet-5", "fable-5")
    )


def _supports_native_xhigh_effort(model: Model) -> bool:
    candidates = _get_model_match_candidates(model.id, model.name)
    return any(
        marker in candidate
        for candidate in candidates
        for marker in ("opus-4-7", "opus-4-8", "opus-5", "sonnet-5", "fable-5")
    )


def _map_thinking_level_to_effort(model: Model, level: Optional[ThinkingLevel]) -> str:
    if level == "xhigh" and _supports_native_xhigh_effort(model):
        return "xhigh"
    level_map = getattr(model, "thinking_level_map", None) or {}
    mapped = level_map.get(level) if level else None
    if isinstance(mapped, str):
        return mapped
    if level in ("minimal", "low"):
        return "low"
    if level == "medium":
        return "medium"
    return "high"


def _is_anthropic_claude_model(model: Model) -> bool:
    model_id = model.id.lower()
    name = (model.name or "").lower()
    return (
        "anthropic.claude" in model_id
        or "anthropic/claude" in model_id
        or "anthropic.claude" in name
        or "anthropic/claude" in name
        or "claude" in name
    )


def supports_prompt_caching(model: Model, env=None) -> bool:
    candidates = _get_model_match_candidates(model.id, model.name)
    if not any("claude" in candidate for candidate in candidates):
        # Application inference profiles don't contain the model name in the ARN.
        return get_provider_env_value("AWS_BEDROCK_FORCE_CACHE", env) == "1"
    if any(marker in candidate for candidate in candidates for marker in ("fable-5", "opus-5", "sonnet-5")):
        return True
    if any("-4-" in candidate for candidate in candidates):
        return True
    if any("claude-3-7-sonnet" in candidate for candidate in candidates):
        return True
    if any("claude-3-5-haiku" in candidate for candidate in candidates):
        return True
    return False


def _supports_thinking_signature(model: Model) -> bool:
    """Only Anthropic Claude models support the reasoningText signature field."""
    return _is_anthropic_claude_model(model)


def _resolve_cache_retention(cache_retention: Optional[CacheRetention], env=None) -> CacheRetention:
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


# -- Region / endpoint / credentials --------------------------------------------


def _get_configured_bedrock_region(options: BedrockOptions) -> Optional[str]:
    return (
        options.region
        or get_provider_env_value("AWS_REGION", options.env)
        or get_provider_env_value("AWS_DEFAULT_REGION", options.env)
    )


def get_standard_bedrock_endpoint_region(base_url: Optional[str]) -> Optional[str]:
    if not base_url:
        return None
    try:
        hostname = urlsplit(base_url).hostname or ""
    except ValueError:
        return None
    match = re.match(r"^bedrock-runtime(?:-fips)?\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?$", hostname.lower())
    return match.group(1) if match else None


def _should_use_explicit_bedrock_endpoint(
    base_url: str,
    configured_region: Optional[str],
    has_ambient_configured_profile: bool,
) -> bool:
    endpoint_region = get_standard_bedrock_endpoint_region(base_url)
    if not endpoint_region:
        return True
    return not configured_region and not has_ambient_configured_profile


def _is_gov_cloud_bedrock_target(model: Model, options: BedrockOptions) -> bool:
    region = _get_configured_bedrock_region(options)
    if region and region.lower().startswith("us-gov-"):
        return True
    model_id = model.id.lower()
    return model_id.startswith("us-gov.") or model_id.startswith("arn:aws-us-gov:")


def _resolve_region(model: Model, options: BedrockOptions, endpoint_region: Optional[str], use_explicit: bool) -> str:
    # Region resolution: ARN-embedded > explicit option > env vars > shared config
    # (profile) > endpoint-derived > us-east-1.
    arn_match = re.match(r"^arn:aws(?:-[a-z0-9-]+)?:bedrock:([a-z0-9-]+):", model.id)
    if arn_match:
        return arn_match.group(1)
    configured = _get_configured_bedrock_region(options)
    if configured:
        return configured
    profile_region = resolve_aws_profile_region(options.env, options.profile)
    if profile_region:
        return profile_region
    if endpoint_region and use_explicit:
        return endpoint_region
    return DEFAULT_REGION


# -- Payload conversion ----------------------------------------------------------


def _encode_path_segment(segment: str) -> str:
    # Smithy extendedEncodeURIComponent: RFC 3986 unreserved characters only.
    return quote(segment, safe="-_.~")


def _encode_model_id_path(model_id: str) -> str:
    # Greedy label: each `/`-separated segment is encoded independently.
    return "/".join(_encode_path_segment(segment) for segment in model_id.split("/"))


def _normalize_tool_call_id(tool_call_id: str, *_args: Any) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_call_id)
    return sanitized[:64] if len(sanitized) > 64 else sanitized


def _create_non_blank_text_block(text: str) -> Optional[Dict[str, Any]]:
    sanitized = sanitize_surrogates(text)
    return {"text": sanitized} if sanitized.strip() else None


def _create_required_text_block(text: str) -> Dict[str, Any]:
    return _create_non_blank_text_block(text) or {"text": EMPTY_TEXT_PLACEHOLDER}


def _sanitize_bedrock_document(value: Any) -> Any:
    if isinstance(value, list):
        return [_sanitize_bedrock_document(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_bedrock_document(nested) for key, nested in value.items() if len(key) > 0}
    return value


def _create_image_block(mime_type: str, data: str) -> Dict[str, Any]:
    formats = {
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
    }
    image_format = formats.get(mime_type)
    if image_format is None:
        raise ValueError(f"Unknown image type: {mime_type}")
    return {"source": {"bytes": data}, "format": image_format}


def _convert_tool_result_content(content: List[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for block in content:
        if block.type == "image":
            result.append({"image": _create_image_block(block.mime_type, block.data)})
        else:
            text_block = _create_non_blank_text_block(block.text)
            if text_block:
                result.append(text_block)
    if not result:
        result.append({"text": EMPTY_TEXT_PLACEHOLDER})
    return result


def _decode_redacted_content(signature: Optional[str]) -> Optional[str]:
    """A persisted session carries redacted reasoning as base64; a hand-edited one
    can hold a signature that is not base64 — drop that block instead of failing."""
    if not signature:
        return None
    try:
        base64.b64decode(signature, validate=True)
        return signature
    except Exception:
        return None


def _build_system_prompt(
    system_prompt: Optional[str],
    model: Model,
    cache_retention: CacheRetention,
    env=None,
) -> Optional[List[Dict[str, Any]]]:
    if not system_prompt:
        return None
    blocks: List[Dict[str, Any]] = [{"text": sanitize_surrogates(system_prompt)}]
    if cache_retention != "none" and supports_prompt_caching(model, env):
        cache_point: Dict[str, Any] = {"type": "default"}
        if cache_retention == "long":
            cache_point["ttl"] = "one_hour"
        blocks.append({"cachePoint": cache_point})
    return blocks


def convert_messages(
    context: TranscriptContext,
    model: Model,
    cache_retention: CacheRetention,
    env=None,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    transformed = transform_messages(without_initial_system_message(context.messages), model, _normalize_tool_call_id)

    i = 0
    while i < len(transformed):
        message = transformed[i]
        if message.role == "user":
            content: List[Dict[str, Any]] = []
            if isinstance(message.content, str):
                content.append(_create_required_text_block(message.content))
            else:
                for block in message.content:
                    if block.type == "text":
                        text_block = _create_non_blank_text_block(block.text)
                        if text_block:
                            content.append(text_block)
                    elif block.type == "image":
                        content.append({"image": _create_image_block(block.mime_type, block.data)})
                if not content:
                    content.append({"text": EMPTY_TEXT_PLACEHOLDER})
            result.append({"role": "user", "content": content})
        elif message.role == "assistant":
            # Bedrock rejects messages with empty content arrays.
            if not message.content:
                i += 1
                continue
            content_blocks: List[Dict[str, Any]] = []
            for block in message.content:
                if block.type == "text":
                    text_block = _create_non_blank_text_block(block.text)
                    if not text_block:
                        continue
                    content_blocks.append(text_block)
                elif block.type == "toolCall":
                    content_blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": block.id,
                                "name": block.name,
                                "input": _sanitize_bedrock_document(block.arguments),
                            }
                        }
                    )
                elif block.type == "thinking":
                    if block.redacted:
                        redacted_content = _decode_redacted_content(block.thinking_signature)
                        if redacted_content:
                            content_blocks.append(
                                {"reasoningContent": {"redactedContent": base64.b64decode(redacted_content)}}
                            )
                        continue
                    thinking = sanitize_surrogates(block.thinking)
                    if not thinking.strip():
                        continue
                    if _supports_thinking_signature(model):
                        # Signatures arrive after thinking deltas; replay without one
                        # gets rejected — fall back to plain text, matching Anthropic.
                        if not block.thinking_signature or not block.thinking_signature.strip():
                            content_blocks.append({"text": thinking})
                        else:
                            content_blocks.append(
                                {
                                    "reasoningContent": {
                                        "reasoningText": {"text": thinking, "signature": block.thinking_signature}
                                    }
                                }
                            )
                    else:
                        content_blocks.append({"reasoningContent": {"reasoningText": {"text": thinking}}})
            if not content_blocks:
                i += 1
                continue
            result.append({"role": "assistant", "content": content_blocks})
        elif message.role == "toolResult":
            # Bedrock requires all consecutive tool results in a single user message.
            tool_results: List[Dict[str, Any]] = [
                {
                    "toolResult": {
                        "toolUseId": message.tool_call_id,
                        "content": _convert_tool_result_content(message.content),
                        "status": "error" if message.is_error else "success",
                    }
                }
            ]
            j = i + 1
            while j < len(transformed) and transformed[j].role == "toolResult":
                next_message = transformed[j]
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": next_message.tool_call_id,
                            "content": _convert_tool_result_content(next_message.content),
                            "status": "error" if next_message.is_error else "success",
                        }
                    }
                )
                j += 1
            i = j - 1
            result.append({"role": "user", "content": tool_results})
        i += 1

    # Cache point on the last user message for supported Claude models.
    if cache_retention != "none" and supports_prompt_caching(model, env) and result:
        last_message = result[-1]
        if last_message["role"] == "user":
            cache_point = {"type": "default"}
            if cache_retention == "long":
                cache_point["ttl"] = "one_hour"
            last_message["content"].append({"cachePoint": cache_point})

    return result


def convert_tool_config(
    tools: Optional[List[Tool]],
    tool_choice: Optional[BedrockToolChoice],
    supports_strict_mode: bool,
) -> Optional[Dict[str, Any]]:
    if not tools:
        return None
    if tool_choice == "none":
        return None

    bedrock_tools = []
    for tool in tools:
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        bedrock_tools.append(
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {"json": get_json_schema_tool_parameters(tool, strict)},
                    **({"strict": True} if strict is True else {}),
                }
            }
        )

    bedrock_tool_choice: Optional[Dict[str, Any]] = None
    if tool_choice == "auto":
        bedrock_tool_choice = {"auto": {}}
    elif tool_choice == "any":
        bedrock_tool_choice = {"any": {}}
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        bedrock_tool_choice = {"tool": {"name": tool_choice["name"]}}

    return {"tools": bedrock_tools, "toolChoice": bedrock_tool_choice}


def _build_additional_model_request_fields(model: Model, options: BedrockOptions) -> Optional[Dict[str, Any]]:
    if not options.reasoning or not model.reasoning:
        return None

    if _is_anthropic_claude_model(model):
        # GovCloud Bedrock currently rejects the Claude thinking.display field.
        display = None if _is_gov_cloud_bedrock_target(model, options) else (options.thinking_display or "summarized")
        if supports_adaptive_thinking(model.id, model.name):
            result: Dict[str, Any] = {
                "thinking": {"type": "adaptive", **({"display": display} if display is not None else {})},
                "output_config": {"effort": _map_thinking_level_to_effort(model, options.reasoning)},
            }
        else:
            default_budgets: Dict[str, int] = {
                "minimal": 1024,
                "low": 2048,
                "medium": 8192,
                "high": 16384,
                "xhigh": 16384,  # Budget-based Claude clamps extended levels to high
                "max": 16384,
            }
            level = "high" if options.reasoning in ("xhigh", "max") else options.reasoning
            budgets = options.thinking_budgets.model_dump() if options.thinking_budgets else {}
            budget = budgets.get(level) or default_budgets[options.reasoning]
            result = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": budget,
                    **({"display": display} if display is not None else {}),
                }
            }
        if not supports_adaptive_thinking(model.id, model.name) and (options.interleaved_thinking is not False):
            result["anthropic_beta"] = ["interleaved-thinking-2025-05-14"]
        return result

    return None


def _map_stop_reason(reason: Optional[str]) -> "tuple[str, Optional[str]]":
    if reason in ("end_turn", "stop_sequence"):
        return "stop", None
    if reason in ("max_tokens", "model_context_window_exceeded"):
        return "length", None
    if reason == "tool_use":
        return "toolUse", None
    if reason:
        return "error", f"Provider stopped with: {reason}"
    return "error", None


# -- Streaming ---------------------------------------------------------------------


def _handle_content_block_start(
    event: Dict[str, Any],
    output: AssistantMessage,
    block_indexes: List[Optional[int]],
    stream: AssistantMessageEventStream,
) -> None:
    index = event.get("contentBlockIndex", 0)
    start = event.get("start") or {}
    tool_use = start.get("toolUse")
    if tool_use:
        block = ToolCall(id=tool_use.get("toolUseId") or "", name=tool_use.get("name") or "", arguments={})
        block.partial_json = ""
        output.content.append(block)
        block_indexes.append(index)
        stream.push(ToolCallStartEvent(content_index=len(output.content) - 1, partial=output))


def _handle_content_block_delta(
    event: Dict[str, Any],
    output: AssistantMessage,
    block_indexes: List[Optional[int]],
    redacted_chunks: Dict[int, List[bytes]],
    stream: AssistantMessageEventStream,
) -> None:
    content_block_index = event.get("contentBlockIndex", 0)
    delta = event.get("delta") or {}
    index = next(
        (position for position, block_index in enumerate(block_indexes) if block_index == content_block_index),
        -1,
    )
    block = output.content[index] if index >= 0 else None

    if "text" in delta:
        # handleContentBlockStart is not sent for text blocks; create lazily.
        if block is None:
            new_block = TextContent(text="")
            output.content.append(new_block)
            block_indexes.append(content_block_index)
            index = len(output.content) - 1
            block = new_block
            stream.push(TextStartEvent(content_index=index, partial=output))
        if isinstance(block, TextContent):
            block.text += delta["text"]
            stream.push(TextDeltaEvent(content_index=index, delta=delta["text"], partial=output))
    elif delta.get("toolUse") is not None and isinstance(block, ToolCall):
        partial = delta["toolUse"].get("input") or ""
        block.partial_json = (block.partial_json or "") + partial
        block.arguments = parse_streaming_json(block.partial_json)
        stream.push(ToolCallDeltaEvent(content_index=index, delta=partial, partial=output))
    elif delta.get("reasoningContent") is not None:
        reasoning = delta["reasoningContent"]
        if block is None:
            new_block = ThinkingContent(thinking="", thinking_signature="")
            output.content.append(new_block)
            block_indexes.append(content_block_index)
            index = len(output.content) - 1
            block = new_block
            stream.push(ThinkingStartEvent(content_index=index, partial=output))

        if isinstance(block, ThinkingContent):
            if reasoning.get("text"):
                block.thinking += reasoning["text"]
                stream.push(ThinkingDeltaEvent(content_index=index, delta=reasoning["text"], partial=output))
            # `thinkingSignature` holds either an Anthropic signature or an opaque
            # redacted payload, never both: mixing them would corrupt whichever arrived first.
            if reasoning.get("signature") and not block.redacted:
                block.thinking_signature = (block.thinking_signature or "") + reasoning["signature"]
            redacted = reasoning.get("redactedContent")
            if redacted:
                # Encrypted reasoning from non-Anthropic models on Bedrock (e.g. OpenAI
                # GPT-5.6). Keep the opaque payload verbatim for replay next turn.
                if not block.redacted:
                    block.redacted = True
                    block.thinking_signature = ""
                    block.thinking += REDACTED_THINKING_PLACEHOLDER
                    stream.push(
                        ThinkingDeltaEvent(content_index=index, delta=REDACTED_THINKING_PLACEHOLDER, partial=output)
                    )
                redacted_chunks.setdefault(index, []).append(base64.b64decode(redacted))


def _flush_redacted_content(block: Any, chunks: Optional[List[bytes]]) -> None:
    """Encode buffered encrypted reasoning into `thinkingSignature`; the scratch
    buffer must never reach a persisted message."""
    if not isinstance(block, ThinkingContent) or not chunks:
        return
    block.thinking_signature = base64.b64encode(b"".join(chunks)).decode("ascii")


def _finalize_streaming_block(position: int, output: AssistantMessage, redacted_chunks: Dict[int, List[bytes]]) -> None:
    block = output.content[position]
    if isinstance(block, ToolCall):
        # partial_json is only a streaming scratch buffer; never persist it.
        block.partial_json = None
    _flush_redacted_content(block, redacted_chunks.pop(position, None))


def _handle_content_block_stop(
    event: Dict[str, Any],
    output: AssistantMessage,
    block_indexes: List[Optional[int]],
    redacted_chunks: Dict[int, List[bytes]],
    stream: AssistantMessageEventStream,
) -> None:
    index = next(
        (
            position
            for position, block_index in enumerate(block_indexes)
            if block_index == event.get("contentBlockIndex")
        ),
        -1,
    )
    if index < 0:
        return
    block = output.content[index]
    block_indexes[index] = None

    if isinstance(block, TextContent):
        stream.push(TextEndEvent(content_index=index, content=block.text, partial=output))
    elif isinstance(block, ThinkingContent):
        _flush_redacted_content(block, redacted_chunks.pop(index, None))
        stream.push(ThinkingEndEvent(content_index=index, content=block.thinking, partial=output))
    elif isinstance(block, ToolCall):
        block.arguments = parse_streaming_json(block.partial_json)
        block.partial_json = None
        stream.push(ToolCallEndEvent(content_index=index, tool_call=block, partial=output))


def _handle_metadata(event: Dict[str, Any], model: Model, output: AssistantMessage) -> None:
    usage = event.get("usage")
    if usage:
        output.usage.input = usage.get("inputTokens") or 0
        output.usage.output = usage.get("outputTokens") or 0
        output.usage.cache_read = usage.get("cacheReadInputTokens") or 0
        output.usage.cache_write = usage.get("cacheWriteInputTokens") or 0
        cache_details = usage.get("cacheDetails") or []
        total_1h = sum(
            detail.get("inputTokens") or 0 for detail in cache_details if detail.get("ttl") == "one_hour"
        )
        output.usage.cache_write1h = total_1h if cache_details else None
        output.usage.total_tokens = usage.get("totalTokens") or output.usage.input + output.usage.output
        calculate_cost(model, output.usage)


# -- Entry points ----------------------------------------------------------------


def stream(
    model: Model,
    context: TranscriptContext,
    options: Optional[BedrockOptions] = None,
) -> AssistantMessageEventStream:
    options = options or BedrockOptions()
    event_stream = AssistantMessageEventStream()
    # Bedrock has no mid-conversation system messages; fold them into the leading prompt.
    normalized_context = collapse_system_messages(context)

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
        block_indexes: List[Optional[int]] = []
        redacted_chunks: Dict[int, List[bytes]] = {}
        response_request_id: Optional[str] = None

        try:
            env = options.env
            options_profile = options.profile or (env or {}).get("AWS_PROFILE")
            configured_region = _get_configured_bedrock_region(options)
            has_ambient_profile = bool(get_provider_env_value("AWS_PROFILE"))
            endpoint_region = get_standard_bedrock_endpoint_region(model.base_url)
            use_explicit_endpoint = _should_use_explicit_bedrock_endpoint(
                model.base_url, configured_region, has_ambient_profile
            )
            # Only pin standard AWS endpoints when no region or ambient profile is
            # configured; custom endpoints (VPC/proxy) always pass through.
            endpoint = model.base_url.rstrip("/") if use_explicit_endpoint else None
            region = _resolve_region(model, options, endpoint_region, use_explicit_endpoint)
            if endpoint is None:
                endpoint = f"https://bedrock-runtime.{region}.amazonaws.com"

            skip_auth = get_provider_env_value("AWS_BEDROCK_SKIP_AUTH", env) == "1"
            bearer_token = (
                options.bearer_token
                or options.api_key
                or get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", env)
            )
            use_bearer_token = bearer_token is not None and not skip_auth

            credentials: Optional[AwsCredentials] = None
            if not use_bearer_token:
                if skip_auth:
                    credentials = AwsCredentials("dummy-access-key", "dummy-secret-key")
                else:
                    credentials = resolve_aws_credentials(env, options_profile)
                    if credentials is None:
                        raise ValueError(
                            "No AWS credentials for Bedrock: set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, "
                            "configure a profile, or use AWS_BEARER_TOKEN_BEDROCK"
                        )

            supports_strict_mode = bool(getattr(model.compat, "supports_strict_mode", False))
            cache_retention = _resolve_cache_retention(options.cache_retention, env)
            inference_max_tokens = (
                options.max_tokens if options.max_tokens is not None else (model.max_tokens if _is_anthropic_claude_model(model) else None)
            )
            initial_system_message = get_initial_system_message(normalized_context.messages)
            initial_system_prompt = (
                get_system_message_text(initial_system_message) if initial_system_message else None
            )

            body: Dict[str, Any] = {
                "messages": convert_messages(normalized_context, model, cache_retention, env),
                "system": _build_system_prompt(initial_system_prompt, model, cache_retention, env),
                "inferenceConfig": {
                    **({"maxTokens": inference_max_tokens} if inference_max_tokens is not None else {}),
                    **({"temperature": options.temperature} if options.temperature is not None else {}),
                },
                "toolConfig": convert_tool_config(
                    get_current_tools(normalized_context.messages), options.tool_choice, supports_strict_mode
                ),
                "additionalModelRequestFields": _build_additional_model_request_fields(model, options),
                **({"requestMetadata": options.request_metadata} if options.request_metadata is not None else {}),
            }
            body = {key: value for key, value in body.items() if value is not None}
            if options.on_payload:
                next_body = options.on_payload(body, model)
                if asyncio.iscoroutine(next_body):
                    next_body = await next_body
                if next_body is not None:
                    body = next_body

            body_bytes = json.dumps(body).encode("utf-8")
            url = f"{endpoint}/model/{_encode_model_id_path(model.id)}/converse-stream"

            base_headers: Dict[str, str] = {
                "content-type": "application/json",
                "accept": "application/vnd.amazon.eventstream",
            }
            custom_headers = provider_headers_to_record(options.headers)
            for name, value in (custom_headers or {}).items():
                # Reserved SigV4/auth headers are silently skipped.
                lower = name.lower()
                if lower.startswith("x-amz-") or lower in ("authorization", "host"):
                    continue
                base_headers[name] = value

            if use_bearer_token:
                base_headers["Authorization"] = f"Bearer {bearer_token}"
                signed_headers = base_headers
            else:
                assert credentials is not None
                signed_headers = sign_request(
                    method="POST",
                    url=url,
                    headers=base_headers,
                    body=body_bytes,
                    credentials=credentials,
                    region=region,
                    service=_BEDROCK_SERVICE,
                )

            timeout = httpx.Timeout((options.timeout_ms or 120_000) / 1000, connect=60.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async def do_request() -> httpx.Response:
                    request = client.build_request("POST", url, content=body_bytes, headers=signed_headers)
                    response = await client.send(request, stream=True)
                    if response.status_code >= 400:
                        error_body = (await response.aread()).decode("utf-8", "replace")
                        request_id = response.headers.get("x-amzn-requestid") or response.headers.get(
                            "x-amzn-request-id"
                        )
                        await response.aclose()
                        error_name = "UnknownError"
                        error_message = error_body
                        try:
                            parsed = json.loads(error_body)
                            if isinstance(parsed, dict):
                                raw_name = parsed.get("__type") or parsed.get("code") or error_name
                                raw_name = raw_name.split("#")[-1]
                                error_name = raw_name[:1].upper() + raw_name[1:] if raw_name else error_name
                                error_message = parsed.get("message") or parsed.get("Message") or error_body
                        except ValueError:
                            pass
                        raise BedrockServiceException(
                            error_name, error_message, status=response.status_code, request_id=request_id
                        )
                    return response

                response = await retry_provider_request(
                    do_request,
                    max_retries=options.max_retries if options.max_retries is not None else 2,
                    max_retry_delay_ms=options.max_retry_delay_ms,
                    signal=options.signal,
                )

                try:
                    response_request_id = _normalize_diagnostic_value(
                        response.headers.get("x-amzn-requestid") or response.headers.get("x-amzn-request-id")
                    )
                    if options.on_response:
                        maybe = options.on_response(
                            ProviderResponse(status=response.status_code, headers=dict(response.headers)), model
                        )
                        if asyncio.iscoroutine(maybe):
                            await maybe

                    decoder = EventStreamDecoder()
                    async for chunk in response.aiter_bytes():
                        if options.signal and options.signal.aborted:
                            raise AbortError("Request was aborted")
                        for message in decoder.feed(chunk):
                            await _handle_stream_message(
                                message, model, output, block_indexes, redacted_chunks, event_stream, options
                            )
                    decoder.flush()

                    if options.signal and options.signal.aborted:
                        raise AbortError("Request was aborted")
                    if output.stop_reason == "pending":
                        raise RuntimeError("Bedrock stream ended without a stop reason")
                    if output.stop_reason in ("error", "aborted"):
                        raise RuntimeError(output.error_message or "An unknown error occurred")

                    # A stream can settle without stopping every block, so finalize here too.
                    for position in range(len(output.content)):
                        _finalize_streaming_block(position, output, redacted_chunks)
                    event_stream.push(DoneEvent(reason=output.stop_reason, message=output))  # type: ignore[arg-type]
                    event_stream.end()
                finally:
                    await response.aclose()

        except Exception as error:
            for position in range(len(output.content)):
                _finalize_streaming_block(position, output, redacted_chunks)
            output.stop_reason = "aborted" if (options.signal and options.signal.aborted) else "error"
            output.error_message = _format_bedrock_error(error)
            if output.stop_reason == "error":
                _append_bedrock_failure_diagnostic(output, error, response_request_id)
            event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))  # type: ignore[arg-type]
            event_stream.end()

    asyncio.get_running_loop().create_task(run())
    return event_stream


async def _handle_stream_message(
    message: EventStreamMessage,
    model: Model,
    output: AssistantMessage,
    block_indexes: List[Optional[int]],
    redacted_chunks: Dict[int, List[bytes]],
    event_stream: AssistantMessageEventStream,
    options: BedrockOptions,
) -> None:
    message_type = message.message_type
    event_type = message.event_type or ""
    payload = message.json_payload() or {}

    if options.on_provider_stream_event:
        maybe = options.on_provider_stream_event(
            {"messageType": message_type, "eventType": event_type, "payload": payload}, model
        )
        if asyncio.iscoroutine(maybe):
            await maybe

    if message_type == "exception":
        # Shape names arrive camelCase ("throttlingException"); the SDK-style
        # class name is capitalized ("ThrottlingException").
        raw_name = message.exception_type or event_type
        error_name = raw_name[:1].upper() + raw_name[1:] if raw_name else "ModelStreamErrorException"
        error_message = payload.get("message") or payload.get("Message") or error_name
        raise BedrockServiceException(error_name, error_message)

    if event_type == "messageStart":
        if payload.get("role") != "assistant":
            raise RuntimeError("Unexpected assistant message start but got user message start instead")
        event_stream.push(StartEvent(partial=output))
    elif event_type == "contentBlockStart":
        _handle_content_block_start(payload, output, block_indexes, event_stream)
    elif event_type == "contentBlockDelta":
        _handle_content_block_delta(payload, output, block_indexes, redacted_chunks, event_stream)
    elif event_type == "contentBlockStop":
        _handle_content_block_stop(payload, output, block_indexes, redacted_chunks, event_stream)
    elif event_type == "messageStop":
        output.raw_stop_reason = payload.get("stopReason")
        stop_reason, error_message = _map_stop_reason(payload.get("stopReason"))
        output.stop_reason = stop_reason  # type: ignore[assignment]
        if error_message:
            output.error_message = error_message
    elif event_type == "metadata":
        _handle_metadata(payload, model, output)


def stream_simple(
    model: Model,
    context: TranscriptContext,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    base = build_base_options(model, context, options, None).model_dump()
    base["tool_choice"] = options.tool_choice if options else None
    if not options or not options.reasoning:
        return stream(model, context, BedrockOptions(**base, reasoning=None))

    if _is_anthropic_claude_model(model):
        if supports_adaptive_thinking(model.id, model.name):
            return stream(
                model,
                context,
                BedrockOptions(**base, reasoning=options.reasoning, thinking_budgets=options.thinking_budgets),
            )

        # Undefined means the caller did not request an output cap; let the helper use the model cap.
        adjusted_max_tokens, adjusted_budget = adjust_max_tokens_for_thinking(
            base.get("max_tokens"), model.max_tokens, options.reasoning, options.thinking_budgets
        )
        max_tokens = clamp_max_tokens_to_context(model, context, adjusted_max_tokens)
        reasoning = clamp_reasoning(options.reasoning)
        thinking_budgets = options.thinking_budgets.model_dump() if options.thinking_budgets else {}
        if reasoning:
            thinking_budgets[reasoning] = min(adjusted_budget, max(0, max_tokens - 1024))
        return stream(
            model,
            context,
            BedrockOptions(
                **base,
                max_tokens=max_tokens,
                reasoning=options.reasoning,
                thinking_budgets=thinking_budgets,
            ),
        )

    return stream(
        model,
        context,
        BedrockOptions(**base, reasoning=options.reasoning, thinking_budgets=options.thinking_budgets),
    )

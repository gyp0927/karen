"""pi-messages API implementation, mirroring pi-ai's api/pi-messages.ts.

Streams pi's own message protocol directly to a backend: the request is a
single POST of `{ model, context, options }` to `<baseUrl>/messages`, the
response is an SSE stream of serialized assistant-message events plus a
terminal `done`/`error` event. This is the wire protocol spoken by the Radius
gateway, but any backend implementing it can be used.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Literal, Optional, Union

import httpx

from ..event_stream import AssistantMessageEventStream
from ..types import (
    AssistantMessage,
    AssistantMessageDiagnostic,
    CacheRetention,
    ContentBlock,
    DoneEvent,
    ErrorEvent,
    Model,
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
    ThinkingLevel,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TranscriptContext,
    Usage,
)
from ..utils.diagnostics import append_assistant_message_diagnostic, create_assistant_message_diagnostic
from ..utils.headers import headers_to_record, provider_headers_to_record
from ..utils.json_parse import parse_streaming_json
from ..utils.provider_env import get_provider_env_value
from ..utils.retry import ProviderHttpError, retry_provider_request
from ..utils.sse import iterate_sse_messages


class PiMessagesOptions(StreamOptions):
    reasoning: Optional[ThinkingLevel] = None
    tool_choice: Optional[Any] = None
    """Ask the backend for debug metadata (e.g. routing response headers)."""
    debug: Optional[bool] = None


class PiMessagesResponseError(Exception):
    def __init__(self, message: str, code: Optional[str], diagnostic_details: Dict[str, Any]) -> None:
        super().__init__(message)
        self.name = "PiMessagesResponseError"
        self.code = code
        self.diagnostic_details = diagnostic_details


def _parse_error_body(body: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return None
    error = parsed.get("error") if isinstance(parsed, dict) else None
    return parsed if isinstance(error, dict) else None


def _truncate_diagnostic_string(value: str) -> str:
    max_length = 8192
    return f"{value[:max_length]}…" if len(value) > max_length else value


def _format_response_error(status: int, reason_phrase: str, body: str, error_body: Optional[Dict[str, Any]]) -> str:
    error = (error_body or {}).get("error") or {}
    message = error.get("message") if isinstance(error.get("message"), str) else None
    code = error.get("code") if isinstance(error.get("code"), str) else None
    suffix = message if message is not None else body
    code_suffix = f" ({code})" if code else ""
    return f"{status} {reason_phrase}: {suffix}{code_suffix}"


def _create_response_error(model: Model, url: str, response: httpx.Response, body: str) -> PiMessagesResponseError:
    error_body = _parse_error_body(body)
    error = (error_body or {}).get("error")
    code = error.get("code") if isinstance(error, dict) and isinstance(error.get("code"), str) else None
    details: Dict[str, Any] = {
        "version": 1,
        "provider": model.provider,
        "model": model.id,
        "url": url,
        "status": response.status_code,
        "statusText": response.reason_phrase,
        "timestampMs": int(time.time() * 1000),
    }
    if isinstance(error, dict):
        details["error"] = error
    elif error_body is None:
        details["body"] = _truncate_diagnostic_string(body)
    return PiMessagesResponseError(
        _format_response_error(response.status_code, response.reason_phrase, body, error_body), code, details
    )


def _append_rewrite_diagnostic(message: AssistantMessage, rewrite: Optional[Dict[str, Any]]) -> None:
    if not rewrite:
        return
    append_assistant_message_diagnostic(
        message,
        AssistantMessageDiagnostic(
            type="pi_messages_rewrite", timestamp=int(time.time() * 1000), details=dict(rewrite)
        ),
    )


def _set_block(partial: AssistantMessage, index: int, block: ContentBlock) -> None:
    # The wire protocol numbers blocks sequentially from 0; grow defensively.
    while len(partial.content) <= index:
        partial.content.append(TextContent(text=""))
    partial.content[index] = block


def _get_block(partial: AssistantMessage, index: int) -> Any:
    return partial.content[index] if index < len(partial.content) else None


class _EventConverter:
    """Folds pi-messages wire events into a partial AssistantMessage."""

    def __init__(self, model: Model) -> None:
        self.partial = AssistantMessage(
            role="assistant",
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="pending",
            timestamp=int(time.time() * 1000),
        )
        self.tool_json: Dict[int, str] = {}

    def convert(self, event: Dict[str, Any]):
        event_type = event.get("type")
        partial = self.partial

        if event_type == "done" or event_type == "error":
            usage = event.get("usage")
            partial.stop_reason = event.get("reason", "stop" if event_type == "done" else "error")
            if isinstance(usage, dict):
                partial.usage = Usage.model_validate(usage)
            if event.get("responseId") is not None:
                partial.response_id = event["responseId"]
            if event.get("providerThinkingLevel") is not None:
                partial.provider_thinking_level = event["providerThinkingLevel"]
            if event_type == "error":
                partial.error_message = event.get("errorMessage")
            _append_rewrite_diagnostic(partial, event.get("rewrite"))
            if event_type == "done":
                return DoneEvent(reason=partial.stop_reason, message=partial)  # type: ignore[arg-type]
            return ErrorEvent(reason=partial.stop_reason, error=partial)  # type: ignore[arg-type]

        index = int(event.get("contentIndex", 0))

        if event_type == "start":
            return StartEvent(partial=partial)
        if event_type == "text_start":
            _set_block(partial, index, TextContent(text=""))
            return TextStartEvent(content_index=index, partial=partial)
        if event_type == "text_delta":
            block = _get_block(partial, index)
            if isinstance(block, TextContent):
                block.text += event.get("delta", "")
            return TextDeltaEvent(content_index=index, delta=event.get("delta", ""), partial=partial)
        if event_type == "text_end":
            block = _get_block(partial, index)
            if isinstance(block, TextContent):
                block.text = event.get("content", block.text)
                block.text_signature = event.get("contentSignature")
            return TextEndEvent(content_index=index, content=event.get("content", ""), partial=partial)
        if event_type == "thinking_start":
            _set_block(partial, index, ThinkingContent(thinking=""))
            return ThinkingStartEvent(content_index=index, partial=partial)
        if event_type == "thinking_delta":
            block = _get_block(partial, index)
            if isinstance(block, ThinkingContent):
                block.thinking += event.get("delta", "")
            return ThinkingDeltaEvent(content_index=index, delta=event.get("delta", ""), partial=partial)
        if event_type == "thinking_end":
            block = _get_block(partial, index)
            if isinstance(block, ThinkingContent):
                block.thinking = event.get("content", block.thinking)
                block.thinking_signature = event.get("contentSignature")
                block.redacted = event.get("redacted")
            return ThinkingEndEvent(content_index=index, content=event.get("content", ""), partial=partial)
        if event_type == "toolcall_start":
            _set_block(
                partial, index, ToolCall(id=event.get("id", ""), name=event.get("toolName", ""), arguments={})
            )
            self.tool_json[index] = ""
            return ToolCallStartEvent(content_index=index, partial=partial)
        if event_type == "toolcall_delta":
            delta = event.get("delta", "")
            json_so_far = f"{self.tool_json.get(index, '')}{delta}"
            self.tool_json[index] = json_so_far
            block = _get_block(partial, index)
            if isinstance(block, ToolCall):
                block.arguments = parse_streaming_json(json_so_far) or {}
            return ToolCallDeltaEvent(content_index=index, delta=delta, partial=partial)
        if event_type == "toolcall_end":
            wire_call = event.get("toolCall") or {}
            block = _get_block(partial, index)
            if isinstance(block, ToolCall):
                block.id = wire_call.get("id", block.id)
                block.name = wire_call.get("name", block.name)
                block.arguments = wire_call.get("arguments", block.arguments) or {}
                tool_call = block
            else:
                tool_call = ToolCall(
                    id=wire_call.get("id", ""),
                    name=wire_call.get("name", ""),
                    arguments=wire_call.get("arguments") or {},
                )
                _set_block(partial, index, tool_call)
            self.tool_json.pop(index, None)
            return ToolCallEndEvent(content_index=index, tool_call=tool_call, partial=partial)

        # Unknown event: surface as a start-shaped no-op is wrong; ignore it.
        return None


def _create_error_event(model: Model, error: BaseException, aborted: bool):
    reason = "aborted" if aborted else "error"
    assistant_message = AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason=reason,  # type: ignore[arg-type]
        error_message=str(error),
        timestamp=int(time.time() * 1000),
    )
    if not aborted and isinstance(error, PiMessagesResponseError):
        append_assistant_message_diagnostic(
            assistant_message,
            create_assistant_message_diagnostic(
                "pi_messages_response_failure", error, error.diagnostic_details
            ),
        )
    return ErrorEvent(reason=reason, error=assistant_message)  # type: ignore[arg-type]


def _resolve_cache_retention(
    cache_retention: Optional[CacheRetention], env: Optional[Dict[str, str]]
) -> Optional[CacheRetention]:
    if cache_retention:
        return cache_retention
    # Backend defaults apply when unset; only the legacy env opt-in is mapped.
    return "long" if get_provider_env_value("PI_CACHE_RETENTION", env) == "long" else None


def stream(
    model: Model,
    context: TranscriptContext,
    options: Optional[PiMessagesOptions] = None,
) -> AssistantMessageEventStream:
    event_stream = AssistantMessageEventStream()
    converter = _EventConverter(model)

    async def run() -> None:
        try:
            api_key = options.api_key if options else None
            if not api_key:
                raise ValueError(f'No API key provided for provider "{model.provider}"')

            url = f"{model.base_url.rstrip('/')}/messages"
            if options and options.debug:
                url = f"{url}?debug=1"

            payload: Any = {
                "model": model.id,
                "context": context.model_dump(mode="json", by_alias=True, exclude_none=True),
                "options": {
                    "temperature": options.temperature if options else None,
                    "maxTokens": options.max_tokens if options else None,
                    "reasoning": options.reasoning if options else None,
                    "cacheRetention": _resolve_cache_retention(
                        options.cache_retention if options else None, options.env if options else None
                    ),
                    "sessionId": options.session_id if options else None,
                    "toolChoice": options.tool_choice if options else None,
                },
            }
            if options and options.on_payload:
                next_payload = options.on_payload(payload, model)
                if asyncio.iscoroutine(next_payload):
                    next_payload = await next_payload
                if next_payload is not None:
                    payload = next_payload

            headers = {
                "authorization": f"Bearer {api_key}",
                "accept": "text/event-stream",
                "content-type": "application/json",
                **(provider_headers_to_record(options.headers if options else None) or {}),
            }

            timeout = httpx.Timeout((options.timeout_ms or 600_000) / 1000, connect=60.0)
            async with httpx.AsyncClient(timeout=timeout) as client:

                async def do_request() -> httpx.Response:
                    request = client.build_request("POST", url, json=payload, headers=headers)
                    response = await client.send(request, stream=True)
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        await response.aclose()
                        raise _create_response_error(model, url, response, body)
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
                            ProviderResponse(status=response.status_code, headers=headers_to_record(response.headers)),
                            model,
                        )
                        if asyncio.iscoroutine(maybe):
                            await maybe

                    async for sse in iterate_sse_messages(
                        response.aiter_text(), options.signal if options else None
                    ):
                        data = sse.data.strip()
                        if not data or data == "[DONE]":
                            continue
                        pi_event = json.loads(data)
                        if options and options.on_provider_stream_event:
                            maybe = options.on_provider_stream_event(pi_event, model)
                            if asyncio.iscoroutine(maybe):
                                await maybe
                        event = converter.convert(pi_event)
                        if event is None:
                            continue
                        event_stream.push(event)
                        if isinstance(event, (DoneEvent, ErrorEvent)):
                            return

                    raise ProviderHttpError(f"{model.provider} stream ended without a terminal event")
                finally:
                    await response.aclose()
        except Exception as error:
            aborted = bool(options and options.signal and options.signal.aborted)
            event_stream.push(_create_error_event(model, error, aborted))
        finally:
            event_stream.end()

    asyncio.get_running_loop().create_task(run())
    return event_stream


def stream_simple(
    model: Model,
    context: TranscriptContext,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    pi_options = PiMessagesOptions.model_validate(options.model_dump()) if options else PiMessagesOptions()
    return stream(model, context, pi_options)

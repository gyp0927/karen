"""Proxy stream function (pi's `proxy.ts`).

For apps that route LLM calls through a server: the server holds the provider
credentials, receives `{model, context, options}` at `{proxyUrl}/api/stream`,
and streams back `ProxyAssistantMessageEvent`s as `data: `-prefixed JSON lines
(SSE framing, one event per line, the `partial` field stripped to save
bandwidth). The partial assistant message is reconstructed client-side.

Use as the agent's `stream_fn`:

    agent = Agent(stream_fn=lambda model, context, options: stream_proxy(
        model, context,
        ProxyStreamOptions(**vars(options), auth_token=token, proxy_url="https://genai.example.com"),
    ))
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
from karen_ai import (
    AbortSignal,
    AssistantMessage,
    AssistantMessageEvent,
    CacheRetention,
    DoneEvent,
    ErrorEvent,
    EventStream,
    Model,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingBudgets,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingLevel,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TranscriptContext,
    Transport,
    Usage,
)
from karen_ai.utils.json_parse import parse_streaming_json

__all__ = [
    "ProxyStreamOptions",
    "build_proxy_request_options",
    "process_proxy_event",
    "stream_proxy",
]


@dataclass
class ProxyStreamOptions:
    """SimpleStreamOptions' serializable subset plus proxy connection settings."""

    #: Auth token for the proxy server.
    auth_token: str
    #: Proxy server URL (e.g. "https://genai.example.com").
    proxy_url: str
    #: Local abort signal for the proxy request.
    signal: Optional[AbortSignal] = None
    temperature: Optional[float] = None
    sampling_params: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = None
    reasoning: Optional[ThinkingLevel] = None
    cache_retention: Optional[CacheRetention] = None
    session_id: Optional[str] = None
    headers: Optional[Dict[str, Optional[str]]] = None
    metadata: Optional[Dict[str, Any]] = None
    transport: Optional[Transport] = None
    thinking_budgets: Optional[ThinkingBudgets] = None
    max_retry_delay_ms: Optional[int] = None


def build_proxy_request_options(options: ProxyStreamOptions) -> Dict[str, Any]:
    """The serializable options slice sent to the proxy (pi's buildProxyRequestOptions).

    Unset options are omitted, mirroring JSON.stringify dropping `undefined`.
    """
    candidates: Dict[str, Any] = {
        "temperature": options.temperature,
        "samplingParams": options.sampling_params,
        "maxTokens": options.max_tokens,
        "reasoning": options.reasoning,
        "cacheRetention": options.cache_retention,
        "sessionId": options.session_id,
        "headers": options.headers,
        "metadata": options.metadata,
        "transport": options.transport,
        "thinkingBudgets": (
            options.thinking_budgets.model_dump(mode="json", by_alias=True, exclude_none=True)
            if options.thinking_budgets is not None
            else None
        ),
        "maxRetryDelayMs": options.max_retry_delay_ms,
    }
    return {key: value for key, value in candidates.items() if value is not None}


def _content_at(partial: AssistantMessage, index: int) -> Optional[Any]:
    return partial.content[index] if 0 <= index < len(partial.content) else None


def process_proxy_event(
    proxy_event: Dict[str, Any], partial: AssistantMessage
) -> Optional[AssistantMessageEvent]:
    """Apply one proxy event to `partial` and return the full event to publish."""
    event_type = proxy_event.get("type")

    if event_type == "start":
        return StartEvent(partial=partial)

    if event_type == "text_start":
        index = proxy_event["contentIndex"]
        _set_content(partial, index, TextContent(text=""))
        return TextStartEvent(content_index=index, partial=partial)

    if event_type == "text_delta":
        index = proxy_event["contentIndex"]
        content = _content_at(partial, index)
        if isinstance(content, TextContent):
            content.text += proxy_event["delta"]
            return TextDeltaEvent(content_index=index, delta=proxy_event["delta"], partial=partial)
        raise ValueError("Received text_delta for non-text content")

    if event_type == "text_end":
        index = proxy_event["contentIndex"]
        content = _content_at(partial, index)
        if isinstance(content, TextContent):
            content.text_signature = proxy_event.get("contentSignature")
            return TextEndEvent(content_index=index, content=content.text, partial=partial)
        raise ValueError("Received text_end for non-text content")

    if event_type == "thinking_start":
        index = proxy_event["contentIndex"]
        _set_content(partial, index, ThinkingContent(thinking=""))
        return ThinkingStartEvent(content_index=index, partial=partial)

    if event_type == "thinking_delta":
        index = proxy_event["contentIndex"]
        content = _content_at(partial, index)
        if isinstance(content, ThinkingContent):
            content.thinking += proxy_event["delta"]
            return ThinkingDeltaEvent(content_index=index, delta=proxy_event["delta"], partial=partial)
        raise ValueError("Received thinking_delta for non-thinking content")

    if event_type == "thinking_end":
        index = proxy_event["contentIndex"]
        content = _content_at(partial, index)
        if isinstance(content, ThinkingContent):
            content.thinking_signature = proxy_event.get("contentSignature")
            return ThinkingEndEvent(content_index=index, content=content.thinking, partial=partial)
        raise ValueError("Received thinking_end for non-thinking content")

    if event_type == "toolcall_start":
        index = proxy_event["contentIndex"]
        tool_call = ToolCall(id=proxy_event["id"], name=proxy_event["toolName"], arguments={})
        tool_call._partial_json = ""
        _set_content(partial, index, tool_call)
        return ToolCallStartEvent(content_index=index, partial=partial)

    if event_type == "toolcall_delta":
        index = proxy_event["contentIndex"]
        content = _content_at(partial, index)
        if isinstance(content, ToolCall):
            content._partial_json = (content._partial_json or "") + proxy_event["delta"]
            content.arguments = parse_streaming_json(content._partial_json) or {}
            return ToolCallDeltaEvent(content_index=index, delta=proxy_event["delta"], partial=partial)
        raise ValueError("Received toolcall_delta for non-toolCall content")

    if event_type == "toolcall_end":
        index = proxy_event["contentIndex"]
        content = _content_at(partial, index)
        if isinstance(content, ToolCall):
            final_call = ToolCall.model_validate(proxy_event["toolCall"])
            final_call._partial_json = None
            _set_content(partial, index, final_call)
            return ToolCallEndEvent(content_index=index, tool_call=final_call, partial=partial)
        return None

    if event_type == "done":
        partial.stop_reason = proxy_event["reason"]
        partial.usage = Usage.model_validate(proxy_event["usage"])
        if proxy_event.get("providerThinkingLevel") is not None:
            partial.provider_thinking_level = proxy_event["providerThinkingLevel"]
        return DoneEvent(reason=proxy_event["reason"], message=partial)

    if event_type == "error":
        partial.stop_reason = proxy_event["reason"]
        partial.error_message = proxy_event.get("errorMessage")
        partial.usage = Usage.model_validate(proxy_event["usage"])
        if proxy_event.get("providerThinkingLevel") is not None:
            partial.provider_thinking_level = proxy_event["providerThinkingLevel"]
        return ErrorEvent(reason=proxy_event["reason"], error=partial)

    import warnings

    warnings.warn(f"Unhandled proxy event type: {event_type}")
    return None


def _set_content(partial: AssistantMessage, index: int, block: Any) -> None:
    """pi's `partial.content[i] = block`, which extends the array past the end."""
    while len(partial.content) < index:
        partial.content.append(TextContent(text=""))
    if index == len(partial.content):
        partial.content.append(block)
    else:
        partial.content[index] = block


def stream_proxy(
    model: Model,
    context: TranscriptContext,
    options: ProxyStreamOptions,
) -> EventStream[AssistantMessageEvent, AssistantMessage]:
    """Stream function that proxies through a server instead of calling providers directly."""
    stream: EventStream[AssistantMessageEvent, AssistantMessage] = EventStream(
        lambda event: event.type in ("done", "error"),
        lambda event: event.message if isinstance(event, DoneEvent) else event.error,  # type: ignore[union-attr]
    )
    signal = options.signal

    async def run() -> None:
        partial = AssistantMessage(
            role="assistant",
            stop_reason="pending",
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            timestamp=int(time.time() * 1000),
        )
        response_cell: Dict[str, Any] = {}
        abort_watcher: Optional[asyncio.Task[None]] = None

        if signal is not None:

            async def watch_abort() -> None:
                await signal.wait()
                response = response_cell.get("response")
                if response is not None:
                    await response.aclose()

            abort_watcher = asyncio.create_task(watch_abort())

        try:
            payload = {
                "model": model.model_dump(mode="json", by_alias=True, exclude_none=True),
                "context": context.model_dump(mode="json", by_alias=True, exclude_none=True),
                "options": build_proxy_request_options(options),
            }
            # pi's fetch sets no timeout; keep only a connect timeout so a dead
            # host cannot hang the stream forever.
            timeout = httpx.Timeout(None, connect=60.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.send(
                    client.build_request(
                        "POST",
                        f"{options.proxy_url}/api/stream",
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {options.auth_token}",
                            "Content-Type": "application/json",
                        },
                    ),
                    stream=True,
                )
                response_cell["response"] = response
                try:
                    if response.status_code >= 400:
                        error_message = f"Proxy error: {response.status_code} {response.reason_phrase}"
                        try:
                            error_data = json.loads((await response.aread()).decode("utf-8", "replace"))
                            if isinstance(error_data, dict) and error_data.get("error"):
                                error_message = f"Proxy error: {error_data['error']}"
                        except (ValueError, TypeError):
                            pass  # couldn't parse error response
                        raise RuntimeError(error_message)

                    saw_terminal_event = False

                    def process_line(line: str) -> None:
                        nonlocal saw_terminal_event
                        if not line.startswith("data: "):
                            return
                        data = line[6:].strip()
                        if not data:
                            return
                        proxy_event = json.loads(data)
                        event = process_proxy_event(proxy_event, partial)
                        if event is not None:
                            if event.type in ("done", "error"):
                                saw_terminal_event = True
                            stream.push(event)

                    buffer = ""
                    async for chunk in response.aiter_text():
                        if signal is not None and signal.aborted:
                            raise RuntimeError("Request aborted by user")
                        buffer += chunk
                        lines = buffer.split("\n")
                        buffer = lines.pop() if lines else ""
                        for line in lines:
                            process_line(line)

                    if signal is not None and signal.aborted:
                        raise RuntimeError("Request aborted by user")

                    # The final event may not be newline-terminated.
                    if buffer:
                        process_line(buffer)

                    if not saw_terminal_event:
                        # A clean EOF without done/error means the server dropped the
                        # response mid-stream; don't leave consumers waiting.
                        partial.stop_reason = "error"
                        partial.error_message = "Connection closed by proxy server before the response completed"
                        stream.push(ErrorEvent(reason="error", error=partial))
                finally:
                    await response.aclose()
        except Exception as error:  # noqa: BLE001 - all failures become terminal error events
            error_message = str(error)
            reason = "aborted" if signal is not None and signal.aborted else "error"
            partial.stop_reason = reason  # type: ignore[assignment]
            partial.error_message = error_message
            stream.push(ErrorEvent(reason=reason, error=partial))  # type: ignore[arg-type]
        finally:
            if abort_watcher is not None:
                abort_watcher.cancel()
            stream.end()

    asyncio.get_running_loop().create_task(run())
    return stream

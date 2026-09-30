"""Proxy stream function (pi's proxy.ts): event reconstruction + wire behavior.

`stream_proxy` is tested against a real loopback HTTP server streaming scripted
`data: ` lines, since the proxy wire format is the whole point of the module.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from karen_ai import (
    AbortController,
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
)
from karen_ai.providers import faux_model
from karen_ai.types import TranscriptContext, UserMessage

from karen_agent.proxy import (
    ProxyStreamOptions,
    build_proxy_request_options,
    process_proxy_event,
    stream_proxy,
)


def _partial() -> AssistantMessage:
    model = faux_model()
    return AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="pending",
        timestamp=1,
    )


_USAGE = {
    "input": 12,
    "output": 4,
    "cacheRead": 1,
    "cacheWrite": 2,
    "totalTokens": 19,
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
}


# -- process_proxy_event ------------------------------------------------------


def test_start_event():
    partial = _partial()
    event = process_proxy_event({"type": "start"}, partial)
    assert event is not None and event.type == "start" and event.partial is partial


def test_text_flow_accumulates_and_signs():
    partial = _partial()
    assert process_proxy_event({"type": "text_start", "contentIndex": 0}, partial).type == "text_start"
    event = process_proxy_event({"type": "text_delta", "contentIndex": 0, "delta": "Hel"}, partial)
    assert event.delta == "Hel" and partial.content[0].text == "Hel"
    process_proxy_event({"type": "text_delta", "contentIndex": 0, "delta": "lo"}, partial)
    event = process_proxy_event({"type": "text_end", "contentIndex": 0, "contentSignature": "sig"}, partial)
    assert event.type == "text_end" and event.content == "Hello"
    block = partial.content[0]
    assert isinstance(block, TextContent) and block.text_signature == "sig"


def test_thinking_flow():
    partial = _partial()
    process_proxy_event({"type": "thinking_start", "contentIndex": 0}, partial)
    process_proxy_event({"type": "thinking_delta", "contentIndex": 0, "delta": "hmm"}, partial)
    event = process_proxy_event({"type": "thinking_end", "contentIndex": 0, "contentSignature": "tsig"}, partial)
    assert event.content == "hmm"
    block = partial.content[0]
    assert isinstance(block, ThinkingContent) and block.thinking_signature == "tsig"


def test_mismatched_content_raises():
    partial = _partial()
    process_proxy_event({"type": "text_start", "contentIndex": 0}, partial)
    with pytest.raises(ValueError, match="thinking_delta for non-thinking"):
        process_proxy_event({"type": "thinking_delta", "contentIndex": 0, "delta": "x"}, partial)
    with pytest.raises(ValueError, match="text_end for non-text"):
        process_proxy_event({"type": "text_end", "contentIndex": 1}, partial)


def test_toolcall_flow_parses_streaming_json():
    partial = _partial()
    process_proxy_event({"type": "toolcall_start", "contentIndex": 0, "id": "call-1", "toolName": "read"}, partial)
    block = partial.content[0]
    assert isinstance(block, ToolCall) and block.id == "call-1" and block.name == "read"
    assert block.arguments == {} and block._partial_json == ""

    process_proxy_event({"type": "toolcall_delta", "contentIndex": 0, "delta": '{"path": "a.txt", "x":'}, partial)
    # partial-json recovery keeps complete keys, drops the torn trailing one
    assert block.arguments == {"path": "a.txt"}
    event = process_proxy_event({"type": "toolcall_delta", "contentIndex": 0, "delta": " 1}"}, partial)
    assert block.arguments == {"path": "a.txt", "x": 1}

    final = {"type": "toolCall", "id": "call-1", "name": "read", "arguments": {"path": "a.txt", "x": 1}}
    event = process_proxy_event({"type": "toolcall_end", "contentIndex": 0, "toolCall": final}, partial)
    assert event.type == "toolcall_end" and event.tool_call.arguments["x"] == 1
    finished = partial.content[0]
    assert finished._partial_json is None


def test_toolcall_end_without_toolcall_content_is_ignored():
    partial = _partial()
    process_proxy_event({"type": "text_start", "contentIndex": 0}, partial)
    event = process_proxy_event(
        {"type": "toolcall_end", "contentIndex": 0, "toolCall": {"type": "toolCall", "id": "c", "name": "n", "arguments": {}}},
        partial,
    )
    assert event is None


def test_done_and_error_events_finalize_partial():
    partial = _partial()
    event = process_proxy_event(
        {"type": "done", "reason": "toolUse", "usage": _USAGE, "providerThinkingLevel": "high"}, partial
    )
    assert isinstance(event, DoneEvent)
    assert partial.stop_reason == "toolUse"
    assert partial.usage.input == 12 and partial.usage.cache_write == 2
    assert partial.provider_thinking_level == "high"

    failing = _partial()
    event = process_proxy_event(
        {"type": "error", "reason": "error", "errorMessage": "boom", "usage": _USAGE}, failing
    )
    assert isinstance(event, ErrorEvent)
    assert failing.stop_reason == "error" and failing.error_message == "boom"
    assert failing.provider_thinking_level is None


def test_unknown_event_type_warns_and_is_ignored():
    partial = _partial()
    with pytest.warns(UserWarning, match="Unhandled proxy event type"):
        assert process_proxy_event({"type": "mystery"}, partial) is None


# -- build_proxy_request_options -----------------------------------------------


def test_build_proxy_request_options_omits_unset():
    options = ProxyStreamOptions(auth_token="t", proxy_url="http://x")
    assert build_proxy_request_options(options) == {}

    options = ProxyStreamOptions(
        auth_token="t",
        proxy_url="http://x",
        temperature=0.5,
        max_tokens=128,
        session_id="sess",
        reasoning="low",
        max_retry_delay_ms=1000,
    )
    assert build_proxy_request_options(options) == {
        "temperature": 0.5,
        "maxTokens": 128,
        "sessionId": "sess",
        "reasoning": "low",
        "maxRetryDelayMs": 1000,
    }


# -- stream_proxy against a scripted loopback server ----------------------------


class _ScriptedServer:
    """Minimal HTTP/1.1 server replaying a scripted response per request."""

    def __init__(self, responder):
        self._responder = responder
        self.requests = []
        self.server = None

    async def __aenter__(self):
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def _handle(self, reader, writer):
        try:
            request_line = (await reader.readline()).decode().strip()
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b""):
                    break
                name, _, value = line.decode().partition(":")
                headers[name.strip().lower()] = value.strip()
            length = int(headers.get("content-length", "0"))
            body = await reader.readexactly(length) if length else b""
            self.requests.append((request_line, headers, body))

            status, reason, chunks, inter_chunk_delay = self._responder()
            payload = b"".join(chunks)
            head = (
                f"HTTP/1.1 {status} {reason}\r\n"
                f"Content-Type: text/event-stream\r\n"
                f"Content-Length: {len(payload)}\r\n"
                f"Connection: close\r\n\r\n"
            )
            writer.write(head.encode())
            for chunk in chunks:
                writer.write(chunk)
                await writer.drain()
                if inter_chunk_delay:
                    await asyncio.sleep(inter_chunk_delay)
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        finally:
            try:
                writer.close()
            except OSError:
                pass


def _sse(*events) -> bytes:
    return "".join(f"data: {json.dumps(event)}\n" for event in events).encode()


def _context() -> TranscriptContext:
    return TranscriptContext(messages=[UserMessage(content="hi", timestamp=1)])


async def _collect(stream):
    return [event async for event in stream]


async def test_stream_proxy_happy_path_and_request_shape():
    done = {"type": "done", "reason": "stop", "usage": _USAGE, "providerThinkingLevel": "high"}
    body = _sse(
        {"type": "start"},
        {"type": "text_start", "contentIndex": 0},
        {"type": "text_delta", "contentIndex": 0, "delta": "Hello"},
    )
    # remaining events, with the final one NOT newline-terminated (EOF flush path)
    tail = _sse(
        {"type": "text_delta", "contentIndex": 0, "delta": " world"},
        {"type": "text_end", "contentIndex": 0, "contentSignature": "sig"},
    ) + f"data: {json.dumps(done)}".encode()

    async with _ScriptedServer(lambda: (200, "OK", [body, tail], 0)) as server:
        stream = stream_proxy(
            faux_model(),
            _context(),
            ProxyStreamOptions(auth_token="test-token", proxy_url=server.url, temperature=0.5),
        )
        events = await _collect(stream)
        message = await stream.result()

    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    assert isinstance(events[-1], DoneEvent)
    assert message.content[0].text == "Hello world"
    assert message.content[0].text_signature == "sig"
    assert message.stop_reason == "stop"
    assert message.usage.input == 12 and message.usage.total_tokens == 19
    assert message.provider_thinking_level == "high"

    request_line, headers, body_bytes = server.requests[0]
    assert request_line == "POST /api/stream HTTP/1.1"
    assert headers["authorization"] == "Bearer test-token"
    payload = json.loads(body_bytes)
    assert payload["model"]["id"] == "faux-1"
    assert payload["context"]["messages"][0]["role"] == "user"
    assert payload["options"] == {"temperature": 0.5}


async def test_stream_proxy_http_error_uses_server_error_field():
    body = json.dumps({"error": "bad token"}).encode()
    async with _ScriptedServer(lambda: (401, "Unauthorized", [body], 0)) as server:
        events = await _collect(
            stream_proxy(faux_model(), _context(), ProxyStreamOptions(auth_token="t", proxy_url=server.url))
        )
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ErrorEvent)
    assert event.reason == "error"
    assert event.error.error_message == "Proxy error: bad token"
    assert event.error.stop_reason == "error"


async def test_stream_proxy_http_error_falls_back_to_status_text():
    async with _ScriptedServer(lambda: (500, "Internal Server Error", [b"not json"], 0)) as server:
        events = await _collect(
            stream_proxy(faux_model(), _context(), ProxyStreamOptions(auth_token="t", proxy_url=server.url))
        )
    assert events[0].error.error_message == "Proxy error: 500 Internal Server Error"


async def test_stream_proxy_eof_without_terminal_event_surfaces_error():
    body = _sse({"type": "start"}, {"type": "text_start", "contentIndex": 0})
    async with _ScriptedServer(lambda: (200, "OK", [body], 0)) as server:
        events = await _collect(
            stream_proxy(faux_model(), _context(), ProxyStreamOptions(auth_token="t", proxy_url=server.url))
        )
    assert [event.type for event in events] == ["start", "text_start", "error"]
    event = events[-1]
    assert event.reason == "error"
    assert event.error.error_message == "Connection closed by proxy server before the response completed"


async def test_stream_proxy_abort_mid_stream():
    controller = AbortController()
    # server sends "start" then stalls; the abort closes the response
    start = _sse({"type": "start"})
    async with _ScriptedServer(lambda: (200, "OK", [start, _sse({"type": "text_start", "contentIndex": 0})], 5)) as server:
        stream = stream_proxy(
            faux_model(),
            _context(),
            ProxyStreamOptions(auth_token="t", proxy_url=server.url, signal=controller.signal),
        )
        events = []
        async for event in stream:
            events.append(event)
            if event.type == "start":
                controller.abort()

    assert events[0].type == "start"
    terminal = events[-1]
    assert isinstance(terminal, ErrorEvent)
    assert terminal.reason == "aborted"
    assert terminal.error.stop_reason == "aborted"


async def test_stream_proxy_malformed_delta_becomes_error_event():
    # text_delta before any text_start: process_proxy_event raises inside the pump
    body = _sse({"type": "start"}, {"type": "text_delta", "contentIndex": 0, "delta": "x"})
    async with _ScriptedServer(lambda: (200, "OK", [body], 0)) as server:
        events = await _collect(
            stream_proxy(faux_model(), _context(), ProxyStreamOptions(auth_token="t", proxy_url=server.url))
        )
    assert events[0].type == "start"
    terminal = events[-1]
    assert isinstance(terminal, ErrorEvent)
    assert terminal.reason == "error"
    assert "non-text content" in terminal.error.error_message

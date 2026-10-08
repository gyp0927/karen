"""`StreamableHttpTransport` and `consume_sse_stream` — pi's `test/streamable-http.test.ts`.

pi's "calls fetch without a receiver" test does not carry over: Python has no
`this` binding for a stored callable, and its property — the injected fetch is
the callable used everywhere, including the auth context — is covered by
`test_an_injected_fetch_serves_everything_including_the_auth_context`.
"""

import asyncio
import json
import re
import threading
import time
from typing import Any, Dict, List, Optional

import pytest

from karen_mcp import (
    LATEST_PROTOCOL_VERSION,
    HttpRequest,
    McpAuthRequiredError,
    McpClient,
    McpClientOptions,
    McpError,
    McpHttpError,
    McpSessionExpiredError,
    SseEvent,
    ConsumeSseOptions,
    StreamableHttpReconnectOptions,
    StreamableHttpTransport,
    StreamableHttpTransportOptions,
    consume_sse_stream,
    http_fetch,
)


async def _chunks(pieces: List[bytes]):
    for piece in pieces:
        yield piece


async def until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


def protocol_handler(request, response, requests, message: Optional[Dict[str, Any]] = None):
    """pi's `protocolHandler`: a JSON answer for `initialize` and `tools/list`,
    an SSE answer for everything else, 405 for GET, 202 for notifications."""
    if request.method == "GET":
        response.write_head(405)
        response.end()
        return
    if request.method == "DELETE":
        response.write_head(200)
        response.end()
        return
    message = message if message is not None else (request.message or {})
    if "id" not in message:
        response.write_head(202)
        response.end()
        return
    if message.get("method") == "initialize":
        response.write_head(200, {"content-type": "application/json", "mcp-session-id": "session-1"})
        response.end(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "protocolVersion": LATEST_PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "http-fixture", "version": "1.0.0"},
                    },
                }
            )
        )
        return
    if message.get("method") == "tools/list":
        response.write_head(200, {"content-type": "application/json"})
        response.end(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]},
                }
            )
        )
        return
    response.write_head(200, {"content-type": "text/event-stream"})
    response.write("id: tool-result\n")
    response.end(
        "data: "
        + json.dumps(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {"content": [{"type": "text", "text": "hello"}]},
            }
        )
        + "\n\n"
    )


def _client(name: str = "http-test") -> McpClient:
    return McpClient(McpClientOptions(name=name, version="1.0.0"))


async def test_parses_chunked_crlf_events_comments_ids_and_multiline_data():
    pieces = [': keepalive\r\nid: 7\r\ndata: {"one":\r\n'.encode(), b"data: 1}\r\n\r\n"]
    events: List[SseEvent] = []
    await consume_sse_stream(_chunks(pieces), ConsumeSseOptions(on_event=events.append))
    assert events == [SseEvent(id="7", data='{"one":\n1}')]


async def test_dispatches_an_event_the_stream_ends_without_a_blank_line():
    # pi's final `dispatch()`: a server may close right after `data:`.
    events: List[SseEvent] = []
    await consume_sse_stream(_chunks([b"data: unfinished"]), ConsumeSseOptions(on_event=events.append))
    assert events == [SseEvent(data="unfinished")]


async def test_rejects_events_whose_data_lines_exceed_the_limit_without_a_blank_line():
    produced = 0

    async def stream():
        nonlocal produced
        while produced <= 1000:
            produced += 1
            yield b"data: xxxxxxxxxxxxxxxx\n"

    with pytest.raises(ValueError, match="MCP SSE event exceeds 256 bytes"):
        await consume_sse_stream(
            stream(), ConsumeSseOptions(on_event=lambda event: None, max_event_bytes=256)
        )
    assert produced < 100


async def test_handles_json_and_sse_responses_with_session_and_protocol_headers(listen):
    server = listen(protocol_handler)
    transport = StreamableHttpTransport(StreamableHttpTransportOptions(url=server.url))
    client = _client()
    await client.connect(transport)
    assert transport.session_id == "session-1"

    tools = await client.list_tools()
    assert [tool.model_dump(by_alias=True, exclude_none=True) for tool in tools] == [
        {"name": "echo", "inputSchema": {"type": "object"}}
    ]
    assert (await client.call_tool("echo", {"text": "hello"})).content == [{"type": "text", "text": "hello"}]
    assert await until(lambda: any(r.method == "GET" for r in server.requests))
    await client.close()

    list_request = next(
        (request for request in server.requests if (request.message or {}).get("method") == "tools/list"), None
    )
    assert list_request is not None
    assert list_request.header("mcp-session-id") == "session-1"
    assert list_request.header("mcp-protocol-version") == LATEST_PROTOCOL_VERSION
    assert any(request.method == "DELETE" for request in server.requests)


async def test_caller_headers_do_not_duplicate_the_built_in_ones(listen):
    """A caller header that differs only in case from a built-in one is
    replaced, as pi's `Headers` does — not sent a second time."""

    def handler(request, response, requests):
        if request.method == "GET":
            response.write_head(405)
            response.end()
            return
        if request.method == "DELETE":
            response.write_head(200)
            response.end()
            return
        message = request.message or {}
        if "id" not in message:
            response.write_head(202)
            response.end()
            return
        result = (
            {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "serverInfo": {"name": "dupes", "version": "1.0.0"},
            }
            if message.get("method") == "initialize"
            else {"tools": []}
        )
        response.write_head(200, {"content-type": "application/json", "mcp-session-id": "session-1"})
        response.end(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}))

    server = listen(handler)
    transport = StreamableHttpTransport(
        StreamableHttpTransportOptions(
            url=server.url,
            headers={"mcp-session-id": "caller-value", "mcp-protocol-version": "caller-value"},
        )
    )
    client = _client()
    await client.connect(transport)
    await client.list_tools()
    await client.close()

    list_request = next(r for r in server.requests if (r.message or {}).get("method") == "tools/list")
    assert list_request.header_values("mcp-session-id") == ["session-1"]
    assert list_request.header_values("mcp-protocol-version") == [LATEST_PROTOCOL_VERSION]


async def test_classifies_authentication_failures(listen):
    def handler(request, response, requests):
        response.write_head(401, {"www-authenticate": 'Bearer resource_metadata="https://example.com/meta"'})
        response.end("login required")

    server = listen(handler)
    with pytest.raises(McpAuthRequiredError) as error:
        await _client().connect(StreamableHttpTransport(StreamableHttpTransportOptions(url=server.url)))
    assert error.value.status == 401
    assert error.value.body == "login required"
    assert error.value.www_authenticate == 'Bearer resource_metadata="https://example.com/meta"'


async def test_fails_only_the_request_whose_sse_stream_breaks(listen):
    slow_gate = threading.Event()

    def handler(request, response, requests):
        if request.method != "POST":
            return protocol_handler(request, response, requests)
        message = request.message or {}
        name = (message.get("params") or {}).get("name")
        if name == "broken":
            response.write_head(200, {"content-type": "text/event-stream"})
            response.end("data: not json\n\n")
            return
        if name == "slow":
            slow_gate.wait(10)
        protocol_handler(request, response, requests, message)

    server = listen(handler)
    client = _client()
    errors: List[BaseException] = []
    client.on_error(errors.append)
    try:
        await client.connect(
            StreamableHttpTransport(StreamableHttpTransportOptions(url=server.url, open_get_stream=False))
        )

        slow = asyncio.ensure_future(client.call_tool("slow"))
        with pytest.raises(McpError, match="MCP response stream failed"):
            await client.call_tool("broken")
        slow_gate.set()
        assert (await slow).content == [{"type": "text", "text": "hello"}]
        assert len(errors) == 1
    finally:
        slow_gate.set()
        await client.close()


async def test_the_get_stream_opens_only_after_the_initialized_notification(listen):
    gets: List[int] = []

    def handler(request, response, requests):
        if request.method == "GET":
            gets.append(1)
        protocol_handler(request, response, requests)

    server = listen(handler)
    transport = StreamableHttpTransport(StreamableHttpTransportOptions(url=server.url))
    await transport.start()
    await transport.send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
    # The notification's POST has settled, and still no GET may have opened.
    await asyncio.sleep(0.05)
    assert not gets
    await transport.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert await until(lambda: len(gets) == 1)
    await transport.close()


async def test_opens_the_get_stream_after_initialization_and_sends_last_event_id_only_when_resuming(listen):
    order: List[str] = []

    def handler(request, response, requests):
        if request.method == "POST":
            order.append(str((request.message or {}).get("method")))
        else:
            order.append(request.method)
        protocol_handler(request, response, requests)

    server = listen(handler)
    client = _client()
    await client.connect(StreamableHttpTransport(StreamableHttpTransportOptions(url=server.url)))
    await client.call_tool("echo")
    await client.list_tools()
    assert await until(lambda: "GET" in order)
    await client.close()
    assert order.index("GET") > order.index("notifications/initialized")
    assert all(request.header("last-event-id") is None for request in server.requests)


async def test_resumes_a_response_stream_the_server_closed_before_answering(listen):
    resume_headers: List[Optional[str]] = []

    def handler(request, response, requests):
        if request.method == "GET" and request.header("last-event-id"):
            resume_headers.append(request.header("last-event-id"))
            response.write_head(200, {"content-type": "text/event-stream"})
            response.end(
                "id: 2\ndata: "
                + json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": {"content": [{"type": "text", "text": "resumed"}]},
                    }
                )
                + "\n\n"
            )
            return
        if request.method != "POST":
            return protocol_handler(request, response, requests)
        message = request.message or {}
        if message.get("method") != "tools/call":
            return protocol_handler(request, response, requests, message)
        # Priming event (ID, no data) and a retry hint, then the server drops
        # the stream.
        response.write_head(200, {"content-type": "text/event-stream"})
        response.end("id: 1\nretry: 5\ndata:\n\n")

    server = listen(handler)
    client = _client()
    errors: List[BaseException] = []
    client.on_error(errors.append)
    # The server's `retry: 5` hint, not this one-minute floor, must pace the
    # resume — the test would time out otherwise.
    await client.connect(
        StreamableHttpTransport(
            StreamableHttpTransportOptions(
                url=server.url,
                open_get_stream=False,
                reconnect=StreamableHttpReconnectOptions(initial_delay_ms=60_000),
            )
        )
    )
    assert (await client.call_tool("echo", {}, timeout_ms=5_000)).content == [
        {"type": "text", "text": "resumed"}
    ]
    assert resume_headers == ["1"]
    assert errors == []
    await client.close()


async def test_fails_a_request_whose_response_stream_ends_without_an_answer(listen):
    def handler(request, response, requests):
        if request.method != "POST":
            return protocol_handler(request, response, requests)
        message = request.message or {}
        if message.get("method") != "tools/call":
            return protocol_handler(request, response, requests, message)
        response.write_head(200, {"content-type": "text/event-stream"})
        response.end(": nothing here\n\n")

    server = listen(handler)
    client = _client()
    await client.connect(
        StreamableHttpTransport(StreamableHttpTransportOptions(url=server.url, open_get_stream=False))
    )
    with pytest.raises(McpError, match="MCP response stream failed: stream ended without a response"):
        await client.call_tool("echo", {}, timeout_ms=5_000)
    await client.close()


async def test_reconnects_the_get_stream_after_it_drops(listen):
    gets: List[Optional[str]] = []
    release = threading.Event()

    def handler(request, response, requests):
        if request.method != "GET":
            return protocol_handler(request, response, requests)
        gets.append(request.header("last-event-id"))
        notification = json.dumps({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
        response.write_head(200, {"content-type": "text/event-stream"})
        if len(gets) == 1:
            response.end(f"id: g1\ndata: {notification}\n\n")
            return
        response.write(f"id: g2\ndata: {notification}\n\n")
        release.wait(10)

    server = listen(handler)
    client = _client()
    second = asyncio.Event()
    changes: List[Any] = []

    def listener(params: Any) -> None:
        changes.append(params)
        if len(changes) == 2:
            second.set()

    client.on_notification("notifications/tools/list_changed", listener)
    try:
        await client.connect(
            StreamableHttpTransport(
                StreamableHttpTransportOptions(
                    url=server.url, reconnect=StreamableHttpReconnectOptions(initial_delay_ms=1)
                )
            )
        )
        await asyncio.wait_for(second.wait(), 5)
        assert gets == [None, "g1"]
    finally:
        await client.close()
        release.set()


async def test_rejects_a_request_the_server_accepts_without_a_response(listen):
    def handler(request, response, requests):
        if request.method != "POST":
            return protocol_handler(request, response, requests)
        message = request.message or {}
        if message.get("method") != "tools/call":
            return protocol_handler(request, response, requests, message)
        response.write_head(202)
        response.end()

    server = listen(handler)
    client = _client()
    await client.connect(
        StreamableHttpTransport(StreamableHttpTransportOptions(url=server.url, open_get_stream=False))
    )
    with pytest.raises(McpHttpError, match="without a response"):
        await client.call_tool("echo")
    await client.close()


async def test_includes_the_response_body_in_http_errors(listen):
    def handler(request, response, requests):
        response.write_head(400)
        response.end("Invalid Accept header")

    server = listen(handler)
    with pytest.raises(
        McpHttpError, match=re.escape("MCP HTTP request failed with status 400: Invalid Accept header")
    ):
        await _client().connect(StreamableHttpTransport(StreamableHttpTransportOptions(url=server.url)))


async def test_hands_401_and_insufficient_scope_403_to_the_auth_provider_with_the_rejected_token(listen):
    seen: List[tuple] = []
    token = ["old"]

    def handler(request, response, requests):
        if request.method != "POST":
            return protocol_handler(request, response, requests)
        message = request.message or {}
        if request.header("authorization") == "Bearer old":
            response.write_head(401, {"www-authenticate": "Bearer"})
            response.end()
            return
        if message.get("method") == "tools/call" and request.header("authorization") == "Bearer new":
            response.write_head(403, {"www-authenticate": 'Bearer error="insufficient_scope", scope="admin"'})
            response.end()
            return
        return protocol_handler(request, response, requests, message)

    class Provider:
        async def token(self):
            return token[0]

        async def on_unauthorized(self, context):
            seen.append((context.response.status, context.token))
            token[0] = "new" if context.response.status == 401 else "admin"

    server = listen(handler)
    client = _client()
    await client.connect(
        StreamableHttpTransport(
            StreamableHttpTransportOptions(url=server.url, open_get_stream=False, auth_provider=Provider())
        )
    )
    assert (await client.call_tool("echo")).content == [{"type": "text", "text": "hello"}]
    assert seen == [(401, "old"), (403, "new")]
    await client.close()


async def test_an_injected_fetch_serves_everything_including_the_auth_context(listen):
    calls: List[tuple] = []

    async def fetch(url, request):
        calls.append((url, request.method))
        return await http_fetch(url, request)

    token: List[Optional[str]] = [None]

    class Provider:
        async def token(self):
            return token[0]

        async def on_unauthorized(self, context):
            assert context.fetch is fetch
            response = await context.fetch(server.url, HttpRequest(method="GET", headers={}))
            await response.close()
            token[0] = "token"

    def handler(request, response, requests):
        if request.method == "POST" and request.header("authorization") is None:
            response.write_head(401, {"www-authenticate": "Bearer"})
            response.end()
            return
        return protocol_handler(request, response, requests)

    server = listen(handler)
    client = _client()
    await client.connect(
        StreamableHttpTransport(
            StreamableHttpTransportOptions(
                url=server.url, fetch=fetch, open_get_stream=False, auth_provider=Provider()
            )
        )
    )
    tools = await client.list_tools()
    assert [tool.name for tool in tools] == ["echo"]
    await client.close()
    assert calls and all(entry[0] == server.url for entry in calls)


async def test_classifies_an_expired_established_session(listen):
    posts = 0

    def handler(request, response, requests):
        nonlocal posts
        if request.method == "POST":
            posts += 1
            if posts >= 3:
                response.write_head(404)
                response.end("gone")
                return
        protocol_handler(request, response, requests)

    server = listen(handler)
    client = _client()
    await client.connect(
        StreamableHttpTransport(StreamableHttpTransportOptions(url=server.url, open_get_stream=False))
    )
    with pytest.raises(McpSessionExpiredError):
        await client.list_tools()
    await client.close()


async def test_a_404_before_any_session_is_a_plain_http_error(listen):
    def handler(request, response, requests):
        response.write_head(404)
        response.end("gone")

    server = listen(handler)
    with pytest.raises(McpHttpError) as error:
        await _client().connect(
            StreamableHttpTransport(StreamableHttpTransportOptions(url=server.url, open_get_stream=False))
        )
    assert not isinstance(error.value, McpSessionExpiredError)
    assert error.value.status == 404

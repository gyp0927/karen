"""`http_fetch` — body framing, abort, and the `Headers` lookup."""

import asyncio
import threading

import pytest

from karen_mcp import Headers, HttpRequest, McpConnectionClosedError, Signal, http_fetch


async def _body(response) -> str:
    return await response.text()


def test_headers_lookup_is_case_insensitive():
    headers = Headers([("Content-Type", "application/json"), ("MCP-Session-Id", "s1")])
    assert headers.get("content-type") == "application/json"
    assert headers.get("Mcp-Session-Id") == "s1"
    assert headers.get("missing") is None
    assert headers.get("missing", "d") == "d"
    assert "CONTENT-TYPE" in headers
    assert headers.items() == [("Content-Type", "application/json"), ("MCP-Session-Id", "s1")]


async def test_reads_a_content_length_body(listen):
    def handler(request, response, requests):
        response.write_head(200, {"content-type": "application/json", "content-length": "12"})
        response.end('{"hello": 1}')

    server = listen(handler)
    response = await http_fetch(f"{server.url}", HttpRequest(method="GET", headers={}))
    assert response.ok and response.status == 200
    assert response.header("content-type") == "application/json"
    assert await _body(response) == '{"hello": 1}'


async def test_reads_a_chunked_body(listen):
    def handler(request, response, requests):
        response.write_head(200, {"content-type": "text/plain", "transfer-encoding": "chunked"})
        response.write("5\r\nhello\r\n6\r\n world\r\nB\r\n, streamed!\r\n0\r\nX-Trailer: ignored\r\n\r\n")

    server = listen(handler)
    response = await http_fetch(server.url, HttpRequest(method="GET", headers={}))
    assert await _body(response) == "hello world, streamed!"


async def test_reads_a_close_delimited_body_and_streams_it(listen):
    def handler(request, response, requests):
        response.write_head(200, {"content-type": "text/event-stream"})
        response.write("id: 1\n")
        response.write("data: one\n\n")
        response.end()

    server = listen(handler)
    response = await http_fetch(server.url, HttpRequest(method="GET", headers={}))
    seen = []
    async for chunk in response.chunks():
        seen.append(chunk)
    assert b"".join(seen) == b"id: 1\ndata: one\n\n"


async def test_sends_method_headers_and_a_body(listen):
    def handler(request, response, requests):
        assert request.method == "POST"
        assert request.header("x-custom") == "yes"
        assert request.body == b'{"n": 1}'
        response.write_head(200, {"content-length": "2"})
        response.end("ok")

    server = listen(handler)
    await http_fetch(
        server.url,
        HttpRequest(method="POST", headers={"x-custom": "yes", "content-type": "application/json"}, body=b'{"n": 1}'),
    )


async def test_abort_interrupts_a_read(listen):
    hold = threading.Event()

    def handler(request, response, requests):
        response.write_head(200, {"content-type": "text/event-stream"})
        response.write("data: first\n\n")
        hold.wait(10)

    server = listen(handler)
    try:
        signal = Signal()
        response = await http_fetch(server.url, HttpRequest(method="GET", headers={}, signal=signal))
        stream = response.chunks()
        first = await asyncio.wait_for(stream.__anext__(), 5)
        assert b"data:" in first
        signal.abort()
        with pytest.raises(McpConnectionClosedError):
            await asyncio.wait_for(stream.__anext__(), 5)
    finally:
        hold.set()


async def test_abort_interrupts_the_wait_for_response_headers(listen):
    hold = threading.Event()

    def handler(request, response, requests):
        hold.wait(10)
        response.write_head(200)
        response.end()

    server = listen(handler)
    signal = Signal()
    try:
        task = asyncio.ensure_future(
            http_fetch(server.url, HttpRequest(method="POST", headers={}, body=b"{}", signal=signal))
        )
        await asyncio.sleep(0.05)
        signal.abort()
        with pytest.raises(McpConnectionClosedError):
            await asyncio.wait_for(task, 5)
    finally:
        hold.set()


async def test_combines_repeated_response_headers(listen):
    def handler(request, response, requests):
        response.set_header("www-authenticate", 'Bearer error="invalid_token"')
        response.set_header("www-authenticate", 'Bearer error="insufficient_scope"')
        response.write_head(401, {"content-length": "0"})
        response.end()

    server = listen(handler)
    response = await http_fetch(server.url, HttpRequest(method="GET", headers={}))
    assert response.header("www-authenticate") == (
        'Bearer error="invalid_token", Bearer error="insufficient_scope"'
    )


async def test_skips_interim_responses():
    async def handle(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        response = await http_fetch(f"http://127.0.0.1:{port}/mcp", HttpRequest(method="GET", headers={}))
        assert response.status == 200
        assert await response.text() == "ok"
    finally:
        server.close()
        await server.wait_closed()


async def test_rejects_an_invalid_content_length(listen):
    def handler(request, response, requests):
        response.set_header("content-length", "banana")
        response.write_head(200)
        response.end()

    server = listen(handler)
    with pytest.raises(ValueError, match="Invalid Content-Length"):
        await http_fetch(server.url, HttpRequest(method="GET", headers={}))


async def test_rejects_a_url_without_a_scheme():
    with pytest.raises(ValueError, match="Invalid MCP server URL"):
        await http_fetch("not-a-url", HttpRequest(method="GET", headers={}))

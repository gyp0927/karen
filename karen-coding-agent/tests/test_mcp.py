"""MCP tool integration tests.

Drives `McpToolManager` against a fake in-memory MCP server (an
`InMemoryTransport` pair with a scripted `tools/list`/`tools/call` handler)
so the whole bridge — connect, wrap, call, LLM-content projection — runs
without a subprocess or a socket.
"""

from __future__ import annotations

import asyncio

import pytest

from karen_ai import ImageContent, TextContent
from karen_mcp import create_in_memory_transport_pair
from karen_mcp.transports.in_memory import InMemoryTransport

from karen_coding_agent.mcp import McpServerConfig, McpToolManager


class FakeMcpServer:
    """An in-memory MCP server side: answers the client's MCP requests.

    `tools` is the list of tool dicts a `tools/list` returns; `on_call` is an
    optional `(name, arguments) -> dict` that produces the `tools/call`
    result (content blocks in the MCP wire shape). `make()` returns the
    client-side transport, ready for `McpClient.connect`.
    """

    def __init__(self, tools=None, on_call=None):
        self._tools = tools or []
        self._on_call = on_call
        self.calls: list = []

    def make(self) -> InMemoryTransport:
        client, server = create_in_memory_transport_pair()
        self._server = server
        server.on_message(self._handle)
        # Start the server side lazily: `McpClient.connect` starts the
        # client-side transport, which then starts the server side.
        return _ServerStartedWrapper(client, server)

    def _handle(self, message):
        method = message.get("method")
        msg_id = message.get("id")
        server = self._server
        if method == "initialize":
            _respond(server, msg_id, {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-mcp", "version": "0"},
            })
        elif method == "tools/list":
            _respond(server, msg_id, {"tools": self._tools})
        elif method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            self.calls.append((name, args))
            if self._on_call is not None:
                result = self._on_call(name, args)
            else:
                result = {"content": [{"type": "text", "text": f"{name} called"}]}
            _respond(server, msg_id, result)
        elif method == "ping":
            _respond(server, msg_id, {})


def _discard(task):
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass


def _respond(server: InMemoryTransport, msg_id, result):
    """Schedule an async `server.send` from a sync message handler.

    The handler runs as a sync callback fired from the peer's
    `call_soon`, inside the running loop; a fire-and-forget task is the only
    way to await `send` from there. Failures are swallowed — a server that
    cannot answer is a test-fixture problem, not an application one.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(server.send({"jsonrpc": "2.0", "id": msg_id, "result": result}))
    task.add_done_callback(_discard)


class _ServerStartedWrapper:
    """Wraps the client-side transport so the server side is started too.

    `McpClient.connect` calls `transport.start()` on whatever is handed to it;
    for an in-memory pair that only flags the client side. This wrapper adds a
    `start()` that also starts the server side, so its `send` is enabled by
    the time the client's `initialize` request arrives. All other attributes
    delegate straight through.
    """

    def __init__(self, client: InMemoryTransport, server: InMemoryTransport):
        self._client = client
        self._server = server

    async def start(self) -> None:
        await self._client.start()
        await self._server.start()

    def __getattr__(self, name):
        return getattr(self._client, name)

    async def send(self, message):
        return await self._client.send(message)

    async def close(self):
        await self._client.close()

    def on_message(self, listener):
        return self._client.on_message(listener)

    def on_error(self, listener):
        return self._client.on_error(listener)

    def on_close(self, listener):
        return self._client.on_close(listener)


async def test_wraps_a_server_tool_as_an_agent_tool():
    fake = FakeMcpServer(
        tools=[
            {
                "name": "search",
                "description": "search things",
                "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ]
    )
    config = McpServerConfig(name="fs", transport=fake.make())
    manager = McpToolManager(configs=[config])
    try:
        await manager.connect()
        tools = manager.tools()
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "mcp_fs_search"
        assert tool.label == "fs:search"
        assert tool.description == "search things"
        assert tool.parameters["type"] == "object"
    finally:
        await manager.aclose()


async def test_calls_a_server_tool_and_projects_text_content():
    def on_call(name, args):
        return {"content": [{"type": "text", "text": f"results for {args.get('q')}"}]}

    fake = FakeMcpServer(tools=[{"name": "search", "inputSchema": {"type": "object"}}], on_call=on_call)
    config = McpServerConfig(name="fs", transport=fake.make())
    manager = McpToolManager(configs=[config])
    try:
        await manager.connect()
        tool = manager.tools()[0]
        result = await tool.execute("tc1", {"q": "hello"}, None, None)
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "results for hello"
        assert fake.calls == [("search", {"q": "hello"})]
    finally:
        await manager.aclose()


async def test_projects_image_content():
    def on_call(name, args):
        return {"content": [{"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}]}

    fake = FakeMcpServer(tools=[{"name": "img", "inputSchema": {"type": "object"}}], on_call=on_call)
    config = McpServerConfig(name="img", transport=fake.make())
    manager = McpToolManager(configs=[config])
    try:
        await manager.connect()
        result = await manager.tools()[0].execute("tc", {}, None, None)
        assert isinstance(result.content[0], ImageContent)
        assert result.content[0].data == "aGVsbG8="
        assert result.content[0].mime_type == "image/png"
    finally:
        await manager.aclose()


async def test_a_failing_server_is_dropped_not_fatal():
    from karen_mcp import McpConnectionClosedError

    class _BrokenTransport(InMemoryTransport):
        async def start(self):
            raise McpConnectionClosedError("no peer")

    config = McpServerConfig(name="broken", transport=_BrokenTransport())
    good = FakeMcpServer(tools=[{"name": "ok", "inputSchema": {"type": "object"}}])
    good_config = McpServerConfig(name="ok", transport=good.make())
    manager = McpToolManager(configs=[config, good_config])
    try:
        await manager.connect()  # must not raise
        names = [t.name for t in manager.tools()]
        assert "mcp_ok_ok" in names
    finally:
        await manager.aclose()

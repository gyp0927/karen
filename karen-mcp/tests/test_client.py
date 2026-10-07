"""`McpClient` over an in-memory transport — pi's `test/client.test.ts`."""

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pytest

from karen_mcp import (
    LATEST_PROTOCOL_VERSION,
    McpAbortError,
    McpClient,
    McpClientOptions,
    McpConnectionClosedError,
    McpError,
    McpTimeoutError,
    Root,
    create_in_memory_transport_pair,
)


class Signal:
    """What the client needs of karen-ai's `AbortSignal` (pi's AbortController)."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: Any = None

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> Any:
        return self._reason if self._reason is not None else "Aborted"

    async def wait(self) -> None:
        await self._event.wait()

    def abort(self, reason: Any = None) -> None:
        self._reason = reason
        self._event.set()


class TestServer:
    """pi's `TestServer`: a message log plus one handler per method."""

    # Not a test case, despite the name pytest looks for.
    __test__ = False

    def __init__(self, transport: Any, client_transport: Any) -> None:
        self.transport = transport
        #: The client's end of the pair, for tests that inject transport-level
        #: behaviour (a stray error, a dropped connection).
        self.client_transport = client_transport
        self.messages: List[Dict[str, Any]] = []
        self.handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self.tasks: List["asyncio.Task[Any]"] = []

    def set_handler(self, method: str, handler: Callable[[Dict[str, Any]], Any]) -> None:
        self.handlers[method] = handler

    def handle_message(self, message: Dict[str, Any]) -> None:
        self.messages.append(message)
        if "id" not in message or "method" not in message:
            return
        task = asyncio.ensure_future(self._answer(message))
        self.tasks.append(task)

    async def _answer(self, request: Dict[str, Any]) -> None:
        try:
            handler = self.handlers.get(request["method"])
            if handler is None:
                raise McpError(-32601, f"Method not found: {request['method']}")
            result = handler(request)
            if inspect.isawaitable(result):
                result = await result
            await self.transport.send({"jsonrpc": "2.0", "id": request["id"], "result": result})
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            mcp_error = error if isinstance(error, McpError) else McpError(-32603, str(error))
            await self.transport.send(
                {"jsonrpc": "2.0", "id": request["id"], "error": mcp_error.to_dict()}
            )

    async def aclose(self) -> None:
        for task in self.tasks:
            task.cancel()
        self.tasks = []
        await self.transport.close()


def _initialize_result() -> Dict[str, Any]:
    return {
        "protocolVersion": LATEST_PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": True}},
        "serverInfo": {"name": "test-server", "version": "1.0.0"},
        "instructions": "Use test tools.",
    }


async def settle(ticks: int = 3) -> None:
    """What pi's tests await a macrotask for: let queued deliveries run."""
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.fixture
async def mcp():
    """A client/server pair per test; both ends are closed afterwards."""
    created: List[Any] = []

    async def create(
        client_options: Optional[McpClientOptions] = None, server_setup: Optional[Callable[[TestServer], None]] = None
    ):
        client_transport, server_transport = create_in_memory_transport_pair()
        server = TestServer(server_transport, client_transport)
        server_transport.on_message(server.handle_message)
        await server_transport.start()
        server.set_handler("initialize", lambda request: _initialize_result())
        if server_setup is not None:
            server_setup(server)
        client = McpClient(client_options or McpClientOptions(name="test-client", version="2.0.0"))
        created.append((client, server))
        return client, server

    async def connect(
        client_options: Optional[McpClientOptions] = None, server_setup: Optional[Callable[[TestServer], None]] = None
    ):
        client, server = await create(client_options, server_setup)
        await client.connect(server.client_transport)
        return client, server

    yield SimpleNamespace(create=create, connect=connect)

    for client, server in created:
        await client.close()
        await server.aclose()


async def test_initializes_the_connection_before_exposing_server_information(mcp):
    client, server = await mcp.connect()

    assert client.connection_state == "connected"
    assert client.protocol_version == LATEST_PROTOCOL_VERSION
    assert client.server_info.model_dump(exclude_none=True) == {"name": "test-server", "version": "1.0.0"}
    assert client.server_capabilities.model_dump(exclude_none=True) == {"tools": {"listChanged": True}}
    assert client.instructions == "Use test tools."
    await settle()
    assert server.messages == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "2.0.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]


async def test_paginates_tools_and_preserves_protocol_tool_definitions(mcp):
    def setup(server: TestServer) -> None:
        def handler(request: Dict[str, Any]) -> Dict[str, Any]:
            if (request.get("params") or {}).get("cursor") is None:
                return {
                    "tools": [{"name": "search", "description": "Search", "inputSchema": {"type": "object"}}],
                    "nextCursor": "page-2",
                }
            return {
                "tools": [
                    {
                        "name": "read",
                        "inputSchema": {"type": "object"},
                        "outputSchema": {"type": "object"},
                        "annotations": {"readOnlyHint": True},
                    }
                ],
                # Some servers end pagination with an empty cursor instead of omitting it.
                "nextCursor": "",
            }

        server.set_handler("tools/list", handler)

    client, _server = await mcp.connect(server_setup=setup)

    tools = await client.list_tools()
    assert [tool.model_dump(by_alias=True, exclude_none=True) for tool in tools] == [
        {"name": "search", "description": "Search", "inputSchema": {"type": "object"}},
        {
            "name": "read",
            "inputSchema": {"type": "object"},
            "outputSchema": {"type": "object"},
            "annotations": {"readOnlyHint": True},
        },
    ]


async def test_lists_and_reads_resources(mcp):
    def setup(server: TestServer) -> None:
        server.set_handler(
            "resources/list",
            lambda request: (
                {"resources": [{"uri": "file:///a", "name": "a", "mimeType": "text/plain"}], "nextCursor": "2"}
                if (request.get("params") or {}).get("cursor") is None
                else {"resources": [{"uri": "file:///b"}]}
            ),
        )
        server.set_handler(
            "resources/templates/list",
            lambda request: {"resourceTemplates": [{"uriTemplate": "repo://{owner}/{repo}", "name": "repo"}]},
        )
        server.set_handler(
            "resources/read", lambda request: {"contents": [{"uri": request["params"]["uri"], "text": "hello"}]}
        )

    client, server = await mcp.connect(server_setup=setup)

    def dump(model: Any) -> Dict[str, Any]:
        return model.model_dump(by_alias=True, exclude_none=True)

    # A missing name falls back to the URI.
    assert [dump(resource) for resource in await client.list_resources()] == [
        {"uri": "file:///a", "name": "a", "mimeType": "text/plain"},
        {"uri": "file:///b", "name": "file:///b"},
    ]
    assert [dump(template) for template in await client.list_resource_templates()] == [
        {"uriTemplate": "repo://{owner}/{repo}", "name": "repo"}
    ]
    # Single pages pass the cursor through.
    assert dump(await client.list_resources_page()) == {
        "resources": [{"uri": "file:///a", "name": "a", "mimeType": "text/plain"}],
        "nextCursor": "2",
    }
    assert dump(await client.list_resources_page("2")) == {
        "resources": [{"uri": "file:///b", "name": "file:///b"}]
    }
    assert dump(await client.read_resource("file:///a")) == {"contents": [{"uri": "file:///a", "text": "hello"}]}

    server.set_handler("resources/read", lambda request: {"contents": [{"uri": "file:///a"}]})
    with pytest.raises(McpError, match="Invalid contents in MCP resources/read result"):
        await client.read_resource("file:///a")
    server.set_handler("resources/list", lambda request: {"resources": [{"name": "no uri"}]})
    with pytest.raises(McpError, match="Invalid entry in MCP resources/list result"):
        await client.list_resources()


async def test_returns_structured_tool_content_and_surfaces_json_rpc_errors(mcp):
    def setup(server: TestServer) -> None:
        def handler(request: Dict[str, Any]) -> Dict[str, Any]:
            params = request["params"]
            if params["name"] == "fail":
                raise McpError(1234, "tool failed", {"retryable": False})
            return {"content": [{"type": "text", "text": "ok"}], "structuredContent": {"count": params["arguments"]["count"]}}

        server.set_handler("tools/call", handler)

    client, _server = await mcp.connect(server_setup=setup)

    result = await client.call_tool("count", {"count": 3})
    assert result.content == [{"type": "text", "text": "ok"}]
    assert result.structured_content == {"count": 3}
    with pytest.raises(McpError) as error:
        await client.call_tool("fail")
    assert (error.value.code, error.value.message, error.value.data) == (1234, "tool failed", {"retryable": False})


async def test_renews_the_timeout_on_progress(mcp):
    def setup(server: TestServer) -> None:
        async def handler(request: Dict[str, Any]) -> Dict[str, Any]:
            token = request["params"]["_meta"]["progressToken"]

            async def report() -> None:
                await asyncio.sleep(0.2)
                await server.transport.send(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/progress",
                        "params": {"progressToken": token, "progress": 1, "total": 2},
                    }
                )

            asyncio.ensure_future(report())
            # Later than the request's own 400ms timeout: only a progress
            # report can keep the request alive this long.
            await asyncio.sleep(0.52)
            return {"content": [{"type": "text", "text": "done"}]}

        server.set_handler("tools/call", handler)

    client, _server = await mcp.connect(server_setup=setup)

    progress: List[Any] = []
    result = await client.call_tool("slow", {}, timeout_ms=400, on_progress=progress.append)

    assert result.content == [{"type": "text", "text": "done"}]
    assert len(progress) == 1
    # `initialize` was request 1, so the progress token has to be 2.
    assert progress[0].progress_token == 2
    assert (progress[0].progress, progress[0].total) == (1, 2)


async def test_cancels_aborted_and_timed_out_requests(mcp):
    def setup(server: TestServer) -> None:
        server.set_handler("tools/call", lambda request: asyncio.Event().wait())

    client, server = await mcp.connect(server_setup=setup)

    signal = Signal()
    aborted = asyncio.ensure_future(client.call_tool("wait", {}, signal=signal))
    await settle()
    signal.abort("stop")
    with pytest.raises(McpAbortError):
        await aborted
    await settle()
    assert {
        "jsonrpc": "2.0",
        "method": "notifications/cancelled",
        "params": {"requestId": 2, "reason": "stop"},
    } in server.messages

    with pytest.raises(McpTimeoutError):
        await client.call_tool("wait", {}, timeout_ms=5)
    await settle()
    assert {
        "jsonrpc": "2.0",
        "method": "notifications/cancelled",
        "params": {"requestId": 3, "reason": "Request timed out"},
    } in server.messages


async def test_reports_transport_errors_without_failing_pending_requests(mcp):
    client, server = await mcp.create()
    await client.connect(server.client_transport)
    errors: List[BaseException] = []
    client.on_error(errors.append)
    gate = asyncio.Event()

    async def handler(request: Dict[str, Any]) -> Dict[str, Any]:
        await gate.wait()
        return {"content": []}

    server.set_handler("tools/call", handler)
    call = asyncio.ensure_future(client.call_tool("wait"))
    await settle()
    server.client_transport.emit_error(Exception("stray log line"))
    gate.set()

    assert (await call).content == []
    assert [str(error) for error in errors] == ["stray log line"]


async def test_accepts_servers_that_answer_with_an_older_protocol_version(mcp):
    def old(server: TestServer) -> None:
        server.set_handler(
            "initialize",
            lambda request: {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "serverInfo": {"name": "old-server", "version": "0.1.0"},
            },
        )

    client, _server = await mcp.connect(server_setup=old)
    assert client.protocol_version == "2024-11-05"

    def ancient(server: TestServer) -> None:
        server.set_handler(
            "initialize",
            lambda request: {
                "protocolVersion": "1999-01-01",
                "capabilities": {},
                "serverInfo": {"name": "ancient-server", "version": "0.1.0"},
            },
        )

    rejected, _rejected_server = await mcp.create(server_setup=ancient)
    with pytest.raises(Exception, match="unsupported protocol version"):
        await rejected.connect(_rejected_server.client_transport)
    assert rejected.connection_state == "closed"


async def test_defaults_missing_tool_result_content_to_an_empty_list(mcp):
    def setup(server: TestServer) -> None:
        server.set_handler("tools/call", lambda request: {"structuredContent": {"ok": True}})

    client, server = await mcp.connect(server_setup=setup)
    result = await client.call_tool("structured")
    assert result.content == []
    assert result.structured_content == {"ok": True}

    server.set_handler("tools/call", lambda request: {"content": "not a list"})
    with pytest.raises(McpError, match="Invalid MCP tools/call result"):
        await client.call_tool("broken")


async def test_does_not_send_notifications_cancelled_for_a_timed_out_initialize(mcp):
    client, server = await mcp.create(
        McpClientOptions(name="test-client", version="1.0.0", request_timeout_ms=5),
        server_setup=lambda server: server.set_handler("initialize", lambda request: asyncio.Event().wait()),
    )
    # What the client asks the transport to send, not what arrives: `connect`
    # closes the connection on failure, which would drop a queued notification
    # before it reached the server either way.
    sent: List[Dict[str, Any]] = []
    deliver = server.client_transport.send

    async def recording_send(message: Dict[str, Any]) -> None:
        sent.append(message)
        await deliver(message)

    server.client_transport.send = recording_send

    with pytest.raises(McpTimeoutError):
        await client.connect(server.client_transport)
    await settle()
    assert [message["method"] for message in sent] == ["initialize"]


async def test_notifies_close_listeners_once_when_the_transport_drops(mcp):
    client, server = await mcp.connect()
    closed: List[bool] = []
    client.on_close(lambda: closed.append(True))
    server.set_handler("tools/call", lambda request: asyncio.Event().wait())
    pending = asyncio.ensure_future(client.call_tool("wait"))
    await settle()

    await server.transport.close()

    with pytest.raises(McpConnectionClosedError, match="MCP connection closed"):
        await pending
    assert client.connection_state == "closed"
    await client.close()
    assert closed == [True]


async def test_answers_roots_list_and_dispatches_notifications(mcp):
    client, server = await mcp.connect(
        McpClientOptions(
            name="test-client",
            version="1.0.0",
            roots=[Root(uri="file:///workspace", name="workspace")],
        )
    )
    changed: List[Any] = []
    client.on_notification("notifications/tools/list_changed", changed.append)

    await server.transport.send({"jsonrpc": "2.0", "id": "roots", "method": "roots/list"})
    await server.transport.send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
    await settle(5)

    assert {
        "jsonrpc": "2.0",
        "id": "roots",
        "result": {"roots": [{"uri": "file:///workspace", "name": "workspace"}]},
    } in server.messages
    assert changed == [None]

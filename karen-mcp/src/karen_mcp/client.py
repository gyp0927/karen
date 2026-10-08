"""Transport-neutral MCP client (pi's `client.ts`).

The client owns everything that is not framing: initialization, request
correlation, timeouts (renewed by progress), cancellation, the server's own
requests to us, and the protocol-level helpers (`tools/list`, `tools/call`,
`resources/*`) with pi's exact validation and leniency — a server that omits
`content`, ends pagination with `""`, or drops a resource's `name` is handled
rather than rejected.

The helpers also parse their results into the pydantic protocol models, so a
known member with the wrong type is rejected where pi's plain casts would pass
it through; unknown members always pass through (`extra="allow"`).
"""

from __future__ import annotations

import asyncio
import inspect
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pydantic import ValidationError

from .cancellation import Signal
from .protocol.content import CallToolResult, ContentBlock
from .protocol.jsonrpc import (
    JSON_RPC_ERROR_CODES,
    JsonRpcId,
    McpAbortError,
    McpConnectionClosedError,
    McpError,
    McpTimeoutError,
    is_json_rpc_id,
    is_json_rpc_notification,
    is_json_rpc_request,
    is_json_rpc_response,
    is_object,
    to_error,
)
from .protocol.types import (
    LATEST_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    ClientCapabilities,
    Implementation,
    InitializeResult,
    ListResourceTemplatesResult,
    ListResourcesResult,
    ProgressNotification,
    ReadResourceResult,
    Resource,
    ResourceTemplate,
    Root,
    ServerCapabilities,
    Tool,
)
from .transports.transport import McpTransport

__all__ = [
    "DEFAULT_REQUEST_TIMEOUT_MS",
    "MAX_LIST_PAGES",
    "McpClient",
    "McpClientOptions",
    "RequestContext",
]

DEFAULT_REQUEST_TIMEOUT_MS = 30_000
MAX_LIST_PAGES = 1_000

ClientState = str  # "idle" | "connecting" | "connected" | "closed"

NotificationListener = Callable[[Any], None]
ErrorListener = Callable[[BaseException], None]
CloseListener = Callable[[], None]
RequestHandler = Callable[[Any, "RequestContext"], Any]
RootsSource = Union[Sequence[Root], Callable[[], Union[Sequence[Root], Awaitable[Sequence[Root]]]]]


@dataclass
class RequestContext:
    """What a request handler gets besides the params (pi's `{ signal }`)."""

    signal: Any


@dataclass
class _PendingRequest:
    future: "asyncio.Future[Any]"
    timeout_ms: float
    timer: Optional[asyncio.TimerHandle] = None
    signal: Any = None
    abort_waiter: Optional["asyncio.Task[None]"] = None
    cancellable: bool = True
    on_progress: Optional[Callable[[ProgressNotification], None]] = None
    progress_token: Optional[JsonRpcId] = None


@dataclass
class McpClientOptions:
    """pi's `McpClientOptions`."""

    name: str
    version: str
    title: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None
    protocol_version: Optional[str] = None
    request_timeout_ms: Optional[float] = None
    roots: Optional[RootsSource] = None


def _invalid(message: str) -> McpError:
    return McpError(JSON_RPC_ERROR_CODES["invalidRequest"], message)


def _validate_initialize_result(value: Any) -> Dict[str, Any]:
    if (
        not is_object(value)
        or not isinstance(value.get("protocolVersion"), str)
        or not is_object(value.get("capabilities"))
        or not is_object(value.get("serverInfo"))
        or not isinstance(value["serverInfo"].get("name"), str)
        or not isinstance(value["serverInfo"].get("version"), str)
        or ("instructions" in value and not isinstance(value["instructions"], str))
    ):
        raise _invalid("Invalid MCP initialize result")
    return value


def _is_tool(item: Dict[str, Any]) -> bool:
    return isinstance(item.get("name"), str) and is_object(item.get("inputSchema"))


# `name` is required by the spec, but some servers omit it; the URI stands in.
def _is_resource(item: Dict[str, Any]) -> bool:
    return isinstance(item.get("uri"), str) and ("name" not in item or isinstance(item["name"], str))


def _is_resource_template(item: Dict[str, Any]) -> bool:
    return isinstance(item.get("uriTemplate"), str) and (
        "name" not in item or isinstance(item["name"], str)
    )


def _validate_list_page(
    method: str,
    key: str,
    value: Any,
    is_item: Callable[[Dict[str, Any]], bool],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """One page of a paginated list: the items under `key`, each checked by `is_item`."""
    items = value.get(key) if is_object(value) else None
    if not is_object(value) or not isinstance(items, list):
        raise _invalid(f"Invalid MCP {method} result")
    for item in items:
        if not is_object(item) or not is_item(item):
            raise _invalid(f"Invalid entry in MCP {method} result")
    # Some servers end pagination with `null` or `""` instead of omitting the cursor.
    next_cursor = value.get("nextCursor")
    if next_cursor is None or next_cursor == "":
        next_cursor = None
    if next_cursor is not None and not isinstance(next_cursor, str):
        raise _invalid(f"Invalid MCP {method} cursor")
    return items, next_cursor


def _validate_read_resource_result(value: Any) -> Dict[str, Any]:
    if not is_object(value) or not isinstance(value.get("contents"), list):
        raise _invalid("Invalid MCP resources/read result")
    for contents in value["contents"]:
        if (
            not is_object(contents)
            or not isinstance(contents.get("uri"), str)
            or (not isinstance(contents.get("text"), str) and not isinstance(contents.get("blob"), str))
        ):
            raise _invalid("Invalid contents in MCP resources/read result")
    return value


def _validate_call_tool_result(value: Any) -> Dict[str, Any]:
    if not is_object(value) or ("content" in value and not isinstance(value["content"], list)):
        raise _invalid("Invalid MCP tools/call result")
    # `structuredContent` is an object when present; an explicit `null` is not
    # one, exactly as in pi.
    if "structuredContent" in value and not is_object(value["structuredContent"]):
        raise _invalid("Invalid MCP tools/call structured content")
    if "content" not in value:
        value = {**value, "content": []}
    return value


class McpClient:
    """A client for one MCP server over one transport."""

    def __init__(self, options: McpClientOptions) -> None:
        self.options = options
        self._state: ClientState = "idle"
        self._transport: Optional[McpTransport] = None
        self._next_request_id = 1
        self._server_info: Optional[Implementation] = None
        self._server_capabilities: Optional[ServerCapabilities] = None
        self._instructions: Optional[str] = None
        self._protocol_version: Optional[str] = None
        self._pending: Dict[JsonRpcId, _PendingRequest] = {}
        self._progress_requests: Dict[JsonRpcId, JsonRpcId] = {}
        self._incoming: Dict[JsonRpcId, Tuple["asyncio.Task[Any]", Signal]] = {}
        self._request_handlers: Dict[str, RequestHandler] = {}
        self._notification_listeners: Dict[str, List[NotificationListener]] = {}
        self._error_listeners: List[ErrorListener] = []
        self._close_listeners: List[CloseListener] = []
        self._disposers: List[Callable[[], None]] = []

        self._request_handlers["ping"] = lambda params, context: {}
        if options.roots is not None:
            self._request_handlers["roots/list"] = self._handle_roots_list

    # -- state ---------------------------------------------------------------

    @property
    def connection_state(self) -> ClientState:
        return self._state

    @property
    def server_info(self) -> Optional[Implementation]:
        return self._server_info

    @property
    def server_capabilities(self) -> Optional[ServerCapabilities]:
        return self._server_capabilities

    @property
    def instructions(self) -> Optional[str]:
        return self._instructions

    @property
    def protocol_version(self) -> Optional[str]:
        return self._protocol_version

    # -- lifecycle -----------------------------------------------------------

    async def connect(self, transport: McpTransport) -> InitializeResult:
        if self._state != "idle":
            raise RuntimeError(f"Cannot connect MCP client in {self._state} state")
        self._state = "connecting"
        self._transport = transport
        self._disposers = [
            transport.on_message(self._handle_message),
            # Transport errors are reported only. Pending requests fail when the
            # transport closes.
            transport.on_error(self._emit_error),
            transport.on_close(self._handle_transport_close),
        ]

        try:
            await transport.start()
            capabilities: Dict[str, Any] = dict(self.options.capabilities or {})
            if self.options.roots is not None and capabilities.get("roots") is None:
                capabilities["roots"] = {}
            client_info: Dict[str, Any] = {"name": self.options.name, "version": self.options.version}
            if self.options.title is not None:
                client_info["title"] = self.options.title
            raw = _validate_initialize_result(
                await self._request_internal(
                    "initialize",
                    {
                        "protocolVersion": self.options.protocol_version or LATEST_PROTOCOL_VERSION,
                        "capabilities": capabilities,
                        "clientInfo": client_info,
                    },
                    signal=None,
                    timeout_ms=None,
                    on_progress=None,
                    allow_connecting=True,
                )
            )
            if raw["protocolVersion"] not in SUPPORTED_PROTOCOL_VERSIONS:
                raise Exception(f"MCP server selected unsupported protocol version {raw['protocolVersion']}")
            result = InitializeResult.model_validate(raw)
            self._protocol_version = result.protocol_version
            self._server_info = result.server_info
            self._server_capabilities = result.capabilities
            self._instructions = result.instructions
            transport.set_protocol_version(result.protocol_version)
            await self._notify_internal("notifications/initialized", None, True)
            self._state = "connected"
            return result
        except BaseException:
            # A failed close must not replace the error the caller is owed.
            try:
                await self.close()
            except BaseException:
                pass
            raise

    async def close(self) -> None:
        transport = self._transport
        self._transport = None
        self._dispose_transport_listeners()
        self._mark_closed(McpConnectionClosedError())
        if transport is not None:
            await transport.close()

    # -- requests ------------------------------------------------------------

    async def request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        signal: Any = None,
        timeout_ms: Optional[float] = None,
        on_progress: Optional[Callable[[ProgressNotification], None]] = None,
    ) -> Any:
        return await self._request_internal(method, params, signal, timeout_ms, on_progress, False)

    async def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        await self._notify_internal(method, params, False)

    def set_request_handler(self, method: str, handler: RequestHandler) -> Callable[[], None]:
        self._request_handlers[method] = handler
        return lambda: self._discard_handler(method, handler)

    def on_notification(self, method: str, listener: NotificationListener) -> Callable[[], None]:
        listeners = self._notification_listeners.setdefault(method, [])
        listeners.append(listener)

        def unsubscribe() -> None:
            try:
                listeners.remove(listener)
            except ValueError:
                return
            if not listeners:
                self._notification_listeners.pop(method, None)

        return unsubscribe

    def on_error(self, listener: ErrorListener) -> Callable[[], None]:
        self._error_listeners.append(listener)
        return lambda: self._discard_listener(self._error_listeners, listener)

    def on_close(self, listener: CloseListener) -> Callable[[], None]:
        """Called once when the connection closes, whether the transport dropped
        or `close()` was called."""
        self._close_listeners.append(listener)
        return lambda: self._discard_listener(self._close_listeners, listener)

    # -- protocol helpers ----------------------------------------------------

    async def ping(self, **options: Any) -> None:
        await self.request("ping", None, **options)

    async def list_tools(self, **options: Any) -> List[Tool]:
        items = await self._list_all("tools/list", "tools", _is_tool, options)
        return [self._model(Tool, item, "Invalid entry in MCP tools/list result") for item in items]

    async def list_resources(self, **options: Any) -> List[Resource]:
        """Every resource, following `nextCursor` through all pages."""
        items = await self._list_all("resources/list", "resources", _is_resource, options)
        return [self._resource(item) for item in items]

    async def list_resources_page(self, cursor: Optional[str] = None, **options: Any) -> ListResourcesResult:
        """One page of resources, starting at `cursor`."""
        items, next_cursor = await self._list_page("resources/list", "resources", _is_resource, cursor, options)
        return ListResourcesResult(
            resources=[self._resource(item) for item in items], nextCursor=next_cursor
        )

    async def list_resource_templates(self, **options: Any) -> List[ResourceTemplate]:
        """Every resource template, following `nextCursor` through all pages."""
        items = await self._list_all(
            "resources/templates/list", "resourceTemplates", _is_resource_template, options
        )
        return [self._resource_template(item) for item in items]

    async def list_resource_templates_page(
        self, cursor: Optional[str] = None, **options: Any
    ) -> ListResourceTemplatesResult:
        """One page of resource templates, starting at `cursor`."""
        items, next_cursor = await self._list_page(
            "resources/templates/list", "resourceTemplates", _is_resource_template, cursor, options
        )
        return ListResourceTemplatesResult(
            resourceTemplates=[self._resource_template(item) for item in items], nextCursor=next_cursor
        )

    async def read_resource(self, uri: str, **options: Any) -> ReadResourceResult:
        return ReadResourceResult.model_validate(
            _validate_read_resource_result(await self.request("resources/read", {"uri": uri}, **options))
        )

    async def call_tool(
        self, name: str, args: Optional[Dict[str, Any]] = None, **options: Any
    ) -> CallToolResult:
        params: Dict[str, Any] = {"name": name}
        if args is not None:
            params["arguments"] = args
        return CallToolResult.model_validate(
            _validate_call_tool_result(await self.request("tools/call", params, **options))
        )

    async def _list_page(
        self,
        method: str,
        key: str,
        is_item: Callable[[Dict[str, Any]], bool],
        cursor: Optional[str],
        options: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        params = None if cursor is None else {"cursor": cursor}
        return _validate_list_page(method, key, await self.request(method, params, **options), is_item)

    async def _list_all(
        self,
        method: str,
        key: str,
        is_item: Callable[[Dict[str, Any]], bool],
        options: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Every item of a paginated list method."""
        items: List[Dict[str, Any]] = []
        cursors: set = set()
        cursor: Optional[str] = None
        for _page in range(MAX_LIST_PAGES):
            page_items, next_cursor = await self._list_page(method, key, is_item, cursor, options)
            items.extend(page_items)
            if next_cursor is None:
                return items
            if next_cursor in cursors:
                raise Exception(f"MCP {method} returned duplicate cursor: {next_cursor}")
            cursors.add(next_cursor)
            cursor = next_cursor
        raise Exception(f"MCP {method} exceeded {MAX_LIST_PAGES} pages")

    # -- internals -----------------------------------------------------------

    async def _handle_roots_list(self, params: Any, context: RequestContext) -> Dict[str, Any]:
        roots = self.options.roots
        if callable(roots):
            roots = roots()
            if inspect.isawaitable(roots):
                roots = await roots
        return {"roots": [_dump_root(root) for root in (roots or [])]}

    async def _request_internal(
        self,
        method: str,
        params: Optional[Dict[str, Any]],
        signal: Any,
        timeout_ms: Optional[float],
        on_progress: Optional[Callable[[ProgressNotification], None]],
        allow_connecting: bool,
    ) -> Any:
        transport = self._require_transport(allow_connecting)
        if signal is not None and signal.aborted:
            raise McpAbortError()
        loop = asyncio.get_running_loop()
        request_id = self._next_request_id
        self._next_request_id += 1
        progress_token: Optional[JsonRpcId] = request_id if on_progress is not None else None
        request_params = params
        if progress_token is not None:
            meta = dict(params.get("_meta")) if is_object((params or {}).get("_meta")) else {}
            meta["progressToken"] = progress_token
            request_params = {**(params or {}), "_meta": meta}
        message: Dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if request_params is not None:
            message["params"] = request_params

        entry = _PendingRequest(
            future=loop.create_future(),
            timeout_ms=(
                timeout_ms
                if timeout_ms is not None
                else (
                    self.options.request_timeout_ms
                    if self.options.request_timeout_ms is not None
                    else DEFAULT_REQUEST_TIMEOUT_MS
                )
            ),
            signal=signal,
            # The spec forbids cancelling `initialize`.
            cancellable=method != "initialize",
            on_progress=on_progress,
            progress_token=progress_token,
        )
        self._pending[request_id] = entry
        if progress_token is not None:
            self._progress_requests[progress_token] = request_id
        if signal is not None:
            entry.abort_waiter = asyncio.ensure_future(
                self._await_abort(signal, request_id, entry.cancellable)
            )
        self._arm_timeout(request_id, entry)
        send_task = asyncio.ensure_future(transport.send(message))
        send_task.add_done_callback(lambda task: self._on_sent(task, request_id))
        return await entry.future

    async def _await_abort(self, signal: Any, request_id: JsonRpcId, cancellable: bool) -> None:
        await signal.wait()
        reason = str(getattr(signal, "reason", "") or "Aborted")
        self._cancel_pending(request_id, McpAbortError(), cancellable, reason)

    def _on_sent(self, task: "asyncio.Task[Any]", request_id: JsonRpcId) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._cancel_pending(request_id, error, False)

    async def _notify_internal(
        self, method: str, params: Optional[Dict[str, Any]], allow_connecting: bool
    ) -> None:
        message: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        await self._require_transport(allow_connecting).send(message)

    def _require_transport(self, allow_connecting: bool) -> McpTransport:
        if self._transport is not None and (
            self._state == "connected" or (allow_connecting and self._state == "connecting")
        ):
            return self._transport
        raise McpConnectionClosedError(f"MCP client is {self._state}")

    def _handle_message(self, message: Dict[str, Any]) -> None:
        if is_json_rpc_response(message):
            self._handle_response(message)
            return
        if is_json_rpc_request(message):
            asyncio.ensure_future(self._handle_request(message))
            return
        if is_json_rpc_notification(message):
            self._handle_notification(message.get("method"), message.get("params"))
            return
        self._emit_error(McpError(JSON_RPC_ERROR_CODES["invalidRequest"], "Received invalid JSON-RPC message"))

    def _handle_response(self, message: Dict[str, Any]) -> None:
        entry = self._pending.get(message["id"])
        if entry is None:
            self._emit_error(Exception(f"Received response for unknown MCP request {message['id']}"))
            return
        self._remove_pending(message["id"], entry)
        if entry.future.cancelled():
            return
        if "error" in message:
            error = message["error"]
            entry.future.set_exception(McpError(error["code"], error["message"], error.get("data")))
        else:
            entry.future.set_result(message.get("result"))

    async def _handle_request(self, message: Dict[str, Any]) -> None:
        transport = self._transport
        if transport is None:
            return
        handler = self._request_handlers.get(message["method"])
        if handler is None:
            await self._send_quietly(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {
                        "code": JSON_RPC_ERROR_CODES["methodNotFound"],
                        "message": f"Method not found: {message['method']}",
                    },
                }
            )
            return
        signal = Signal()
        task = asyncio.current_task()
        if task is not None:
            self._incoming[message["id"]] = (task, signal)
        try:
            result = handler(message.get("params"), RequestContext(signal=signal))
            if inspect.isawaitable(result):
                result = await result
            await transport.send({"jsonrpc": "2.0", "id": message["id"], "result": result if result is not None else {}})
        except asyncio.CancelledError:
            # The peer cancelled the request, or the client is closing: it gave
            # up on the response, so there is nothing to answer.
            return
        except BaseException as error:
            response_error = (
                error.to_dict()
                if isinstance(error, McpError)
                else {"code": JSON_RPC_ERROR_CODES["internalError"], "message": str(to_error(error))}
            )
            await self._send_quietly({"jsonrpc": "2.0", "id": message["id"], "error": response_error})
        finally:
            self._incoming.pop(message["id"], None)

    def _handle_notification(self, method: Optional[str], params: Any) -> None:
        if method == "notifications/progress":
            self._handle_progress(params)
        elif method == "notifications/cancelled":
            self._handle_cancelled(params)
        for listener in list(self._notification_listeners.get(method or "", [])):
            try:
                listener(params)
            except BaseException as error:
                self._emit_error(error)

    def _handle_progress(self, params: Any) -> None:
        if not is_object(params) or not is_json_rpc_id(params.get("progressToken")) or not isinstance(
            params.get("progress"), (int, float)
        ):
            return
        request_id = self._progress_requests.get(params["progressToken"])
        entry = self._pending.get(request_id) if request_id is not None else None
        if request_id is None or entry is None:
            return
        self._arm_timeout(request_id, entry)
        if entry.on_progress is None:
            return
        try:
            entry.on_progress(ProgressNotification.model_validate(params))
        except BaseException as error:
            self._emit_error(error)

    def _handle_cancelled(self, params: Any) -> None:
        if not is_object(params) or not is_json_rpc_id(params.get("requestId")):
            return
        incoming = self._incoming.get(params["requestId"])
        if incoming is None:
            return
        task, signal = incoming
        signal.abort(McpAbortError(str(params.get("reason") or "Cancelled")))
        task.cancel()

    def _arm_timeout(self, request_id: JsonRpcId, entry: _PendingRequest) -> None:
        if entry.timer is not None:
            entry.timer.cancel()
        if not math.isfinite(entry.timeout_ms) or entry.timeout_ms <= 0:
            return
        entry.timer = asyncio.get_running_loop().call_later(
            entry.timeout_ms / 1000.0,
            self._cancel_pending,
            request_id,
            McpTimeoutError(entry.timeout_ms),
            entry.cancellable,
            "Request timed out",
        )

    def _cancel_pending(
        self, request_id: JsonRpcId, error: BaseException, notify_server: bool, reason: Optional[str] = None
    ) -> None:
        entry = self._pending.get(request_id)
        if entry is None:
            return
        self._remove_pending(request_id, entry)
        if not entry.future.done():
            entry.future.set_exception(error)
        if notify_server and self._transport is not None:
            params: Dict[str, Any] = {"requestId": request_id}
            if reason:
                params["reason"] = reason
            task = asyncio.ensure_future(
                self._transport.send(
                    {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": params}
                )
            )
            task.add_done_callback(self._on_notified)

    def _remove_pending(self, request_id: JsonRpcId, entry: _PendingRequest) -> None:
        self._pending.pop(request_id, None)
        if entry.timer is not None:
            entry.timer.cancel()
            entry.timer = None
        if entry.progress_token is not None:
            self._progress_requests.pop(entry.progress_token, None)
        if entry.abort_waiter is not None:
            entry.abort_waiter.cancel()
            entry.abort_waiter = None

    def _reject_pending(self, error: BaseException) -> None:
        for request_id, entry in list(self._pending.items()):
            self._remove_pending(request_id, entry)
            if not entry.future.done():
                entry.future.set_exception(error)

    def _handle_transport_close(self) -> None:
        self._mark_closed(McpConnectionClosedError())

    def _mark_closed(self, error: BaseException) -> None:
        """Idempotent: rejects in-flight requests, aborts server requests we are
        serving, and flips the state."""
        was_closed = self._state == "closed"
        self._state = "closed"
        self._reject_pending(error)
        for task, signal in list(self._incoming.values()):
            signal.abort(error)
            task.cancel()
        self._incoming.clear()
        if was_closed:
            return
        for listener in list(self._close_listeners):
            try:
                listener()
            except BaseException as listener_error:
                self._emit_error(listener_error)

    def _emit_error(self, error: Any) -> None:
        normalized = to_error(error)
        for listener in list(self._error_listeners):
            listener(normalized)

    async def _send_quietly(self, message: Dict[str, Any]) -> None:
        transport = self._transport
        if transport is None:
            return
        try:
            await transport.send(message)
        except BaseException as error:
            self._emit_error(error)

    def _on_notified(self, task: "asyncio.Task[Any]") -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._emit_error(error)

    def _dispose_transport_listeners(self) -> None:
        for dispose in self._disposers:
            dispose()
        self._disposers = []

    @staticmethod
    def _model(model_type: Any, value: Dict[str, Any], message: str) -> Any:
        try:
            return model_type.model_validate(value)
        except ValidationError as error:
            raise _invalid(message) from error

    def _resource(self, item: Dict[str, Any]) -> Resource:
        filled = item if isinstance(item.get("name"), str) else {**item, "name": item["uri"]}
        return self._model(Resource, filled, "Invalid entry in MCP resources/list result")

    def _resource_template(self, item: Dict[str, Any]) -> ResourceTemplate:
        filled = item if isinstance(item.get("name"), str) else {**item, "name": item["uriTemplate"]}
        return self._model(ResourceTemplate, filled, "Invalid entry in MCP resources/templates/list result")

    def _discard_handler(self, method: str, handler: RequestHandler) -> None:
        if self._request_handlers.get(method) is handler:
            self._request_handlers.pop(method, None)

    @staticmethod
    def _discard_listener(listeners: List[Any], listener: Any) -> None:
        try:
            listeners.remove(listener)
        except ValueError:
            pass


def _dump_root(root: Any) -> Dict[str, Any]:
    if isinstance(root, Root):
        return {"uri": root.uri, **({"name": root.name} if root.name is not None else {})}
    return dict(root)

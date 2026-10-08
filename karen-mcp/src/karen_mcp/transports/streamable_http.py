"""Streamable HTTP transport (pi's `transports/streamable-http.ts`).

One POST per message, and the answer is either JSON or an SSE stream. The
server may hold a server-to-client GET stream open, which the transport opens
after initialization and keeps open, resuming either stream with `Last-Event-ID`
when the server drops it — the reason a request whose stream ends without an
answer can still succeed.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple

from ..auth_provider import AuthProvider, McpFetch, UnauthorizedContext
from ..cancellation import Signal
from ..protocol.jsonrpc import (
    JSON_RPC_ERROR_CODES,
    JsonRpcId,
    McpConnectionClosedError,
    is_json_rpc_request,
    is_json_rpc_response,
    parse_json_rpc_message,
    to_error,
)
from .http_client import Headers, HttpRequest, HttpResponse, http_fetch
from .transport import DEFAULT_MAX_MESSAGE_BYTES, McpTransport, TransportEvents

__all__ = [
    "ConsumeSseOptions",
    "McpAuthRequiredError",
    "McpHttpError",
    "McpSessionExpiredError",
    "SseEvent",
    "StreamableHttpReconnectOptions",
    "StreamableHttpTransport",
    "StreamableHttpTransportOptions",
    "consume_sse_stream",
]

MAX_ERROR_BODY_BYTES = 8 * 1024
ERROR_MESSAGE_BODY_CHARS = 500
DEFAULT_RECONNECT_INITIAL_DELAY_MS = 1_000
DEFAULT_RECONNECT_MAX_DELAY_MS = 30_000
DEFAULT_RECONNECT_MAX_RETRIES = 5


@dataclass
class SseEvent:
    """One dispatched SSE event: the joined `data` lines plus `event`/`id`."""

    data: str
    event: Optional[str] = None
    id: Optional[str] = None


@dataclass
class ConsumeSseOptions:
    on_event: Callable[[SseEvent], None]
    #: Called for every `id` field, including events without data (resumption
    #: priming events).
    on_id: Optional[Callable[[str], None]] = None
    #: Called for every valid `retry` field, in milliseconds.
    on_retry: Optional[Callable[[int], None]] = None
    max_event_bytes: int = DEFAULT_MAX_MESSAGE_BYTES


async def consume_sse_stream(chunks: AsyncIterable[bytes], options: ConsumeSseOptions) -> None:
    """Parse an SSE byte stream, dispatching events as they are terminated.

    The byte stream is closed on the way out, so a caller cannot leave a
    response body half-read by abandoning the parse.
    """
    stream = chunks.__aiter__()
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buffered = ""
    max_event_bytes = options.max_event_bytes
    state: Dict[str, Any] = {"event": None, "id": None, "lines": [], "bytes": 0}

    def dispatch() -> None:
        lines: List[str] = state["lines"]
        if not lines:
            state["event"] = None
            state["id"] = None
            return
        event = SseEvent(data="\n".join(lines))
        if state["event"]:
            event.event = state["event"]
        if state["id"]:
            event.id = state["id"]
        options.on_event(event)
        state["event"] = None
        state["id"] = None
        state["lines"] = []
        state["bytes"] = 0

    def process_line(raw_line: str) -> None:
        line = raw_line[:-1] if raw_line.endswith("\r") else raw_line
        if line == "":
            dispatch()
            return
        if line.startswith(":"):
            return
        colon = line.find(":")
        field_name = line if colon < 0 else line[:colon]
        value = "" if colon < 0 else line[colon + 1 :]
        if value.startswith(" "):
            value = value[1:]
        if field_name == "data":
            # Bytes of the pending event's data, including the "\n" joins, so
            # an event streamed as many short `data:` lines without a
            # terminating blank line cannot grow without bound.
            state["bytes"] += len(value.encode("utf-8")) + (1 if state["lines"] else 0)
            if state["bytes"] > max_event_bytes:
                raise ValueError(f"MCP SSE event exceeds {max_event_bytes} bytes")
            state["lines"].append(value)
        elif field_name == "event":
            state["event"] = value
        elif field_name == "id" and "\0" not in value:
            state["id"] = value
            if options.on_id is not None:
                options.on_id(value)
        elif field_name == "retry" and value.isascii() and value.isdigit():
            if options.on_retry is not None:
                options.on_retry(int(value))

    try:
        while True:
            try:
                chunk = await stream.__anext__()
            except StopAsyncIteration:
                break
            buffered += decoder.decode(chunk)
            newline = buffered.find("\n")
            while newline >= 0:
                process_line(buffered[:newline])
                buffered = buffered[newline + 1 :]
                newline = buffered.find("\n")
            if len(buffered.encode("utf-8")) > max_event_bytes:
                raise ValueError(f"MCP SSE event exceeds {max_event_bytes} bytes")
        buffered += decoder.decode(b"", True)
        if buffered:
            process_line(buffered)
        dispatch()
    finally:
        closer = getattr(stream, "aclose", None)
        if closer is not None:
            await closer()


@dataclass
class StreamableHttpReconnectOptions:
    """Reconnection of dropped SSE streams."""

    #: Delay before the first attempt, unless the server sent a `retry` field.
    initial_delay_ms: int = DEFAULT_RECONNECT_INITIAL_DELAY_MS
    #: Upper bound for the exponential backoff.
    max_delay_ms: int = DEFAULT_RECONNECT_MAX_DELAY_MS
    #: Consecutive failed attempts before giving up on a stream.
    max_retries: int = DEFAULT_RECONNECT_MAX_RETRIES


@dataclass
class StreamableHttpTransportOptions:
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    #: Defaults to this package's own HTTP client.
    fetch: Optional[McpFetch] = None
    #: Open the server-to-client GET stream after initialization.
    open_get_stream: bool = True
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES
    auth_provider: Optional[AuthProvider] = None
    reconnect: Optional[StreamableHttpReconnectOptions] = None


class McpHttpError(Exception):
    def __init__(self, status: int, message: str, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class McpAuthRequiredError(McpHttpError):
    def __init__(self, response: HttpResponse, body: str = "") -> None:
        super().__init__(401, "MCP server requires authentication", body)
        self.www_authenticate = response.header("www-authenticate")


class McpSessionExpiredError(McpHttpError):
    def __init__(self, body: str = "") -> None:
        super().__init__(404, "MCP session expired", body)


def _content_type(response: HttpResponse) -> Optional[str]:
    value = response.header("content-type")
    return value.split(";", 1)[0].strip().lower() if value else None


def _needs_authorization(response: HttpResponse) -> bool:
    """401, or 403 with an `insufficient_scope` bearer challenge (step-up)."""
    if response.status == 401:
        return True
    if response.status != 403:
        return False
    challenge = response.header("www-authenticate") or ""
    return re.search(r'(?:^|[\s,])error="?insufficient_scope"?', challenge, re.IGNORECASE) is not None


def _is_transient_status(status: int) -> bool:
    """Statuses worth retrying when a stream fails to (re)open."""
    return status in (408, 429) or status >= 500


def _describe_http_failure(status: int, body: str) -> str:
    text = body.strip()
    snippet = text[: ERROR_MESSAGE_BODY_CHARS - 3] + "..." if len(text) > ERROR_MESSAGE_BODY_CHARS else text
    return f"MCP HTTP request failed with status {status}" + (f": {snippet}" if snippet else "")


async def _discard(response: HttpResponse) -> None:
    await response.close()


@dataclass
class _StreamCursor:
    last_event_id: Optional[str] = None
    retry_ms: Optional[int] = None
    #: Whether the stream delivered any event since it was (re)opened.
    received: bool = False


class StreamableHttpTransport(TransportEvents, McpTransport):
    def __init__(self, options: StreamableHttpTransportOptions) -> None:
        super().__init__()
        self.options = options
        self.url = options.url
        self._fetch: McpFetch = options.fetch or http_fetch
        self._signal = Signal()
        self._started = False
        self._closed = False
        self._session_id_value: Optional[str] = None
        self._protocol_version: Optional[str] = None
        self._get_stream_started = False
        self._tasks: Set["asyncio.Task[Any]"] = set()

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id_value

    @property
    def protocol_version(self) -> Optional[str]:
        return self._protocol_version

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("MCP Streamable HTTP transport already started")
        if self._closed:
            raise McpConnectionClosedError()
        self._started = True

    def set_protocol_version(self, version: str) -> None:
        self._protocol_version = version

    async def send(self, message: Dict[str, Any]) -> None:
        if not self._started or self._closed:
            raise McpConnectionClosedError()
        response = await self._authorized_fetch(
            "POST",
            {"accept": "application/json, text/event-stream", "content-type": "application/json"},
            json.dumps(message, ensure_ascii=False).encode("utf-8"),
        )
        await self._check_response(response)
        self._capture_session(response)

        if not is_json_rpc_request(message):
            # Notifications and responses are acknowledged with 202 and carry
            # no reply; ignore any body.
            await _discard(response)
            # The server-to-client stream may only open once the session is
            # initialized.
            if message.get("method") == "notifications/initialized":
                self._start_get_stream()
            return
        if response.status in (202, 204):
            await _discard(response)
            raise McpHttpError(
                response.status, f"MCP server accepted request {message['method']} without a response"
            )
        kind = _content_type(response)
        if kind == "application/json":
            body = await response.json()
            for item in body if isinstance(body, list) else [body]:
                self.emit_message(parse_json_rpc_message(item))
            return
        if kind == "text/event-stream" and response.has_body:
            self._spawn(self._consume_response_stream(response, message["id"]))
            return
        await _discard(response)
        raise McpHttpError(response.status, f"Unsupported MCP response content type: {kind or 'missing'}")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._signal.abort()
        tasks, self._tasks = list(self._tasks), set()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            if self._started and self._session_id_value is not None:
                try:
                    headers, _token = await self._headers()
                    await asyncio.wait_for(self._delete_session(headers), 1.0)
                except Exception:
                    # Resolving auth headers failed, or the server did not
                    # answer: the session expires on its own.
                    pass
        finally:
            self.emit_close()

    # -- requests ------------------------------------------------------------

    async def _authorized_fetch(
        self, method: str, headers: Dict[str, str], body: Optional[bytes] = None
    ) -> HttpResponse:
        """Fetch with auth headers. A 401 (or a 403 asking for more scope) is
        handed to the auth provider once, and the request is retried with
        whatever credentials it left behind."""
        provider = self.options.auth_provider
        on_unauthorized = getattr(provider, "on_unauthorized", None) if provider is not None else None
        attempt = 0
        while True:
            prepared, token = await self._headers(headers)
            response = await self._fetch(
                self.url,
                HttpRequest(method=method, headers=prepared, body=body, signal=self._signal),
            )
            if attempt > 0 or on_unauthorized is None or not _needs_authorization(response):
                return response
            try:
                await on_unauthorized(
                    UnauthorizedContext(
                        response=response, server_url=self.url, fetch=self._fetch, token=token
                    )
                )
            finally:
                await _discard(response)
            attempt += 1

    async def _headers(self, extra: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, str], Optional[str]]:
        # Case-insensitive, like pi's `Headers`: a caller header that differs
        # only in case from a built-in one is replaced, not duplicated.
        headers = Headers(self.options.headers)
        for name, value in (extra or {}).items():
            headers.add(name, value)
        if self._session_id_value:
            headers.add("Mcp-Session-Id", self._session_id_value)
        if self._protocol_version:
            headers.add("MCP-Protocol-Version", self._protocol_version)
        provider = self.options.auth_provider
        token = await provider.token() if provider is not None else None
        if token:
            headers.add("Authorization", f"Bearer {token}")
        return dict(headers.items()), token

    def _capture_session(self, response: HttpResponse) -> None:
        session_id = response.header("mcp-session-id")
        if session_id:
            self._session_id_value = session_id

    async def _check_response(self, response: HttpResponse) -> None:
        if response.ok:
            return
        body = ""
        if response.has_body:
            try:
                body = await response.text(MAX_ERROR_BODY_BYTES)
            except Exception:
                body = ""
        if response.status == 401:
            raise McpAuthRequiredError(response, body)
        if response.status == 404 and self._session_id_value is not None:
            raise McpSessionExpiredError(body)
        raise McpHttpError(response.status, _describe_http_failure(response.status, body), body)

    async def _delete_session(self, headers: Dict[str, str]) -> None:
        response = await self._fetch(self.url, HttpRequest(method="DELETE", headers=headers))
        await _discard(response)

    # -- streams -------------------------------------------------------------

    async def _consume_sse(
        self,
        chunks: AsyncIterable[bytes],
        cursor: _StreamCursor,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        def on_event(event: SseEvent) -> None:
            cursor.received = True
            # Events without data prime resumption; other event types are not
            # JSON-RPC.
            if not event.data.strip() or (event.event is not None and event.event != "message"):
                return
            try:
                message = parse_json_rpc_message(json.loads(event.data))
            except BaseException as error:
                self.emit_error(error)
                return
            if on_message is not None:
                on_message(message)
            self.emit_message(message)

        await consume_sse_stream(
            chunks,
            ConsumeSseOptions(
                on_event=on_event,
                on_id=lambda value: setattr(cursor, "last_event_id", value),
                on_retry=lambda delay: setattr(cursor, "retry_ms", delay),
                max_event_bytes=self.options.max_message_bytes,
            ),
        )

    async def _consume_response_stream(self, response: HttpResponse, request_id: JsonRpcId) -> None:
        """Read the SSE stream answering one request. When the stream ends or
        breaks before the response arrives and the server assigned event IDs,
        resume it with GET and `Last-Event-ID`, as the server may close response
        streams at will. Otherwise only this request fails."""
        cursor = _StreamCursor()
        answered = False

        def on_message(message: Dict[str, Any]) -> None:
            nonlocal answered
            if is_json_rpc_response(message) and message.get("id") == request_id:
                answered = True

        stream: Optional[AsyncIterator[bytes]] = response.chunks()
        failure: Optional[BaseException] = None
        attempt = 0
        while True:
            if stream is not None:
                try:
                    await self._consume_sse(stream, cursor, on_message)
                    failure = None
                except BaseException as error:
                    failure = error
            if answered or self._closed:
                return
            if failure is not None and not self._is_retryable(failure):
                break
            if cursor.last_event_id is None or attempt >= self._max_retries():
                break
            if cursor.received:
                attempt = 0
            cursor.received = False
            if not await self._sleep(self._reconnect_delay(attempt, cursor.retry_ms)):
                return
            attempt += 1
            try:
                stream = await self._open_sse_stream(cursor.last_event_id)
            except BaseException as error:
                failure = error
                if not self._is_retryable(error):
                    break
                stream = None
        if self._closed:
            return
        reason = "stream ended without a response" if failure is None else str(to_error(failure))
        self.emit_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": JSON_RPC_ERROR_CODES["internalError"],
                    "message": f"MCP response stream failed: {reason}",
                },
            }
        )

    def _start_get_stream(self) -> None:
        if not self.options.open_get_stream or self._get_stream_started or self._closed:
            return
        self._get_stream_started = True
        self._spawn(self._run_get_stream())

    async def _run_get_stream(self) -> None:
        """Keep the server-to-client stream open, reconnecting with backoff when
        it drops."""
        cursor = _StreamCursor()
        attempt = 0
        while not self._closed:
            try:
                stream = await self._open_sse_stream(cursor.last_event_id)
                # The server does not offer a GET stream.
                if stream is None:
                    return
                opened_at = time.monotonic()
                await self._consume_sse(stream, cursor)
                # A stream that stayed up for a while counts as healthy, even
                # if it was idle.
                if cursor.received or (time.monotonic() - opened_at) * 1000 > self._max_delay():
                    attempt = 0
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                if self._closed:
                    return
                if not self._is_retryable(error):
                    self.emit_error(error)
                    return
            cursor.received = False
            if attempt >= self._max_retries():
                self.emit_error(Exception("MCP server-to-client stream dropped and could not be reopened"))
                return
            if not await self._sleep(self._reconnect_delay(attempt, cursor.retry_ms)):
                return
            attempt += 1

    async def _open_sse_stream(self, last_event_id: Optional[str]) -> Optional[AsyncIterator[bytes]]:
        """Open a GET SSE stream. `None` when the server answers 405 (it has no
        GET stream)."""
        headers = {"accept": "text/event-stream"}
        if last_event_id is not None:
            headers["last-event-id"] = last_event_id
        response = await self._authorized_fetch("GET", headers)
        if response.status == 405:
            await _discard(response)
            return None
        await self._check_response(response)
        self._capture_session(response)
        kind = _content_type(response)
        if kind != "text/event-stream" or not response.has_body:
            await _discard(response)
            raise McpHttpError(response.status, f"Unsupported MCP GET response content type: {kind or 'missing'}")
        return response.chunks()

    # -- helpers -------------------------------------------------------------

    def _spawn(self, coroutine: Any) -> None:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _is_retryable(self, error: BaseException) -> bool:
        """Network failures and transient statuses are retried; auth, session,
        and protocol errors are not. A connection dropped mid-body arrives as
        an `OSError` or a short read; the framing errors that arrive as a
        `ValueError` instead describe a server that is not speaking HTTP, so
        they fail the stream rather than being retried."""
        if isinstance(error, McpHttpError):
            return _is_transient_status(error.status)
        return isinstance(error, (OSError, EOFError))

    def _reconnect_delay(self, attempt: int, server_delay_ms: Optional[int]) -> float:
        if server_delay_ms is not None:
            return server_delay_ms
        reconnect = self.options.reconnect
        initial = reconnect.initial_delay_ms if reconnect is not None else DEFAULT_RECONNECT_INITIAL_DELAY_MS
        return min(initial * 2**attempt, self._max_delay())

    def _max_delay(self) -> int:
        reconnect = self.options.reconnect
        return reconnect.max_delay_ms if reconnect is not None else DEFAULT_RECONNECT_MAX_DELAY_MS

    def _max_retries(self) -> int:
        reconnect = self.options.reconnect
        return reconnect.max_retries if reconnect is not None else DEFAULT_RECONNECT_MAX_RETRIES

    async def _sleep(self, ms: float) -> bool:
        """Resolves false when the transport closed while waiting."""
        if self._signal.aborted:
            return False
        try:
            await asyncio.wait_for(self._signal.wait(), ms / 1000.0)
        except asyncio.TimeoutError:
            return True
        return False

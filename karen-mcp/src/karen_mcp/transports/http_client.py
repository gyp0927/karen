"""A small async HTTP/1.1 client, enough for the Streamable HTTP transport.

pi hands this job to the platform's `fetch`. Python has no async equivalent in
the standard library, and the alternatives — depending on `httpx`, or reading a
blocking socket on a worker thread — cost more than they buy for one request at
a time per connection, so the transport carries its own: connect, write the
request, and frame the body the three ways HTTP/1.1 allows (`content-length`,
`chunked`, or close-delimited).

It deliberately does not do proxies, redirects, cookies, or HTTP/2:
applications that need those inject their own `McpFetch` into the transport.
"""

from __future__ import annotations

import asyncio
import json
import ssl
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from ..protocol.jsonrpc import McpConnectionClosedError

__all__ = [
    "DEFAULT_MAX_HEADER_BYTES",
    "Headers",
    "HttpRequest",
    "HttpResponse",
    "http_fetch",
]

CHUNK_SIZE = 64 * 1024
DEFAULT_MAX_HEADER_BYTES = 64 * 1024
_CRLF = b"\r\n"
_HEAD_END = b"\r\n\r\n"


class Headers:
    """Response headers with case-insensitive lookup (pi's `Headers.get`)."""

    def __init__(self, items: Any = ()) -> None:
        self._values: Dict[str, str] = {}
        self._names: Dict[str, str] = {}
        for name, value in items.items() if isinstance(items, dict) else items:
            self.add(name, value)

    def add(self, name: str, value: str) -> None:
        """Set a header, replacing a value already present under any spelling."""
        key = name.lower()
        self._values[key] = value
        self._names.setdefault(key, name)

    def append(self, name: str, value: str) -> None:
        """Add a header the way a wire duplicate means it: combined with the
        value already there, as a recipient combines repeated fields."""
        key = name.lower()
        if key in self._values:
            self._values[key] = f"{self._values[key]}, {value}"
            return
        self.add(name, value)

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self._values.get(name.lower(), default)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._values

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Any:
        return iter(self._names.values())

    def items(self) -> List[Tuple[str, str]]:
        return [(self._names[key], value) for key, value in self._values.items()]

    def __repr__(self) -> str:
        return f"Headers({self.items()!r})"


@dataclass
class HttpRequest:
    """pi's `RequestInit`, reduced to what a JSON-RPC exchange needs."""

    method: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    #: Anything with an `aborted` flag and an awaitable `wait()` — the
    #: transport's cancellation, so `close()` interrupts a read in flight.
    signal: Any = None


class HttpResponse:
    """A response whose body is streamed by the caller.

    The body has one consumer: `chunks()`, or `read()` to drain it. `close()`
    drops whatever is left, which is what a caller that wanted only the status
    and headers does.
    """

    def __init__(
        self,
        status: int,
        headers: Headers,
        reader: Optional["asyncio.StreamReader"],
        writer: Optional["asyncio.StreamWriter"],
        framing: str,
        length: Optional[int] = None,
        abort: Any = None,
    ) -> None:
        self.status = status
        self.headers = headers
        self._reader = reader
        self._writer = writer
        self._framing = framing
        self._length = length
        self._consumed = False
        self._closed = False
        self._abort_task: Optional["asyncio.Task[Any]"] = None
        if abort is not None and reader is not None:
            self._abort_task = asyncio.ensure_future(_wait_abort(abort))

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def has_body(self) -> bool:
        return self._reader is not None and self._framing != "none"

    def header(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self.headers.get(name, default)

    async def chunks(self) -> AsyncIterator[bytes]:
        """The body in the order it arrives, for as long as the server sends it."""
        if self._consumed:
            raise RuntimeError("MCP HTTP response body was already read")
        reader = self._reader
        if reader is None or self._framing == "none":
            return
        self._consumed = True
        try:
            if self._framing == "chunked":
                async for chunk in self._chunked(reader):
                    yield chunk
            elif self._framing == "length":
                remaining = self._length or 0
                while remaining > 0:
                    chunk = await self._read(reader, min(remaining, CHUNK_SIZE))
                    if not chunk:
                        raise asyncio.IncompleteReadError(b"", remaining)
                    remaining -= len(chunk)
                    yield chunk
            else:
                while True:
                    chunk = await self._read(reader, CHUNK_SIZE)
                    if not chunk:
                        return
                    yield chunk
        finally:
            await self.close()

    async def read(self, limit: Optional[int] = None) -> bytes:
        """Drain the body, keeping at most `limit` bytes of it."""
        stream = self.chunks()
        parts: List[bytes] = []
        total = 0
        try:
            async for chunk in stream:
                if limit is not None:
                    remaining = limit - total
                    if remaining <= 0:
                        break
                    chunk = chunk[:remaining]
                parts.append(chunk)
                total += len(chunk)
        finally:
            await stream.aclose()
        return b"".join(parts)

    async def text(self, limit: Optional[int] = None) -> str:
        return (await self.read(limit)).decode("utf-8", "replace")

    async def json(self) -> Any:
        return json.loads(await self.read())

    async def close(self) -> None:
        """Drop the body and the connection. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._reader = None
        if self._abort_task is not None:
            self._abort_task.cancel()
            self._abort_task = None
        writer, self._writer = self._writer, None
        if writer is not None:
            try:
                writer.close()
            except (RuntimeError, OSError):
                pass

    # -- internals -----------------------------------------------------------

    async def _wait(self, coroutine: Any) -> Any:
        """Await a read, or give up as soon as the request is aborted."""
        if self._closed:
            raise McpConnectionClosedError("MCP HTTP response body is closed")
        if self._abort_task is None:
            return await coroutine
        task = asyncio.ensure_future(coroutine)
        try:
            await asyncio.wait({task, self._abort_task}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            # Cancelled from outside: the read must not be left behind.
            task.cancel()
            raise
        if task.done():
            return task.result()
        task.cancel()
        raise McpConnectionClosedError("MCP HTTP request aborted")

    async def _read(self, reader: "asyncio.StreamReader", size: int) -> bytes:
        return await self._wait(reader.read(size))

    async def _read_line(self, reader: "asyncio.StreamReader") -> bytes:
        try:
            line = await self._wait(reader.readuntil(_CRLF))
        except asyncio.LimitOverrunError:
            raise ValueError("MCP HTTP response line is too long") from None
        except asyncio.IncompleteReadError as error:
            raise asyncio.IncompleteReadError(error.partial, None) from None
        return line[: -len(_CRLF)]

    async def _chunked(self, reader: "asyncio.StreamReader") -> AsyncIterator[bytes]:
        while True:
            line = await self._read_line(reader)
            size_text = line.split(b";", 1)[0].strip()
            try:
                size = int(size_text, 16)
            except ValueError:
                raise ValueError(f"Invalid chunk size in MCP HTTP response: {size_text!r}") from None
            if size == 0:
                # Trailers, terminated by a blank line.
                while await self._read_line(reader):
                    pass
                return
            remaining = size
            while remaining > 0:
                chunk = await self._read(reader, min(remaining, CHUNK_SIZE))
                if not chunk:
                    raise asyncio.IncompleteReadError(b"", remaining)
                remaining -= len(chunk)
                yield chunk
            if await self._read_line(reader):
                raise ValueError("Invalid chunk terminator in MCP HTTP response")


async def _wait_abort(signal: Any) -> None:
    try:
        await signal.wait()
    except asyncio.CancelledError:
        raise
    except BaseException:
        pass


def _target_and_host(parts: Any) -> Tuple[str, str]:
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    host = f"{host}:{parts.port}" if parts.port is not None else host
    target = parts.path or "/"
    return (f"{target}?{parts.query}" if parts.query else target), host


def _split_head(head: bytes) -> Tuple[int, Headers]:
    lines = head.split(_CRLF)
    status_line = lines[0].decode("latin-1")
    pieces = status_line.split(" ", 2)
    if len(pieces) < 2 or not pieces[1].isdigit():
        raise ValueError(f"Invalid MCP HTTP status line: {status_line!r}")
    headers = Headers()
    for line in lines[1:]:
        name, separator, value = line.decode("latin-1").partition(":")
        if separator:
            # A repeated field (a second `www-authenticate`, say) combines,
            # so a challenge further down the list is not lost.
            headers.append(name.strip(), value.strip())
    return int(pieces[1]), headers


def _framing(method: str, status: int, headers: Headers) -> str:
    if method == "HEAD" or status in (204, 304) or 100 <= status < 200:
        return "none"
    if "chunked" in (headers.get("transfer-encoding") or "").lower():
        return "chunked"
    if headers.get("content-length") is not None:
        return "length"
    # No length and no chunking: the body runs until the connection closes,
    # which the `Connection: close` on the request guarantees.
    return "close"


async def http_fetch(url: str, request: HttpRequest) -> HttpResponse:
    """`McpFetch`'s default: one connection, one request, a streamed body.

    The request's signal covers the whole exchange, not only the body: a
    transport that closes while a connect, a write, or the response head is
    still outstanding sees the request fail at once, as pi's `fetch` aborts.
    """
    signal = request.signal
    if signal is None:
        return await _fetch(url, request)
    task = asyncio.ensure_future(_fetch(url, request))
    abort = asyncio.ensure_future(_wait_abort(signal))
    try:
        await asyncio.wait({task, abort}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            return task.result()
        task.cancel()
        raise McpConnectionClosedError("MCP HTTP request aborted")
    finally:
        abort.cancel()
        if not task.done():
            task.cancel()


async def _fetch(url: str, request: HttpRequest) -> HttpResponse:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"Invalid MCP server URL: {url!r}")
    target, host = _target_and_host(parts)
    context = ssl.create_default_context() if parts.scheme == "https" else None
    reader, writer = await asyncio.open_connection(
        parts.hostname,
        parts.port or (443 if parts.scheme == "https" else 80),
        ssl=context,
        server_hostname=parts.hostname if context is not None else None,
        limit=DEFAULT_MAX_HEADER_BYTES,
    )
    try:
        lines = [f"{request.method} {target} HTTP/1.1", f"Host: {host}", "Connection: close"]
        for name, value in request.headers.items():
            lines.append(f"{name}: {value}")
        if request.body is not None:
            lines.append(f"Content-Length: {len(request.body)}")
        writer.write(_CRLF.join(line.encode("latin-1") for line in lines) + _HEAD_END)
        if request.body:
            writer.write(request.body)
        await writer.drain()

        head = await reader.readuntil(_HEAD_END)
        status, headers = _split_head(head[: -len(_HEAD_END)])
        # An interim response (1xx) is not the answer; the real head follows.
        while 100 <= status < 200:
            head = await reader.readuntil(_HEAD_END)
            status, headers = _split_head(head[: -len(_HEAD_END)])
        framing = _framing(request.method, status, headers)
        length = None
        if framing == "length":
            declared = headers.get("content-length") or "0"
            if not declared.isdigit():
                raise ValueError(f"Invalid Content-Length in MCP HTTP response: {declared!r}")
            length = int(declared)
        return HttpResponse(status, headers, reader, writer, framing, length, request.signal)
    except BaseException:
        try:
            writer.close()
        except (RuntimeError, OSError):
            pass
        raise

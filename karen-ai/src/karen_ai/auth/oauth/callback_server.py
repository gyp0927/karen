"""Minimal asyncio localhost HTTP server for OAuth redirect callbacks.

Replaces pi-ai's node:http callback servers. Each flow supplies a handler
that receives (request_path, query_params) and returns the HTML response to
send; settling the login is the handler's job via CallbackWaiter.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Generic, Optional, TypeVar, Union
from urllib.parse import parse_qsl, urlsplit

T = TypeVar("T")


@dataclass
class CallbackHttpResponse:
    status: int
    html: str


CallbackHandler = Callable[
    [str, Dict[str, str]],
    Union[CallbackHttpResponse, Awaitable[CallbackHttpResponse]],
]

_REASONS = {
    200: "OK",
    400: "Bad Request",
    404: "Not Found",
    409: "Conflict",
    500: "Internal Server Error",
    502: "Bad Gateway",
}

_HEAD_TIMEOUT_SECONDS = 10.0


class CallbackWaiter(Generic[T]):
    """One-shot future resolved by the callback handler, manual entry, or abort."""

    def __init__(self) -> None:
        self._future: asyncio.Future = asyncio.get_running_loop().create_future()

    @property
    def settled(self) -> bool:
        return self._future.done()

    def settle(self, value: Optional[T]) -> None:
        if not self._future.done():
            self._future.set_result(value)

    def fail(self, error: BaseException) -> None:
        if not self._future.done():
            self._future.set_exception(error)

    async def wait(self) -> Optional[T]:
        return await self._future


class OAuthCallbackServer:
    def __init__(self, server: asyncio.AbstractServer, host: str, port: int) -> None:
        self._server = server
        self.host = host
        self.port = port

    def close(self) -> None:
        self._server.close()


async def start_oauth_callback_server(
    host: str,
    port: int,
    handler: CallbackHandler,
) -> OAuthCallbackServer:
    """Bind a localhost HTTP server routing every request to `handler`.

    Returns once the socket is listening; raises OSError if the bind fails.
    `port=0` picks an ephemeral port (see OAuthCallbackServer.port).
    """

    async def _handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=_HEAD_TIMEOUT_SECONDS)
            request_line = head.split(b"\r\n", 1)[0].decode("latin-1")
            parts = request_line.split(" ")
            if len(parts) < 2:
                return
            split = urlsplit(parts[1])
            query = dict(parse_qsl(split.query))
            result = handler(split.path or "/", query)
            if asyncio.iscoroutine(result):
                result = await result
            body = result.html.encode("utf-8")
            reason = _REASONS.get(result.status, "OK")
            writer.write(
                (
                    f"HTTP/1.1 {result.status} {reason}\r\n"
                    "content-type: text/html; charset=utf-8\r\n"
                    "cache-control: no-store\r\n"
                    f"content-length: {len(body)}\r\n"
                    "connection: close\r\n"
                    "\r\n"
                ).encode("latin-1")
                + body
            )
            await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    server = await asyncio.start_server(_handle_connection, host, port)
    sockets = server.sockets or []
    bound_port = int(sockets[0].getsockname()[1]) if sockets else port
    return OAuthCallbackServer(server, host, bound_port)

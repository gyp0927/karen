"""The loopback OAuth callback server (pi's `oauth/callback.ts`).

Listens on a free loopback port and hands the browser's redirect — the
authorization code, or the error the authorization server sent back — to the
task waiting on `wait_for_callback`. Where pi uses `node:http`, the port uses
`asyncio.start_server` and reads just enough of the request to route it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import parse_qsl, urlsplit

__all__ = [
    "OAuthCallback",
    "OAuthCallbackPage",
    "OAuthCallbackServer",
    "OAuthCallbackServerOptions",
]

_MAX_HEAD_BYTES = 64 * 1024

_REASONS = {200: "OK", 400: "Bad Request", 404: "Not Found"}


@dataclass
class OAuthCallback:
    code: str
    state: str
    iss: Optional[str] = None


@dataclass
class OAuthCallbackPage:
    """Outcome shown on the browser page after the redirect (pi's
    `{ok: true} | {ok: false, message, details?}`)."""

    ok: bool
    message: Optional[str] = None
    details: Optional[str] = None


@dataclass
class OAuthCallbackServerOptions:
    #: Address to listen on.
    host: str = "127.0.0.1"
    #: Host name in `redirect_url`, for example `localhost` for a client
    #: registered with it while listening on `127.0.0.1`. Defaults to `host`.
    redirect_host: Optional[str] = None
    port: int = 0
    path: str = "/callback"
    timeout_ms: float = 5 * 60_000
    #: Render the browser page as HTML. Default: a plain-text message.
    render_page: Optional[Callable[[OAuthCallbackPage], str]] = None


def _plain_text(page: OAuthCallbackPage) -> str:
    if page.ok:
        return "Authorization complete. You may close this window."
    return f"{page.message}\n\n{page.details}" if page.details else (page.message or "")


def _first_param(query: str, name: str) -> Optional[str]:
    """`URLSearchParams.get`: the first occurrence wins."""
    for key, value in parse_qsl(query, keep_blank_values=True):
        if key == name:
            return value
    return None


class OAuthCallbackServer:
    def __init__(self, options: OAuthCallbackServerOptions) -> None:
        self._options = options
        self._path = options.path
        self._timeout_ms = options.timeout_ms
        self._render_page = options.render_page
        self._server: Optional[asyncio.AbstractServer] = None
        self._pending: Dict[str, Tuple["asyncio.Future[OAuthCallback]", asyncio.TimerHandle]] = {}
        self.redirect_url = ""

    @classmethod
    async def listen(cls, options: Optional[OAuthCallbackServerOptions] = None) -> "OAuthCallbackServer":
        instance = cls(options or OAuthCallbackServerOptions())
        server = await asyncio.start_server(
            instance._handle_connection, instance._options.host, instance._options.port
        )
        instance._server = server
        socket = server.sockets[0] if server.sockets else None
        if socket is None:
            raise Exception("OAuth callback server did not bind to TCP")
        port = socket.getsockname()[1]
        redirect_host = instance._options.redirect_host or instance._options.host
        if ":" in redirect_host:
            redirect_host = f"[{redirect_host}]"
        instance.redirect_url = f"http://{redirect_host}:{port}{instance._path}"
        return instance

    def wait_for_callback(self, state: str) -> "asyncio.Future[OAuthCallback]":
        if state in self._pending:
            raise Exception("OAuth state is already pending")
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[OAuthCallback]" = loop.create_future()
        timer = loop.call_later(self._timeout_ms / 1000.0, self._on_timeout, state)
        self._pending[state] = (future, timer)
        return future

    async def close(self) -> None:
        for future, timer in self._pending.values():
            timer.cancel()
            if not future.done():
                future.set_exception(Exception("OAuth callback server closed"))
        self._pending.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # -- internals -----------------------------------------------------------

    def _on_timeout(self, state: str) -> None:
        pending = self._pending.pop(state, None)
        if pending is None:
            return
        future, _timer = pending
        if not future.done():
            future.set_exception(Exception("OAuth callback timed out"))

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            if len(head) > _MAX_HEAD_BYTES:
                raise ValueError("OAuth callback request head is too large")
            request_line = head.split(b"\r\n", 1)[0].decode("latin-1")
            pieces = request_line.split(" ")
            target = pieces[1] if len(pieces) >= 2 else "/"
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError):
            writer.close()
            return
        # A target `urlsplit` cannot parse must not escape into
        # `client_connected_cb` as an unhandled exception.
        try:
            status, page = self._route(target)
        except ValueError:
            status, page = 400, OAuthCallbackPage(ok=False, message="Bad request")
        if self._render_page is not None:
            body = self._render_page(page).encode("utf-8")
            content_type = "text/html; charset=utf-8"
            extra = "cache-control: no-store\r\n"
        else:
            body = _plain_text(page).encode("utf-8")
            content_type = "text/plain; charset=utf-8"
            extra = ""
        head = (
            f"HTTP/1.1 {status} {_REASONS.get(status, '')}\r\n"
            f"content-type: {content_type}\r\n"
            f"content-length: {len(body)}\r\n"
            f"{extra}"
            f"connection: close\r\n"
            f"\r\n"
        )
        writer.write(head.encode("latin-1") + body)
        try:
            await writer.drain()
        except (ConnectionError, RuntimeError):
            pass
        writer.close()

    def _route(self, target: str) -> Tuple[int, OAuthCallbackPage]:
        parts = urlsplit(target)
        if parts.path != self._path:
            return 404, OAuthCallbackPage(ok=False, message="Not found")
        state = _first_param(parts.query, "state")
        pending = self._pending.pop(state, None) if state else None
        if not state or pending is None:
            return 400, OAuthCallbackPage(ok=False, message="Invalid or expired OAuth state")
        future, timer = pending
        timer.cancel()
        error = _first_param(parts.query, "error")
        if error:
            # pi's `??`: an explicit empty description is kept, only an absent
            # one falls back to the error code.
            description = _first_param(parts.query, "error_description")
            if description is None:
                description = error
            if not future.done():
                future.set_exception(Exception(description))
            return 200, OAuthCallbackPage(
                ok=False, message="Authorization failed. You may close this window.", details=description
            )
        code = _first_param(parts.query, "code")
        if not code:
            if not future.done():
                future.set_exception(Exception("OAuth callback did not include an authorization code"))
            return 400, OAuthCallbackPage(ok=False, message="Missing authorization code")
        iss = _first_param(parts.query, "iss")
        if not future.done():
            future.set_result(OAuthCallback(code=code, state=state, iss=iss or None))
        return 200, OAuthCallbackPage(ok=True)

"""A loopback HTTP server for transport tests (pi's `test/helpers.ts`).

Handlers run on the server's own threads, so they are plain functions: a test
that has to hold a response open coordinates with a `threading.Event`.
"""

from __future__ import annotations

import http.server
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

Handler = Callable[["Request", "Response", List["Request"]], None]


class Request:
    def __init__(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: bytes,
        header_items: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.method = method
        #: The raw request target (`/mcp?x=1`), for path routing in handlers.
        self.path = path
        self.headers = headers
        #: Every header as sent, so a handler can see a repeated name that the
        #: dict above collapsed.
        self.header_items = list(header_items) if header_items is not None else list(headers.items())
        self.body = body

    @property
    def url(self) -> Any:
        return urlsplit(self.path)

    @property
    def message(self) -> Optional[Dict[str, Any]]:
        import json

        return json.loads(self.body.decode("utf-8")) if self.body else None

    def header(self, name: str) -> Optional[str]:
        return self.headers.get(name.lower())

    def header_values(self, name: str) -> List[str]:
        return [value for key, value in self.header_items if key == name.lower()]


class Response:
    """The bits of node's `ServerResponse` the tests use."""

    def __init__(self, handler: "http.server.BaseHTTPRequestHandler") -> None:
        self._handler = handler
        self.status_code = 200
        self._headers: List[Tuple[str, str]] = []
        self._started = False

    def set_header(self, name: str, value: str) -> None:
        self._headers.append((name, value))

    def write_head(self, status_code: int, headers: Optional[Dict[str, str]] = None) -> None:
        self.status_code = status_code
        for name, value in (headers or {}).items():
            self.set_header(name, value)
        self._start()

    def write(self, text: str) -> None:
        self._start()
        self._handler.wfile.write(text.encode("utf-8"))
        self._handler.wfile.flush()

    def end(self, text: str = "") -> None:
        self._start()
        if text:
            self._handler.wfile.write(text.encode("utf-8"))
        self._handler.wfile.flush()

    def _start(self) -> None:
        if self._started:
            return
        self._started = True
        self._handler.send_response_only(self.status_code)
        for name, value in self._headers:
            self._handler.send_header(name, value)
        self._handler.end_headers()


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass

    def _serve(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        header_items = [(name.lower(), value) for name, value in self.headers.items()]
        headers = dict(header_items)
        request = Request(self.command, self.path, headers, body, header_items)
        self.server.record(request)  # type: ignore[attr-defined]
        response = Response(self)
        try:
            self.server.handler(request, response, self.server.requests)  # type: ignore[attr-defined]
        except Exception as error:  # pragma: no cover - surfaces through the client
            response.write_head(500)
            response.end(str(error))

    do_GET = _serve
    do_POST = _serve
    do_DELETE = _serve
    do_PUT = _serve


class LoopbackServer:
    """Starts on a free port; `url` is the origin plus `/mcp`."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: List[Request] = []
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.handler = handler  # type: ignore[attr-defined]
        self._server.requests = self.requests  # type: ignore[attr-defined]
        self._server.record = self.requests.append  # type: ignore[attr-defined]
        # A test that closes mid-stream would otherwise spam a broken-pipe
        # traceback through `socketserver`'s default error handler.
        self._server.handle_error = lambda request, client_address: None  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> str:
        self._thread.start()
        return self.origin

    @property
    def origin(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def url(self) -> str:
        return f"{self.origin}/mcp"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

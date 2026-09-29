"""Codex error taxonomy shared by the SSE and WebSocket transports.

pi-ai keeps these in `openai-codex-responses.ts`; karen-ai splits them out so the
WebSocket transport module can use them without importing the API module back.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: The backend reports this when a session already holds the maximum number of
#: WebSocket connections; the request is retried once on a fresh socket.
WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE = "websocket_connection_limit_reached"
#: The cached continuation references a response the backend no longer knows.
PREVIOUS_RESPONSE_NOT_FOUND_CODE = "previous_response_not_found"


class CodexApiError(Exception):
    """A structured error reported by the Codex backend."""

    def __init__(self, message: str, code: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.payload = payload


class CodexProtocolError(Exception):
    """Malformed Codex stream payload (SSE or WebSocket)."""

    def __init__(self, message: str, payload: Any = None) -> None:
        super().__init__(message)
        self.payload = payload


class ProviderStreamEventCallbackError(Exception):
    """The caller's `on_provider_stream_event` callback raised."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


def is_codex_non_transport_error(error: BaseException) -> bool:
    """True for errors that must not trigger a transport retry or SSE fallback."""
    return isinstance(error, (CodexApiError, CodexProtocolError, ProviderStreamEventCallbackError))


def is_websocket_connection_limit_reached_error(error: BaseException) -> bool:
    return isinstance(error, CodexApiError) and error.code == WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE


def is_previous_response_not_found_error(error: BaseException) -> bool:
    return isinstance(error, CodexApiError) and error.code == PREVIOUS_RESPONSE_NOT_FOUND_CODE


__all__ = [
    "PREVIOUS_RESPONSE_NOT_FOUND_CODE",
    "WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE",
    "CodexApiError",
    "CodexProtocolError",
    "ProviderStreamEventCallbackError",
    "is_codex_non_transport_error",
    "is_previous_response_not_found_error",
    "is_websocket_connection_limit_reached_error",
]

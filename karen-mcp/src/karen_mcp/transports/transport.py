"""The transport interface and the listener bookkeeping every transport shares
(pi's `transports/transport.ts`).

A transport owns framing and I/O only: it hands individual JSON-RPC messages to
the client, which owns request correlation, initialization, timeouts,
cancellation and server requests.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List

from ..protocol.jsonrpc import to_error

__all__ = [
    "DEFAULT_MAX_MESSAGE_BYTES",
    "CloseListener",
    "ErrorListener",
    "McpTransport",
    "MessageListener",
    "TransportEvents",
]

DEFAULT_MAX_MESSAGE_BYTES = 16 * 1024 * 1024

MessageListener = Callable[[Dict[str, Any]], None]
ErrorListener = Callable[[BaseException], None]
CloseListener = Callable[[], None]
Unsubscribe = Callable[[], None]


class McpTransport(abc.ABC):
    """What `McpClient` needs from a transport."""

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def send(self, message: Dict[str, Any]) -> None: ...

    @abc.abstractmethod
    async def close(self) -> None: ...

    @abc.abstractmethod
    def on_message(self, listener: MessageListener) -> Unsubscribe: ...

    @abc.abstractmethod
    def on_error(self, listener: ErrorListener) -> Unsubscribe: ...

    @abc.abstractmethod
    def on_close(self, listener: CloseListener) -> Unsubscribe: ...

    def set_protocol_version(self, version: str) -> None:
        """Transports that send the negotiated version (HTTP) override this."""


class TransportEvents:
    """Listener bookkeeping shared by transports. `emit_close` fires at most
    once per transport."""

    def __init__(self) -> None:
        self._message_listeners: List[MessageListener] = []
        self._error_listeners: List[ErrorListener] = []
        self._close_listeners: List[CloseListener] = []
        self._close_emitted = False

    def on_message(self, listener: MessageListener) -> Unsubscribe:
        self._message_listeners.append(listener)
        return lambda: self._discard(self._message_listeners, listener)

    def on_error(self, listener: ErrorListener) -> Unsubscribe:
        self._error_listeners.append(listener)
        return lambda: self._discard(self._error_listeners, listener)

    def on_close(self, listener: CloseListener) -> Unsubscribe:
        self._close_listeners.append(listener)
        return lambda: self._discard(self._close_listeners, listener)

    def emit_message(self, message: Dict[str, Any]) -> None:
        for listener in list(self._message_listeners):
            listener(message)

    def emit_error(self, error: Any) -> None:
        normalized = to_error(error)
        for listener in list(self._error_listeners):
            listener(normalized)

    def emit_close(self) -> None:
        if self._close_emitted:
            return
        self._close_emitted = True
        for listener in list(self._close_listeners):
            listener()

    @staticmethod
    def _discard(listeners: List[Any], listener: Any) -> None:
        try:
            listeners.remove(listener)
        except ValueError:
            pass

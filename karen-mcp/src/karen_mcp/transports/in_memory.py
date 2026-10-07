"""A transport pair wired to each other in process (pi's `transports/in-memory.ts`).

Used by the client's own tests and exported from `karen_mcp.testing` so servers
can be exercised without a subprocess or a socket.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any, Dict, Optional, Tuple

from ..protocol.jsonrpc import McpConnectionClosedError
from .transport import McpTransport, TransportEvents

__all__ = ["InMemoryTransport", "create_in_memory_transport_pair"]


class InMemoryTransport(TransportEvents, McpTransport):
    def __init__(self) -> None:
        super().__init__()
        self._peer: Optional["InMemoryTransport"] = None
        self._started = False
        self._closed = False

    def connect_peer(self, peer: "InMemoryTransport") -> None:
        if self._peer is not None:
            raise RuntimeError("In-memory MCP transport already has a peer")
        self._peer = peer

    @property
    def closed(self) -> bool:
        return self._closed

    async def start(self) -> None:
        if self._closed:
            raise McpConnectionClosedError()
        self._started = True

    async def send(self, message: Dict[str, Any]) -> None:
        if not self._started or self._closed:
            raise McpConnectionClosedError()
        peer = self._peer
        if peer is None or not peer._started or peer._closed:
            raise McpConnectionClosedError("In-memory MCP peer is not connected")
        # A copy, and delivered on the next loop turn rather than inline: a peer
        # that mutates the message or answers before `send` returns would
        # otherwise reenter the sender.
        delivered = copy.deepcopy(message)
        asyncio.get_running_loop().call_soon(peer._deliver, delivered)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.emit_close()
        if self._peer is not None:
            await self._peer.close()

    def _deliver(self, message: Dict[str, Any]) -> None:
        if self._closed:
            return
        self.emit_message(message)


def create_in_memory_transport_pair() -> Tuple[InMemoryTransport, InMemoryTransport]:
    client = InMemoryTransport()
    server = InMemoryTransport()
    client.connect_peer(server)
    server.connect_peer(client)
    return client, server

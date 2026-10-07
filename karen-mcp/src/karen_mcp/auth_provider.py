"""Bearer tokens for an MCP HTTP transport (pi's `auth-provider.ts`).

`McpFetch` is deliberately structural: it matches `urllib.request`-shaped
callables and any test double, and the HTTP transport calls it without assuming
`httpx` or `aiohttp` is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol

__all__ = ["AuthProvider", "McpFetch", "UnauthorizedContext"]

#: `(url, init) -> response`, with `init` carrying method/headers/body/signal.
McpFetch = Callable[..., Awaitable[Any]]


@dataclass
class UnauthorizedContext:
    """What an auth provider gets when a request is rejected."""

    #: The 401 response, or a 403 whose challenge reports `insufficient_scope`.
    response: Any
    server_url: Any
    fetch: McpFetch
    #: Access token the rejected request carried, if any. A different current
    #: token means another request already refreshed it.
    token: Optional[str] = None


class AuthProvider(Protocol):
    """Supplies bearer tokens to an MCP HTTP transport and may refresh them
    after a 401 response."""

    async def token(self) -> Optional[str]: ...

    async def on_unauthorized(self, context: UnauthorizedContext) -> None: ...

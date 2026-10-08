"""The default stateful OAuth client provider (pi's `oauth/provider.ts`).

One provider serves one exact MCP server URL; persisted state for another URL
is ignored, so credentials never leak across servers. Applications inject
durable storage through the `store` protocol (`load`/`save`, sync or async).

The persisted state dict uses snake_case keys (`server_url`, `tokens_expire_at`,
...) — pi's camelCase — because it is this package's own format, not a server
document; the server documents inside it keep their wire shape.
"""

from __future__ import annotations

import asyncio
import copy
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

from .flow import _maybe_await

__all__ = [
    "MemoryOAuthStateStore",
    "McpOAuthProvider",
    "McpOAuthProviderOptions",
]


def _normalize_url(value: str) -> str:
    """`String(new URL(value))` for the parts `own()` relies on: scheme and
    host lowercased, a default port dropped, a bare origin given its slash."""
    parts = urlsplit(value)
    if not parts.scheme or not parts.netloc:
        return value
    scheme = parts.scheme.lower()
    userinfo, _, hostport = parts.netloc.rpartition("@")
    hostport = hostport.lower()
    default_port = {"http": 80, "https": 443}.get(scheme)
    if default_port is not None and hostport.endswith(f":{default_port}"):
        hostport = hostport[: -len(f":{default_port}")]
    netloc = f"{userinfo}@{hostport}" if userinfo else hostport
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, parts.fragment))


class MemoryOAuthStateStore:
    """In-memory state store; copies on the way in and out, like
    `structuredClone`, so a caller cannot mutate persisted state by aliasing."""

    def __init__(self) -> None:
        self._value: Optional[Dict[str, Any]] = None

    def load(self) -> Optional[Dict[str, Any]]:
        return None if self._value is None else copy.deepcopy(self._value)

    def save(self, state: Dict[str, Any]) -> None:
        self._value = copy.deepcopy(state)


@dataclass
class McpOAuthProviderOptions:
    """pi's `McpOAuthProviderOptions`. `client_metadata` lacks
    `redirect_uris` unless the application sets it."""

    server_url: str
    redirect_url: str
    client_metadata: Dict[str, Any]
    #: Called with the authorization URL when the user has to authorize.
    on_redirect: Callable[[str], Any]
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    store: Optional[Any] = None


class McpOAuthProvider:
    """An `OAuthClientProvider` persisting everything through its store."""

    redirect_url: str
    client_metadata: Dict[str, Any]

    def __init__(self, options: McpOAuthProviderOptions) -> None:
        self._server_url = _normalize_url(str(options.server_url))
        self.redirect_url = str(options.redirect_url)
        client_metadata = dict(options.client_metadata)
        # `??` semantics: an explicit None is absent too.
        if client_metadata.get("redirect_uris") is None:
            client_metadata["redirect_uris"] = [self.redirect_url]
        if client_metadata.get("grant_types") is None:
            client_metadata["grant_types"] = ["authorization_code", "refresh_token"]
        if client_metadata.get("response_types") is None:
            client_metadata["response_types"] = ["code"]
        if client_metadata.get("token_endpoint_auth_method") is None:
            client_metadata["token_endpoint_auth_method"] = "client_secret_post" if options.client_secret else "none"
        self.client_metadata = client_metadata
        self._configured_client: Optional[Dict[str, Any]] = None
        if options.client_id:
            self._configured_client = {"client_id": options.client_id}
            if options.client_secret:
                self._configured_client["client_secret"] = options.client_secret
        self._store = options.store if options.store is not None else MemoryOAuthStateStore()
        self._on_redirect = options.on_redirect
        self._writes = asyncio.Lock()

    async def state(self) -> str:
        existing = (await self._load()).get("oauth_state")
        if existing:
            return existing
        value = secrets.token_hex(32)
        await self._update(lambda state: {**state, "oauth_state": value})
        return value

    async def client_information(self) -> Optional[Dict[str, Any]]:
        if self._configured_client is not None:
            return self._configured_client
        return (await self._load()).get("client_information")

    async def save_client_information(self, information: Dict[str, Any]) -> None:
        if self._configured_client is not None:
            return
        await self._update(lambda state: {**state, "client_information": information})

    async def tokens(self) -> Optional[Dict[str, Any]]:
        return (await self._load()).get("tokens")

    async def save_tokens(self, tokens: Dict[str, Any]) -> None:
        expires_in = tokens.get("expires_in")
        expires_at = None if expires_in is None else int(time.time() * 1000) + int(expires_in * 1000)

        def update(state: Dict[str, Any]) -> Dict[str, Any]:
            next_state = {**state, "tokens": tokens}
            if expires_at is None:
                next_state.pop("tokens_expire_at", None)
            else:
                next_state["tokens_expire_at"] = expires_at
            return next_state

        await self._update(update)

    async def redirect_to_authorization(self, url: str) -> None:
        await _maybe_await(self._on_redirect(url))

    async def save_code_verifier(self, verifier: str) -> None:
        await self._update(lambda state: {**state, "code_verifier": verifier})

    async def code_verifier(self) -> str:
        verifier = (await self._load()).get("code_verifier")
        if not verifier:
            raise Exception("No OAuth PKCE code verifier is stored")
        return verifier

    async def invalidate_credentials(self, kind: str) -> None:
        def update(state: Dict[str, Any]) -> Dict[str, Any]:
            next_state = dict(state)
            if kind in ("all", "client"):
                next_state.pop("client_information", None)
            if kind in ("all", "tokens"):
                next_state.pop("tokens", None)
                next_state.pop("tokens_expire_at", None)
            if kind in ("all", "verifier"):
                next_state.pop("code_verifier", None)
            if kind in ("all", "discovery"):
                next_state.pop("discovery", None)
            if kind == "all":
                next_state.pop("oauth_state", None)
            return next_state

        await self._update(update)

    async def save_discovery_state(self, discovery: Dict[str, Any]) -> None:
        await self._update(lambda state: {**state, "discovery": discovery})

    async def discovery_state(self) -> Optional[Dict[str, Any]]:
        return (await self._load()).get("discovery")

    async def _load(self) -> Dict[str, Any]:
        async with self._writes:
            return self._own(await _maybe_await(self._store.load()))

    async def _update(self, update: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        # Read-modify-write goes through one lock, pi's `writes` promise chain,
        # so concurrent updates cannot lose each other.
        async with self._writes:
            await _maybe_await(self._store.save(update(self._own(await _maybe_await(self._store.load())))))

    def _own(self, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Stored state for another server URL is ignored so credentials never
        leak across servers."""
        if state is not None and state.get("server_url") == self._server_url:
            return state
        return {"server_url": self._server_url}

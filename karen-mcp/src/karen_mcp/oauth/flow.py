"""The OAuth flow (pi's `oauth/flow.ts`).

PKCE authorization-code flow, dynamic client registration, token refresh, and
step-up authorization, plus `adapt_oauth_provider`, which turns an
`OAuthClientProvider` into the transport's `AuthProvider` with one refresh
shared by concurrent 401s.

Providers are duck-typed: the required members are `redirect_url`,
`client_metadata`, `client_information`, `tokens`, `save_tokens`,
`redirect_to_authorization`, `save_code_verifier`, and `code_verifier`; the
optional ones (`state`, `save_client_information`, `client_metadata_url`,
`add_client_authentication`, `invalidate_credentials`, `save_discovery_state`,
`discovery_state`) are looked up with `getattr`, as pi looks up its optional
interface members. Any method may be sync or async.

Envelopes this module builds or reads (the challenge dict, the discovery
state) use the snake_case keys of `discovery.py`, not pi's camelCase — they
are internal to this package, unlike the server documents, which keep their
wire shape.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..auth_provider import McpFetch, UnauthorizedContext
from ..protocol.jsonrpc import is_object
from ..transports.http_client import HttpRequest, http_fetch
from .discovery import (
    discover_authorization_server_metadata,
    discover_oauth_server_info,
    parse_www_authenticate,
    select_resource,
)
from .errors import (
    McpOAuthAuthorizationRequiredError,
    OAuthError,
    OAuthInsecureEndpointError,
    OAuthIssuerMismatchError,
    OAuthRegistrationError,
)
from .types import parse_client_information, parse_oauth_tokens

__all__ = [
    "AddClientAuthentication",
    "AuthorizationStart",
    "OAuthFlowOptions",
    "adapt_oauth_provider",
    "authorize_mcp",
    "exchange_authorization_code",
    "refresh_authorization",
    "register_client",
    "start_authorization",
    "step_up_scope",
]

#: pi's `AddClientAuthentication`: `(headers, params, url, metadata) -> None`,
#: mutating the request headers and form parameters to authenticate the client.
AddClientAuthentication = Callable[[Dict[str, str], Dict[str, str], str, Optional[Dict[str, Any]]], Any]

#: pi's `ClientAuthMethod`.
_ClientAuthMethod = str  # "client_secret_basic" | "client_secret_post" | "none"


@dataclass
class OAuthFlowOptions:
    """pi's `OAuthFlowOptions`. URLs are plain strings in the port."""

    server_url: str
    authorization_code: Optional[str] = None
    #: `iss` parameter of the authorization response that delivered
    #: `authorization_code` (RFC 9207).
    iss: Optional[str] = None
    scope: Optional[str] = None
    resource_metadata_url: Optional[str] = None
    #: Authorization server metadata document to use instead of discovery, for
    #: servers that advertise a wrong authorization server or none. It is
    #: trusted as configured. Must use https, except on loopback.
    authorization_server_metadata_url: Optional[str] = None
    fetch: Optional[McpFetch] = None
    skip_issuer_validation: bool = False
    #: Go straight to the authorization redirect instead of refreshing stored
    #: tokens, for example when the server asks for scopes the current grant
    #: lacks (a refresh keeps the old scope).
    skip_refresh: bool = False


@dataclass
class AuthorizationStart:
    """What `start_authorization` returns (pi's `{authorizationUrl, codeVerifier}`)."""

    authorization_url: str
    code_verifier: str


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def _call_optional(provider: Any, name: str, *args: Any) -> Any:
    """Call an optional provider member; `None` when the provider lacks it."""
    method = getattr(provider, name, None)
    if method is None:
        return None
    return await _maybe_await(method(*args))


def _loopback(hostname: str) -> bool:
    return hostname in ("localhost", "127.0.0.1", "[::1]", "::1")


def _secure_endpoint(value: str) -> str:
    """Credentials may only go to https endpoints, or to a loopback one."""
    parts = urlsplit(str(value))
    if parts.scheme != "https" and not _loopback(parts.hostname or ""):
        raise OAuthInsecureEndpointError(str(value))
    return str(value)


def _resolve_url(base: str, path: str) -> str:
    """`new URL(path, base)` for an absolute path: the base's origin plus `path`."""
    parts = urlsplit(base)
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _select_client_auth_method(information: Dict[str, Any], supported: List[str]) -> _ClientAuthMethod:
    hinted = information.get("token_endpoint_auth_method")
    if hinted in ("client_secret_basic", "client_secret_post", "none") and (not supported or hinted in supported):
        return hinted
    if not supported:
        return "client_secret_basic" if information.get("client_secret") else "none"
    if information.get("client_secret") and "client_secret_basic" in supported:
        return "client_secret_basic"
    if information.get("client_secret") and "client_secret_post" in supported:
        return "client_secret_post"
    if "none" in supported:
        return "none"
    return "client_secret_post" if information.get("client_secret") else "none"


def _apply_client_authentication(
    method: _ClientAuthMethod,
    information: Dict[str, Any],
    headers: Dict[str, str],
    params: Dict[str, str],
) -> None:
    if method == "client_secret_basic":
        if not information.get("client_secret"):
            raise Exception("client_secret_basic requires a client secret")
        credentials = base64.b64encode(f"{information['client_id']}:{information['client_secret']}".encode()).decode()
        headers["Authorization"] = f"Basic {credentials}"
    else:
        params["client_id"] = information["client_id"]
        if method == "client_secret_post" and information.get("client_secret"):
            params["client_secret"] = information["client_secret"]


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _pkce() -> Tuple[str, str]:
    verifier = _base64url(secrets.token_bytes(32))
    challenge = _base64url(hashlib.sha256(verifier.encode("utf-8")).digest())
    return verifier, challenge


def start_authorization(
    authorization_server_url: str,
    *,
    client_information: Dict[str, Any],
    redirect_url: str,
    metadata: Optional[Dict[str, Any]] = None,
    scope: Optional[str] = None,
    state: Optional[str] = None,
    resource: Optional[str] = None,
) -> AuthorizationStart:
    """Build the authorization URL and the PKCE verifier for it."""
    if metadata is not None and "code" not in metadata.get("response_types_supported", []):
        raise Exception("Authorization server does not support authorization codes")
    if metadata is not None:
        methods = metadata.get("code_challenge_methods_supported")
        if methods and "S256" not in methods:
            raise Exception("Authorization server does not support PKCE S256")
    endpoint = (metadata or {}).get("authorization_endpoint") or _resolve_url(authorization_server_url, "/authorize")
    parts = urlsplit(endpoint)
    # `keep_blank_values` with per-key replacement: like pi's
    # `searchParams.set`, each key is replaced in place (or appended);
    # duplicated pre-existing keys collapse to the last, which is what
    # `dict(parse_qsl(...))` would keep anyway.
    params = {}
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        params[key] = value
    verifier, challenge = _pkce()
    params["response_type"] = "code"
    params["client_id"] = client_information["client_id"]
    params["code_challenge"] = challenge
    params["code_challenge_method"] = "S256"
    params["redirect_uri"] = str(redirect_url)
    if state:
        params["state"] = state
    if scope:
        params["scope"] = scope
    if scope and "offline_access" in scope.split():
        params["prompt"] = "consent"
    if resource:
        params["resource"] = resource
    url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))
    return AuthorizationStart(authorization_url=url, code_verifier=verifier)


async def _token_request(
    authorization_server_url: str,
    *,
    client_information: Dict[str, Any],
    params: Dict[str, str],
    metadata: Optional[Dict[str, Any]] = None,
    resource: Optional[str] = None,
    add_client_authentication: Optional[AddClientAuthentication] = None,
    fetch: Optional[McpFetch] = None,
) -> Dict[str, Any]:
    endpoint = _secure_endpoint((metadata or {}).get("token_endpoint") or _resolve_url(authorization_server_url, "/token"))
    headers = {"Accept": "application/json", "content-type": "application/x-www-form-urlencoded"}
    if resource:
        params["resource"] = resource
    if add_client_authentication is not None:
        await _maybe_await(add_client_authentication(headers, params, endpoint, metadata))
    else:
        _apply_client_authentication(
            _select_client_auth_method(client_information, (metadata or {}).get("token_endpoint_auth_methods_supported") or []),
            client_information,
            headers,
            params,
        )
    response = await (fetch or http_fetch)(
        endpoint,
        HttpRequest(method="POST", headers=headers, body=urlencode(params).encode("utf-8")),
    )
    text = await response.text()
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    # Servers may report OAuth errors with any status, so check the body before the status.
    if is_object(value) and isinstance(value.get("error"), str):
        description = value.get("error_description")
        error_uri = value.get("error_uri")
        raise OAuthError(
            value["error"],
            description if isinstance(description, str) else value["error"],
            error_uri if isinstance(error_uri, str) else None,
        )
    if not response.ok:
        raise OAuthError("server_error", f"HTTP {response.status}: {text}")
    return parse_oauth_tokens(value)


async def register_client(
    authorization_server_url: str,
    *,
    client_metadata: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    scope: Optional[str] = None,
    fetch: Optional[McpFetch] = None,
) -> Dict[str, Any]:
    """Dynamic client registration (RFC 7591)."""
    endpoint = (metadata or {}).get("registration_endpoint")
    if metadata is not None and not endpoint:
        raise Exception("Authorization server does not support dynamic client registration")
    body = dict(client_metadata)
    if scope:
        body["scope"] = scope
    response = await (fetch or http_fetch)(
        endpoint or _resolve_url(authorization_server_url, "/register"),
        HttpRequest(
            method="POST",
            headers={"Accept": "application/json", "content-type": "application/json"},
            body=json.dumps(body).encode("utf-8"),
        ),
    )
    if not response.ok:
        raise OAuthRegistrationError(response.status, await response.text())
    return parse_client_information(await response.json())


async def exchange_authorization_code(
    authorization_server_url: str,
    *,
    client_information: Dict[str, Any],
    code: str,
    code_verifier: str,
    redirect_url: str,
    metadata: Optional[Dict[str, Any]] = None,
    resource: Optional[str] = None,
    add_client_authentication: Optional[AddClientAuthentication] = None,
    fetch: Optional[McpFetch] = None,
) -> Dict[str, Any]:
    return await _token_request(
        authorization_server_url,
        client_information=client_information,
        params={
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": str(redirect_url),
        },
        metadata=metadata,
        resource=resource,
        add_client_authentication=add_client_authentication,
        fetch=fetch,
    )


async def refresh_authorization(
    authorization_server_url: str,
    *,
    client_information: Dict[str, Any],
    refresh_token: str,
    metadata: Optional[Dict[str, Any]] = None,
    resource: Optional[str] = None,
    add_client_authentication: Optional[AddClientAuthentication] = None,
    fetch: Optional[McpFetch] = None,
) -> Dict[str, Any]:
    tokens = await _token_request(
        authorization_server_url,
        client_information=client_information,
        params={"grant_type": "refresh_token", "refresh_token": refresh_token},
        metadata=metadata,
        resource=resource,
        add_client_authentication=add_client_authentication,
        fetch=fetch,
    )
    # A refresh response without a new refresh token keeps the current one.
    return {"refresh_token": refresh_token, **tokens}


def _with_scope(tokens: Dict[str, Any], scope: Optional[str]) -> Dict[str, Any]:
    """A response without `scope` grants the requested/kept scope (RFC 6749);
    record it so a later step-up can build on it."""
    if tokens.get("scope") is None and scope:
        return {**tokens, "scope": scope}
    return tokens


def step_up_scope(granted: Optional[str], challenged: Optional[str]) -> Optional[str]:
    """Scopes for a step-up authorization: the challenged scopes plus the ones
    granted so far, since a challenge may list only the missing scopes and a
    token with just those would lose access the old one had (SEP-2350).
    Without challenged scopes, `None` lets the flow pick its default."""
    if not challenged:
        return None
    scopes: List[str] = []
    for value in (granted, challenged):
        if value:
            scopes.extend(value.split())
    seen = set()
    unique = [item for item in scopes if not (item in seen or seen.add(item))]
    return " ".join(unique)


async def _run_flow(provider: Any, options: OAuthFlowOptions) -> str:
    metadata_url = (
        _secure_endpoint(options.authorization_server_metadata_url)
        if options.authorization_server_metadata_url
        else None
    )
    # With a configured metadata URL, discovery is not cached, so changing the
    # URL applies at once.
    cached = None if metadata_url else await _call_optional(provider, "discovery_state")
    if cached is not None and cached.get("authorization_server_url"):
        server_metadata = cached.get("authorization_server_metadata")
        if server_metadata is None:
            server_metadata = await discover_authorization_server_metadata(
                cached["authorization_server_url"],
                fetch=options.fetch,
                skip_issuer_validation=options.skip_issuer_validation,
            )
        discovered = {
            "authorization_server_url": cached["authorization_server_url"],
            "authorization_server_metadata": server_metadata,
            "resource_metadata": cached.get("resource_metadata"),
        }
    else:
        discovered = await discover_oauth_server_info(
            options.server_url,
            resource_metadata_url=options.resource_metadata_url,
            authorization_server_metadata_url=metadata_url,
            fetch=options.fetch,
            skip_issuer_validation=options.skip_issuer_validation,
        )
    if not metadata_url:
        discovery_state = dict(discovered)
        if options.resource_metadata_url:
            discovery_state["resource_metadata_url"] = options.resource_metadata_url
        await _call_optional(provider, "save_discovery_state", discovery_state)
    metadata = discovered["authorization_server_metadata"]
    resource = select_resource(options.server_url, discovered["resource_metadata"])
    # `or`, not a None-coalesce: an empty scope (for example from
    # `scopes_supported: []`) falls through to the next source.
    scope = (
        options.scope
        or " ".join((discovered["resource_metadata"] or {}).get("scopes_supported") or [])
        or (getattr(provider, "client_metadata", None) or {}).get("scope")
    )
    client = await _maybe_await(provider.client_information())
    if not client:
        if options.authorization_code:
            raise Exception("OAuth client information is missing during code exchange")
        client_metadata_url = getattr(provider, "client_metadata_url", None)
        if (metadata or {}).get("client_id_metadata_document_supported") and client_metadata_url:
            parts = urlsplit(str(client_metadata_url))
            if parts.scheme != "https" or parts.path in ("", "/"):
                raise Exception("Invalid OAuth client metadata URL")
            client = {"client_id": str(client_metadata_url)}
            await _call_optional(provider, "save_client_information", client)
        else:
            if getattr(provider, "save_client_information", None) is None:
                raise Exception("OAuth client information cannot be persisted")
            client = await register_client(
                discovered["authorization_server_url"],
                client_metadata=provider.client_metadata,
                metadata=metadata,
                scope=scope,
                fetch=options.fetch,
            )
            await _maybe_await(provider.save_client_information(client))
    token_options = {
        "client_information": client,
        "metadata": metadata,
        "resource": resource,
        "add_client_authentication": getattr(provider, "add_client_authentication", None),
        "fetch": options.fetch,
    }
    if options.authorization_code:
        # RFC 9207: never send a code from another authorization server to this one.
        iss = options.iss
        if metadata is not None and (iss is not None or metadata.get("authorization_response_iss_parameter_supported")):
            if iss != metadata.get("issuer"):
                raise OAuthIssuerMismatchError(metadata.get("issuer"), iss)
        tokens = await exchange_authorization_code(
            discovered["authorization_server_url"],
            **token_options,
            code=options.authorization_code,
            code_verifier=await _maybe_await(provider.code_verifier()),
            redirect_url=provider.redirect_url,
        )
        # A response without `scope` grants the requested scope (RFC 6749 §5.1).
        # Callers pass the options of the authorization request, so `scope` is
        # what was requested.
        await _maybe_await(provider.save_tokens(_with_scope(tokens, scope)))
        return "AUTHORIZED"
    existing = None if options.skip_refresh else await _maybe_await(provider.tokens())
    if existing and existing.get("refresh_token"):
        try:
            tokens = await refresh_authorization(
                discovered["authorization_server_url"],
                **token_options,
                refresh_token=existing["refresh_token"],
            )
            # A refresh without `scope` keeps the scope of the grant (RFC 6749 §6).
            await _maybe_await(provider.save_tokens(_with_scope(tokens, existing.get("scope"))))
            return "AUTHORIZED"
        except OAuthInsecureEndpointError:
            raise
        except OAuthError as error:
            if error.code != "server_error":
                raise
        except Exception:
            # A network failure during refresh falls through to a fresh grant.
            pass
    state = await _call_optional(provider, "state")
    start = start_authorization(
        discovered["authorization_server_url"],
        client_information=client,
        redirect_url=provider.redirect_url,
        metadata=metadata,
        scope=scope,
        state=state,
        resource=resource,
    )
    await _maybe_await(provider.save_code_verifier(start.code_verifier))
    await _maybe_await(provider.redirect_to_authorization(start.authorization_url))
    return "REDIRECT"


async def authorize_mcp(provider: Any, options: OAuthFlowOptions) -> str:
    """Run the flow, re-running it once without the credentials the server
    just rejected (`"AUTHORIZED" | "REDIRECT"`)."""
    try:
        return await _run_flow(provider, options)
    except OAuthError as error:
        if error.code in ("invalid_client", "unauthorized_client"):
            await _call_optional(provider, "invalidate_credentials", "all")
            return await _run_flow(provider, options)
        if error.code == "invalid_grant":
            await _call_optional(provider, "invalidate_credentials", "tokens")
            return await _run_flow(provider, options)
        raise


class _AdaptedOAuthProvider:
    """pi's `adaptOAuthProvider` result: an `AuthProvider` for
    `StreamableHttpTransport`. After a 401 it refreshes the tokens, or raises
    `McpOAuthAuthorizationRequiredError` when the user has to authorize
    (again). Concurrent 401s share one refresh, and a request whose token was
    already replaced is just retried: with rotating refresh tokens, a second
    refresh with the old refresh token would fail and discard the new grant."""

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self._in_flight: "Optional[asyncio.Task[None]]" = None

    async def token(self) -> Optional[str]:
        tokens = await _maybe_await(self._provider.tokens())
        return (tokens or {}).get("access_token")

    async def on_unauthorized(self, context: UnauthorizedContext) -> None:
        challenge = parse_www_authenticate(context.response.header("www-authenticate"))
        insufficient_scope = challenge.get("error") == "insufficient_scope"
        if not insufficient_scope and self._in_flight is None and context.token is not None:
            current = (await _maybe_await(self._provider.tokens()) or {}).get("access_token")
            if current is not None and current != context.token:
                return
        if self._in_flight is None:
            self._in_flight = asyncio.ensure_future(self._authorize(context, challenge, insufficient_scope))
            self._in_flight.add_done_callback(lambda _task: setattr(self, "_in_flight", None))
        # Await through a shield: awaiting a Task directly lets one waiter's
        # cancellation cancel the shared flow for every other waiter. pi's
        # abandoned request only observes the promise, never cancels it.
        await asyncio.shield(self._in_flight)

    async def _authorize(self, context: UnauthorizedContext, challenge: Dict[str, Any], insufficient_scope: bool) -> None:
        granted = await _maybe_await(self._provider.tokens()) if insufficient_scope else None
        result = await authorize_mcp(
            self._provider,
            OAuthFlowOptions(
                server_url=context.server_url,
                resource_metadata_url=challenge.get("resource_metadata_url"),
                scope=(
                    step_up_scope((granted or {}).get("scope"), challenge.get("scope"))
                    if insufficient_scope
                    else challenge.get("scope")
                ),
                fetch=context.fetch,
                skip_refresh=insufficient_scope,
            ),
        )
        if result == "REDIRECT":
            raise McpOAuthAuthorizationRequiredError()


def adapt_oauth_provider(provider: Any) -> _AdaptedOAuthProvider:
    return _AdaptedOAuthProvider(provider)

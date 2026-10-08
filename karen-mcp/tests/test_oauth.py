"""OAuth tests (pi's `test/oauth.test.ts`).

The fetch double is the real `http_fetch` against the loopback server, exactly
as pi drives its own fetch against `node:http` servers.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from urllib.parse import parse_qsl, urlsplit

import pytest

from karen_mcp import McpClient, McpClientOptions, StreamableHttpTransport, StreamableHttpTransportOptions
from karen_mcp.auth_provider import UnauthorizedContext
from karen_mcp.oauth import (
    MemoryOAuthStateStore,
    McpOAuthAuthorizationRequiredError,
    McpOAuthProvider,
    McpOAuthProviderOptions,
    OAuthCallbackServer,
    OAuthCallbackServerOptions,
    OAuthFlowOptions,
    OAuthInsecureEndpointError,
    OAuthIssuerMismatchError,
    adapt_oauth_provider,
    authorize_mcp,
    discover_authorization_server_metadata,
    parse_protected_resource_metadata,
    parse_www_authenticate,
    select_resource,
    start_authorization,
)
from karen_mcp.transports.http_client import HttpRequest, http_fetch


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _query(url: str) -> dict:
    return dict(parse_qsl(urlsplit(url).query))


class StubResponse:
    """The bits of `HttpResponse` an `UnauthorizedContext` needs."""

    def __init__(self, status: int, headers: dict | None = None) -> None:
        self.status = status
        self._headers = {name.lower(): value for name, value in (headers or {}).items()}

    def header(self, name: str, default=None):
        return self._headers.get(name.lower(), default)


class TestOAuthProvider:
    """pi's `TestOAuthProvider`: everything in memory, every method async."""

    #: Named after pi's double, not a test class.
    __test__ = False

    def __init__(self, redirect_url: str) -> None:
        self.redirect_url = redirect_url
        self.client_metadata = {
            "redirect_uris": [redirect_url],
            "client_name": "pi-mcp-test",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }
        self.client = None
        self.token_set = None
        self.verifier = None
        self.discovery = None
        self.authorization_url = None

    async def state(self):
        return "expected-state"

    async def client_information(self):
        return self.client

    async def save_client_information(self, information):
        self.client = information

    async def tokens(self):
        return self.token_set

    async def save_tokens(self, tokens):
        self.token_set = tokens

    async def redirect_to_authorization(self, url):
        self.authorization_url = url

    async def save_code_verifier(self, verifier):
        self.verifier = verifier

    async def code_verifier(self):
        if not self.verifier:
            raise Exception("Missing code verifier")
        return self.verifier

    async def invalidate_credentials(self, kind):
        if kind in ("all", "client"):
            self.client = None
        if kind in ("all", "tokens"):
            self.token_set = None
        if kind in ("all", "verifier"):
            self.verifier = None
        if kind in ("all", "discovery"):
            self.discovery = None

    async def save_discovery_state(self, state):
        self.discovery = state

    async def discovery_state(self):
        return self.discovery


async def _get(url: str):
    response = await http_fetch(url, HttpRequest(method="GET", headers={}))
    return response


async def test_discovers_registers_authorizes_with_pkce_and_refreshes_on_401(listen):
    expected_challenge = []
    refreshes = [0]

    def handler(request, response, requests):
        path = request.url.path
        query = _query(request.path)
        if path == "/oauth-resource-meta":
            # The challenge advertises this path (not the derived well-known
            # one), so a flow that ignored the advertisement gets a 404.
            origin = request.header("host")
            response.set_header("content-type", "application/json")
            response.end(
                json.dumps(
                    {
                        "resource": f"http://{origin}/mcp",
                        "authorization_servers": [f"http://{origin}"],
                        "scopes_supported": ["org:read"],
                    }
                )
            )
            return
        if path == "/.well-known/oauth-authorization-server":
            origin = request.header("host")
            response.set_header("content-type", "application/json")
            response.end(
                json.dumps(
                    {
                        "issuer": f"http://{origin}",
                        "authorization_endpoint": f"http://{origin}/authorize",
                        "token_endpoint": f"http://{origin}/token",
                        "registration_endpoint": f"http://{origin}/register",
                        "response_types_supported": ["code"],
                        "grant_types_supported": ["authorization_code", "refresh_token"],
                        "token_endpoint_auth_methods_supported": ["none"],
                        "code_challenge_methods_supported": ["S256"],
                    }
                )
            )
            return
        if path == "/register":
            metadata = json.loads(request.body.decode("utf-8"))
            response.write_head(201, {"content-type": "application/json"})
            # Empty and null optional fields count as absent (#10266).
            response.end(json.dumps({**metadata, "client_id": "test-client", "client_secret": ""}))
            return
        if path == "/authorize":
            expected_challenge.append(query.get("code_challenge"))
            redirect = query.get("redirect_uri") or ""
            separator = "&" if urlsplit(redirect).query else "?"
            response.write_head(
                302, {"location": f"{redirect}{separator}code=test-code&state={query.get('state') or ''}"}
            )
            response.end()
            return
        if path == "/token":
            params = dict(parse_qsl(request.body.decode("utf-8")))
            if params.get("grant_type") == "refresh_token":
                refreshes[0] += 1
                response.set_header("content-type", "application/json")
                response.end(
                    json.dumps(
                        {
                            "access_token": "refreshed-token",
                            "token_type": "Bearer",
                            "refresh_token": "",
                            "expires_in": None,
                        }
                    )
                )
                return
            challenge = _base64url(hashlib.sha256((params.get("code_verifier") or "").encode()).digest())
            if params.get("code") != "test-code" or challenge != expected_challenge[0]:
                response.write_head(400, {"content-type": "application/json"})
                response.end(json.dumps({"error": "invalid_grant"}))
                return
            response.set_header("content-type", "application/json")
            response.end(
                json.dumps(
                    {
                        "access_token": "first-token",
                        "refresh_token": "refresh-token",
                        "token_type": "Bearer",
                        # An explicit grant scope that differs from the
                        # requested `scopes_supported`, so a refresh keeping
                        # the requested scope instead of the grant's is caught.
                        "scope": "org:read extra",
                    }
                )
            )
            return
        if path != "/mcp":
            response.write_head(404)
            response.end()
            return
        if request.method == "GET":
            response.write_head(405)
            response.end()
            return
        if request.method == "DELETE":
            response.write_head(200)
            response.end()
            return
        token = request.header("authorization")
        if token not in ("Bearer first-token", "Bearer refreshed-token"):
            response.write_head(
                401,
                {
                    # An empty scope falls through to the resource metadata's scopes_supported.
                    # The advertised URL is deliberately not the derived well-known path.
                    "www-authenticate": 'Bearer resource_metadata="{origin}/oauth-resource-meta", scope=""'.format(
                        origin=f"http://{request.header('host')}"
                    )
                },
            )
            response.end("Unauthorized")
            return
        message = request.message or {}
        if "id" not in message:
            response.write_head(202)
            response.end()
            return
        result = (
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "oauth-test", "version": "1.0.0"},
            }
            if message.get("method") == "initialize"
            else {"tools": [{"name": "issues", "inputSchema": {"type": "object"}}]}
        )
        response.set_header("content-type", "application/json")
        response.end(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}))

    server = listen(handler)
    origin = server.origin

    callback = await OAuthCallbackServer.listen()
    provider = TestOAuthProvider(callback.redirect_url)
    first_client = McpClient(McpClientOptions(name="oauth-test", version="1.0.0"))
    with pytest.raises(McpOAuthAuthorizationRequiredError):
        await first_client.connect(
            StreamableHttpTransport(
                StreamableHttpTransportOptions(
                    url=f"{origin}/mcp",
                    headers={"Authorization": "Bearer caller-supplied-stale-token"},
                    auth_provider=adapt_oauth_provider(provider),
                    open_get_stream=False,
                )
            )
        )
    assert _query(provider.authorization_url).get("scope") == "org:read"
    assert _query(provider.authorization_url).get("resource") == f"{origin}/mcp"
    assert _query(provider.authorization_url).get("state") == "expected-state"

    pending = callback.wait_for_callback("expected-state")
    authorization_response = await _get(provider.authorization_url)
    location = authorization_response.header("location")
    await authorization_response.close()
    assert location
    redirect_response = await _get(location)
    await redirect_response.close()
    callback_result = await asyncio.wait_for(pending, 10)
    assert callback_result.state == "expected-state"
    result = await authorize_mcp(
        provider,
        OAuthFlowOptions(server_url=f"{origin}/mcp", authorization_code=callback_result.code, fetch=http_fetch),
    )
    assert result == "AUTHORIZED"

    client = McpClient(McpClientOptions(name="oauth-test", version="1.0.0"))
    await client.connect(
        StreamableHttpTransport(
            StreamableHttpTransportOptions(
                url=f"{origin}/mcp",
                headers={"Authorization": "Bearer caller-supplied-stale-token"},
                auth_provider=adapt_oauth_provider(provider),
                open_get_stream=False,
            )
        )
    )
    tools = await client.list_tools()
    assert [(tool.name, tool.input_schema) for tool in tools] == [("issues", {"type": "object"})]
    await client.close()

    provider.token_set = {**provider.token_set, "access_token": "stale-token"}
    refreshed_client = McpClient(McpClientOptions(name="oauth-test", version="1.0.0"))
    await refreshed_client.connect(
        StreamableHttpTransport(
            StreamableHttpTransportOptions(
                url=f"{origin}/mcp",
                headers={"Authorization": "Bearer caller-supplied-stale-token"},
                auth_provider=adapt_oauth_provider(provider),
                open_get_stream=False,
            )
        )
    )
    # Neither refresh response names a scope, so the grant keeps its own.
    assert provider.token_set == {
        "access_token": "refreshed-token",
        "refresh_token": "refresh-token",
        "token_type": "Bearer",
        "scope": "org:read extra",
    }
    assert refreshes[0] == 1
    await refreshed_client.close()
    await callback.close()


async def test_shares_one_refresh_between_concurrent_401s_when_refresh_tokens_rotate(listen):
    grants = []

    def handler(request, response, requests):
        path = request.url.path
        if path == "/.well-known/oauth-protected-resource/mcp":
            origin = request.header("host")
            # Invalid resource metadata falls back to the server origin instead of failing discovery.
            response.set_header("content-type", "application/json")
            response.end(
                json.dumps({"resource": f"http://{origin}/mcp", "authorization_servers": ["not a url"]})
            )
            return
        if path == "/.well-known/oauth-authorization-server":
            origin = request.header("host")
            response.set_header("content-type", "application/json")
            response.end(
                json.dumps(
                    {
                        # Issuer without the trailing slash that URL parsing adds to the fallback server URL.
                        "issuer": f"http://{origin}",
                        "authorization_endpoint": f"http://{origin}/authorize",
                        "token_endpoint": f"http://{origin}/token",
                        "response_types_supported": ["code"],
                    }
                )
            )
            return
        if path == "/token":
            params = dict(parse_qsl(request.body.decode("utf-8")))
            refresh_token = params.get("refresh_token") or ""
            grants.append(refresh_token)
            response.set_header("content-type", "application/json")
            if refresh_token != "r1":
                response.write_head(400)
                response.end(json.dumps({"error": "invalid_grant"}))
                return
            time.sleep(0.02)
            response.end(
                json.dumps({"access_token": "a2", "refresh_token": "r2", "token_type": "Bearer", "expires_in": 3600})
            )
            return
        response.write_head(404)
        response.end()

    server = listen(handler)
    origin = server.origin
    store = MemoryOAuthStateStore()
    provider = McpOAuthProvider(
        McpOAuthProviderOptions(
            server_url=f"{origin}/mcp",
            redirect_url="http://127.0.0.1/callback",
            client_metadata={"client_name": "test"},
            client_id="client",
            store=store,
            on_redirect=lambda url: None,
        )
    )
    await provider.save_tokens({"access_token": "a1", "refresh_token": "r1", "token_type": "Bearer"})
    auth = adapt_oauth_provider(provider)

    def unauthorized():
        return UnauthorizedContext(
            response=StubResponse(401, {"www-authenticate": "Bearer"}),
            server_url=f"{origin}/mcp",
            fetch=http_fetch,
            token="a1",
        )

    await asyncio.gather(auth.on_unauthorized(unauthorized()), auth.on_unauthorized(unauthorized()))
    # A late 401 for a request that still carried the old token must not refresh again.
    await auth.on_unauthorized(unauthorized())
    assert grants == ["r1"]
    assert await auth.token() == "a2"
    state = store.load()
    assert state["tokens"]["refresh_token"] == "r2"
    assert state["tokens_expire_at"] > time.time() * 1000 + 3_500_000


async def test_a_cancelled_waiter_does_not_cancel_the_shared_refresh(listen):
    def handler(request, response, requests):
        path = request.url.path
        if path == "/.well-known/oauth-protected-resource/mcp":
            origin = request.header("host")
            response.set_header("content-type", "application/json")
            response.end(json.dumps({"resource": f"http://{origin}/mcp", "authorization_servers": ["not a url"]}))
            return
        if path == "/.well-known/oauth-authorization-server":
            origin = request.header("host")
            response.set_header("content-type", "application/json")
            response.end(
                json.dumps(
                    {
                        "issuer": f"http://{origin}",
                        "authorization_endpoint": f"http://{origin}/authorize",
                        "token_endpoint": f"http://{origin}/token",
                        "response_types_supported": ["code"],
                    }
                )
            )
            return
        if path == "/token":
            time.sleep(0.05)
            response.set_header("content-type", "application/json")
            response.end(
                json.dumps({"access_token": "a2", "refresh_token": "r2", "token_type": "Bearer", "expires_in": 3600})
            )
            return
        response.write_head(404)
        response.end()

    server = listen(handler)
    origin = server.origin
    provider = McpOAuthProvider(
        McpOAuthProviderOptions(
            server_url=f"{origin}/mcp",
            redirect_url="http://127.0.0.1/callback",
            client_metadata={"client_name": "test"},
            client_id="client",
            store=MemoryOAuthStateStore(),
            on_redirect=lambda url: None,
        )
    )
    await provider.save_tokens({"access_token": "a1", "refresh_token": "r1", "token_type": "Bearer"})
    auth = adapt_oauth_provider(provider)

    def unauthorized():
        return UnauthorizedContext(
            response=StubResponse(401, {"www-authenticate": "Bearer"}),
            server_url=f"{origin}/mcp",
            fetch=http_fetch,
            token="a1",
        )

    cancelled = asyncio.ensure_future(auth.on_unauthorized(unauthorized()))
    surviving = asyncio.ensure_future(auth.on_unauthorized(unauthorized()))
    await asyncio.sleep(0.01)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    # The other waiter still completes the one refresh they were sharing.
    await surviving
    assert await auth.token() == "a2"


async def test_asks_for_authorization_instead_of_refreshing_when_the_server_needs_more_scope(listen):
    provider = TestOAuthProvider("http://127.0.0.1/callback")
    provider.client = {"client_id": "client"}
    provider.token_set = {"access_token": "a1", "refresh_token": "r1", "token_type": "Bearer", "scope": "repo read:org"}
    token_requests = []

    def handler(request, response, requests):
        path = request.url.path
        if path == "/.well-known/oauth-authorization-server":
            origin = request.header("host")
            response.set_header("content-type", "application/json")
            response.end(
                json.dumps(
                    {
                        "issuer": f"http://{origin}",
                        "authorization_endpoint": f"http://{origin}/authorize",
                        "token_endpoint": f"http://{origin}/token",
                        "response_types_supported": ["code"],
                    }
                )
            )
            return
        if path == "/token":
            token_requests.append(request)
        response.write_head(500 if path == "/token" else 404)
        response.end()

    server = listen(handler)
    origin = server.origin
    auth = adapt_oauth_provider(provider)
    with pytest.raises(McpOAuthAuthorizationRequiredError):
        await auth.on_unauthorized(
            UnauthorizedContext(
                response=StubResponse(
                    403, {"www-authenticate": 'Bearer error="insufficient_scope", scope="repo admin"'}
                ),
                server_url=f"{origin}/mcp",
                fetch=http_fetch,
                token="a1",
            )
        )
    # The challenge may list only the missing scopes; the new grant keeps the old ones too.
    assert _query(provider.authorization_url).get("scope") == "repo read:org admin"
    # The working grant is kept until the user authorizes the new scope.
    assert provider.token_set["access_token"] == "a1"
    # A step-up never refreshes: a refresh here could discard the working grant
    # on a server that rotates refresh tokens.
    assert token_requests == []


async def test_binds_persisted_credentials_to_the_exact_mcp_server_url():
    store = MemoryOAuthStateStore()
    first = McpOAuthProvider(
        McpOAuthProviderOptions(
            server_url="https://one.example/mcp",
            redirect_url="http://127.0.0.1/callback",
            client_metadata={"client_name": "test"},
            store=store,
            on_redirect=lambda url: None,
        )
    )
    await first.save_tokens({"access_token": "secret", "token_type": "Bearer"})
    assert (await first.tokens())["access_token"] == "secret"

    second = McpOAuthProvider(
        McpOAuthProviderOptions(
            server_url="https://two.example/mcp",
            redirect_url="http://127.0.0.1/callback",
            client_metadata={"client_name": "test"},
            store=store,
            on_redirect=lambda url: None,
        )
    )
    assert await second.tokens() is None


async def test_rejects_authorization_metadata_whose_issuer_does_not_match_discovery(listen):
    def handler(request, response, requests):
        if request.url.path == "/.well-known/oauth-authorization-server":
            origin = request.header("host")
            response.set_header("content-type", "application/json")
            response.end(
                json.dumps(
                    {
                        "issuer": "https://attacker.example",
                        "authorization_endpoint": f"http://{origin}/authorize",
                        "token_endpoint": f"http://{origin}/token",
                        "response_types_supported": ["code"],
                    }
                )
            )
            return
        response.write_head(404)
        response.end()

    server = listen(handler)
    with pytest.raises(OAuthIssuerMismatchError):
        await discover_authorization_server_metadata(server.origin)


# #10172
async def test_uses_a_configured_authorization_server_metadata_document_as_is(listen):
    def handler(request, response, requests):
        path = request.url.path
        origin = request.header("host")
        response.set_header("content-type", "application/json")
        if path == "/.well-known/oauth-protected-resource/mcp":
            # Names the MCP server itself, which serves no authorization server metadata.
            response.end(
                json.dumps({"resource": f"http://{origin}/mcp", "authorization_servers": [f"http://{origin}"]})
            )
        elif path == "/idp/metadata.json":
            response.end(
                json.dumps(
                    {
                        # Not derivable from the document URL; a configured document is not checked.
                        "issuer": "https://idp.example",
                        "authorization_endpoint": f"http://{origin}/idp/authorize",
                        "token_endpoint": f"http://{origin}/idp/token",
                        "response_types_supported": ["code"],
                    }
                )
            )
        else:
            response.write_head(404)
            response.end()

    server = listen(handler)
    origin = server.origin
    provider = TestOAuthProvider("http://127.0.0.1/callback")
    provider.client = {"client_id": "client"}
    options = OAuthFlowOptions(
        server_url=f"{origin}/mcp", authorization_server_metadata_url=f"{origin}/idp/metadata.json"
    )
    assert await authorize_mcp(provider, options) == "REDIRECT"
    authorization_url = provider.authorization_url
    parts = urlsplit(authorization_url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == f"{origin}/idp/authorize"
    assert _query(authorization_url).get("resource") == f"{origin}/mcp"

    insecure = OAuthFlowOptions(
        server_url=f"{origin}/mcp", authorization_server_metadata_url="http://idp.example/metadata.json"
    )
    with pytest.raises(OAuthInsecureEndpointError):
        await authorize_mcp(provider, insecure)


async def test_exchanges_a_code_only_when_its_iss_parameter_names_the_authorization_server(listen):
    codes = []

    def handler(request, response, requests):
        params = dict(parse_qsl(request.body.decode("utf-8")))
        codes.append(params.get("code") or "")
        response.set_header("content-type", "application/json")
        response.end(json.dumps({"access_token": "token", "token_type": "Bearer"}))

    server = listen(handler)
    origin = server.origin

    def exchange(code: str, iss, iss_parameter_supported: bool):
        provider = TestOAuthProvider("http://127.0.0.1/callback")
        provider.client = {"client_id": "client"}
        provider.verifier = "verifier"
        provider.discovery = {
            "authorization_server_url": origin,
            "authorization_server_metadata": {
                "issuer": origin,
                "authorization_endpoint": f"{origin}/authorize",
                "token_endpoint": f"{origin}/token",
                "response_types_supported": ["code"],
                "authorization_response_iss_parameter_supported": iss_parameter_supported,
            },
        }
        return authorize_mcp(
            provider, OAuthFlowOptions(server_url=f"{origin}/mcp", authorization_code=code, iss=iss)
        )

    with pytest.raises(OAuthIssuerMismatchError):
        await exchange("other", "https://attacker.example", False)
    with pytest.raises(OAuthIssuerMismatchError):
        await exchange("missing", None, True)
    assert await exchange("matching", origin, True) == "AUTHORIZED"
    # Servers that do not promise the parameter may omit it.
    assert await exchange("omitted", None, False) == "AUTHORIZED"
    assert codes == ["matching", "omitted"]


def test_start_authorization_builds_the_pkce_request():
    start = start_authorization(
        "https://as.example",
        client_information={"client_id": "c1"},
        redirect_url="http://127.0.0.1:1234/callback",
        scope="repo offline_access",
        state="s1",
        resource="https://mcp.example/mcp",
    )
    parts = urlsplit(start.authorization_url)
    query = dict(parse_qsl(parts.query))
    assert (parts.scheme, parts.netloc, parts.path) == ("https", "as.example", "/authorize")
    assert query["response_type"] == "code"
    assert query["client_id"] == "c1"
    assert query["code_challenge_method"] == "S256"
    assert query["code_challenge"] == _base64url(hashlib.sha256(start.code_verifier.encode()).digest())
    assert query["redirect_uri"] == "http://127.0.0.1:1234/callback"
    assert query["scope"] == "repo offline_access"
    assert query["prompt"] == "consent"
    assert query["state"] == "s1"
    assert query["resource"] == "https://mcp.example/mcp"
    # 32 random bytes, base64url without padding (RFC 7636).
    assert len(start.code_verifier) == 43 and "=" not in start.code_verifier


def test_start_authorization_rejects_servers_without_codes_or_pkce():
    with pytest.raises(Exception, match="authorization codes"):
        start_authorization(
            "https://as.example",
            client_information={"client_id": "c"},
            redirect_url="http://127.0.0.1/callback",
            metadata={"response_types_supported": ["token"]},
        )
    with pytest.raises(Exception, match="PKCE S256"):
        start_authorization(
            "https://as.example",
            client_information={"client_id": "c"},
            redirect_url="http://127.0.0.1/callback",
            metadata={"response_types_supported": ["code"], "code_challenge_methods_supported": ["plain"]},
        )


def test_select_resource_compares_origins_not_their_spelling():
    """A host's case and an explicit default port are the same origin, as
    `URL.origin` comparison in pi treats them."""
    resource = "https://MCP.Example.COM:443/mcp"
    assert select_resource("https://mcp.example.com/mcp", {"resource": resource}) == resource
    with pytest.raises(Exception, match="does not match"):
        select_resource("https://mcp.example.com/mcp", {"resource": "https://evil.example/mcp"})
    with pytest.raises(Exception, match="does not match"):
        select_resource("https://mcp.example.com/other", {"resource": resource})
    assert select_resource("https://mcp.example.com/mcp", None) is None


def test_url_documents_reject_what_new_url_rejects():
    with pytest.raises(ValueError, match="Invalid OAuth protected resource metadata resource"):
        parse_protected_resource_metadata({"resource": "https://exa mple.com/mcp"})
    with pytest.raises(ValueError, match="Invalid OAuth protected resource metadata resource"):
        parse_protected_resource_metadata({"resource": "https://example.com:99999/mcp"})
    with pytest.raises(ValueError, match="Invalid OAuth protected resource metadata resource"):
        parse_protected_resource_metadata({"resource": "javascript:alert(1)"})


def test_parse_www_authenticate_drops_a_malformed_resource_metadata_url():
    """`new URL(...)` in a try block: discovery falls back to the well-known
    path instead of fetching an unparseable URL."""
    challenge = parse_www_authenticate('Bearer resource_metadata="https://exa mple.com/meta", scope="read write"')
    assert "resource_metadata_url" not in challenge
    assert challenge["scope"] == "read write"
    assert parse_www_authenticate('Bearer resource_metadata="not-a-url"') == {}
    # A parseable one is kept.
    challenge = parse_www_authenticate('Bearer resource_metadata="https://example.com/meta"')
    assert challenge["resource_metadata_url"] == "https://example.com/meta"


async def test_provider_normalizes_the_server_url_of_its_own_state():
    store = MemoryOAuthStateStore()

    def provider(server_url):
        return McpOAuthProvider(
            McpOAuthProviderOptions(
                server_url=server_url,
                redirect_url="http://127.0.0.1/callback",
                client_metadata={"client_name": "test"},
                store=store,
                on_redirect=lambda url: None,
            )
        )

    await provider("https://ONE.example:443/mcp").save_tokens({"access_token": "t", "token_type": "Bearer"})
    assert (await provider("https://one.example/mcp").tokens())["access_token"] == "t"


async def test_authorizes_again_when_the_server_rejects_the_stored_grant(listen):
    def handler(request, response, requests):
        path = request.url.path
        if path == "/.well-known/oauth-authorization-server":
            origin = request.header("host")
            response.set_header("content-type", "application/json")
            response.end(
                json.dumps(
                    {
                        "issuer": f"http://{origin}",
                        "authorization_endpoint": f"http://{origin}/authorize",
                        "token_endpoint": f"http://{origin}/token",
                        "response_types_supported": ["code"],
                    }
                )
            )
            return
        if path == "/token":
            response.write_head(400, {"content-type": "application/json"})
            response.end(json.dumps({"error": "invalid_grant"}))
            return
        response.write_head(404)
        response.end()

    server = listen(handler)
    origin = server.origin
    provider = TestOAuthProvider("http://127.0.0.1/callback")
    provider.client = {"client_id": "client"}
    provider.token_set = {"access_token": "a1", "refresh_token": "r1", "token_type": "Bearer"}
    result = await authorize_mcp(provider, OAuthFlowOptions(server_url=f"{origin}/mcp", fetch=http_fetch))
    # `invalid_grant` drops the tokens and runs the flow again, which redirects.
    assert result == "REDIRECT"
    assert provider.token_set is None
    assert provider.authorization_url is not None


async def test_callback_server_renders_plain_text_by_default():
    callback = await OAuthCallbackServer.listen()
    try:
        pending = callback.wait_for_callback("s1")
        response = await _get(f"{callback.redirect_url}?code=abc&state=s1")
        assert response.header("content-type") == "text/plain; charset=utf-8"
        assert await response.text() == "Authorization complete. You may close this window."
        assert (await pending).code == "abc"
    finally:
        await callback.close()


async def test_callback_server_renders_pages_through_render_page():
    pages = []

    def render_page(page):
        pages.append(page)
        return "<p>ok</p>" if page.ok else f"<p>{page.message}</p>"

    callback = await OAuthCallbackServer.listen(OAuthCallbackServerOptions(render_page=render_page))
    try:
        denied = callback.wait_for_callback("s1")
        failure = await _get(f"{callback.redirect_url}?error=access_denied&error_description=Denied&state=s1")
        assert failure.header("content-type") == "text/html; charset=utf-8"
        await failure.close()
        with pytest.raises(Exception, match="Denied"):
            await denied
        assert pages[-1].ok is False
        assert pages[-1].message == "Authorization failed. You may close this window."
        assert pages[-1].details == "Denied"

        pending = callback.wait_for_callback("s2")
        success = await _get(f"{callback.redirect_url}?code=abc&state=s2")
        assert await success.text() == "<p>ok</p>"
        assert (await pending).code == "abc"
    finally:
        await callback.close()


async def test_callback_server_keeps_an_explicit_empty_error_description():
    """pi's `??`: only an absent description falls back to the error code."""
    callback = await OAuthCallbackServer.listen()
    try:
        pending = callback.wait_for_callback("s1")
        response = await _get(f"{callback.redirect_url}?error=access_denied&error_description=&state=s1")
        await response.close()
        with pytest.raises(Exception) as error_info:
            await asyncio.wait_for(pending, 10)
        assert str(error_info.value) == ""
    finally:
        await callback.close()


async def test_callback_server_answers_a_malformed_request_line_with_400():
    """A target `urlsplit` cannot parse gets a 400, not an unhandled task
    exception in `client_connected_cb`."""
    callback = await OAuthCallbackServer.listen()
    try:
        port = urlsplit(callback.redirect_url).port
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET http://[ HTTP/1.1\r\nhost: x\r\n\r\n")
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
        assert head.startswith(b"HTTP/1.1 400")
        writer.close()
    finally:
        await callback.close()

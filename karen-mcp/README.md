# karen-mcp

A small, standalone Model Context Protocol client — a Python port of
[`@earendil-works/pi`](https://github.com/earendil-works/pi)'s `packages/mcp`
(pi commit `f29ea3d`).

Like pi's package it depends on nothing but the standard library (plus pydantic
for the typed protocol objects), and it does not wrap the official MCP SDK. The
package provides a transport-neutral client core, stdio and Streamable HTTP
transports, and an in-memory testing transport.

> Status: the client core, the stdio, in-memory, and Streamable HTTP
> transports, and the `karen_mcp.oauth` OAuth client subset are all in place.

## Usage

```python
from karen_mcp import McpClient, McpClientOptions, StdioTransport, StdioTransportOptions

transport = StdioTransport(
    StdioTransportOptions(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
    )
)
client = McpClient(
    McpClientOptions(
        name="my-client",
        version="1.0.0",
        roots=[Root(uri="file:///workspace", name="workspace")],
    )
)

await client.connect(transport)
tools = await client.list_tools()
result = await client.call_tool("search", {"query": "MCP"})
await client.close()
```

For a remote server, use `StreamableHttpTransport(StreamableHttpTransportOptions(url=..., headers=...))`.
The default HTTP client is the package's own (`http_fetch`, stdlib only); pass
`fetch=` to hand the transport an application's HTTP stack instead.

### Tools for an LLM

`to_llm_content(result)` converts a `CallToolResult` to text and image content
for a model, in the shape of karen-ai's `TextContent` and `ImageContent`. Text
and images pass through, embedded text and image resources are unwrapped, and
audio, resource links, and binary resources become short text placeholders. A
result without content blocks but with `structuredContent` becomes its JSON.

Wrapping an MCP tool as a karen-agent `AgentTool`:

```python
from karen_mcp import to_llm_content

tools = [
    AgentTool(
        # Providers allow at most 64 characters of [A-Za-z0-9_-].
        name=re.sub(r"[^A-Za-z0-9_-]", "_", f"mcp_{tool.name}")[:64],
        label=tool.title or tool.name,
        description=tool.description or tool.name,
        # Providers require an object schema, and some reject one without
        # `properties`.
        parameters={**tool.input_schema, "type": "object", "properties": tool.input_schema.get("properties", {})},
        execute=call_mcp_tool,
    )
    for tool in await client.list_tools()
]
```

### OAuth

`karen_mcp.oauth` provides the MCP OAuth client subset (discovery, PKCE
authorization code flow, dynamic client registration, token refresh, step-up
authorization, and a loopback callback server) without depending on the
official SDK. The package does not open a browser or choose where credentials
are stored — applications inject `McpOAuthStateStore`.

```python
from karen_mcp import McpClient, McpClientOptions, StreamableHttpTransport, StreamableHttpTransportOptions
from karen_mcp.oauth import (
    McpOAuthAuthorizationRequiredError,
    McpOAuthProvider,
    McpOAuthProviderOptions,
    OAuthCallbackServer,
    OAuthFlowOptions,
    adapt_oauth_provider,
    authorize_mcp,
)

server_url = "https://mcp.example.com/mcp"
callback = await OAuthCallbackServer.listen()
oauth = McpOAuthProvider(
    McpOAuthProviderOptions(
        server_url=server_url,
        redirect_url=callback.redirect_url,
        client_metadata={"client_name": "My MCP client"},
        on_redirect=open_in_browser,  # the application decides how
    )
)

def connect():
    client = McpClient(McpClientOptions(name="my-client", version="1.0.0"))
    transport = StreamableHttpTransport(
        StreamableHttpTransportOptions(url=server_url, auth_provider=adapt_oauth_provider(oauth))
    )
    return client, client.connect(transport)

client, connected = connect()
try:
    await connected
except McpOAuthAuthorizationRequiredError:
    pending = callback.wait_for_callback(await oauth.state())
    # `on_redirect` has shown the user the authorization page by now.
    result = await pending
    await authorize_mcp(oauth, OAuthFlowOptions(server_url=server_url, authorization_code=result.code))

client, connected = connect()
await connected
```

The OAuth implementation is adapted from the MIT-licensed Model Context
Protocol TypeScript SDK v1.29.0, by way of pi.

## Supported protocol surface

- MCP protocol version `2025-11-25`, accepting servers that negotiate
  `2025-06-18`, `2025-03-26`, or `2024-11-05`
- initialization and `notifications/initialized`
- ping
- paginated `tools/list`
- `tools/call`, including structured content
- progress notifications and timeout renewal
- request cancellation
- Streamable HTTP sessions, the server-to-client GET stream with reconnection,
  and resumption of dropped response streams with `Last-Event-ID`
- stdio shutdown per the spec (close stdin, then SIGTERM, then SIGKILL),
  applied to the server's whole process group
- server `ping` and `roots/list` requests
- logging and tool-list-change notifications through the generic notification API
- OAuth protected-resource and authorization-server discovery
- PKCE authorization code flow, dynamic client registration, token refresh (one
  refresh shared by concurrent 401s), and step-up authorization for
  `insufficient_scope`

Batch JSON-RPC messages, legacy HTTP+SSE, servers, sampling, and tasks are
outside the core, exactly as in pi.

## Testing

`karen_mcp.testing` exports `create_in_memory_transport_pair()` for client and
adapter tests — no subprocess, no socket.

## Port notes

Deliberate differences from pi, each also documented where it applies:

- The built-in HTTP client makes one request per connection and knows no
  proxies, redirects, cookies, or HTTP/2; hand the transport an application's
  HTTP stack with `fetch=` for those.
- Framing errors from that client (a malformed status line, chunk size, or
  chunk terminator) raise `ValueError` and are not retried: they describe a
  server that is not speaking HTTP. pi retries them, because the platform
  `fetch` reports them as `TypeError`.
- When the server cancels a request it sent us, the client aborts the
  handler's signal and cancels its task, so the cancelled request gets no
  response — the spec says it should not. pi aborts the signal but still
  sends whatever the handler returns.
- URLs are strings handled with `urllib.parse`, without full WHATWG
  normalization (dot segments, IDN punycode, empty queries stay as spelled).
  The places that compare URLs normalize both sides: the origin comparison in
  discovery and the server-URL identity in the provider.
- Helper results are parsed into the pydantic protocol models, so a known
  member with the wrong type is rejected where pi's plain casts would pass it
  through. Unknown members pass through (`extra="allow"`).
- Envelopes the package builds and reads itself (the `WWW-Authenticate`
  challenge, the persisted discovery state, `McpOAuthState`) use snake_case
  keys; only the server documents keep the wire shape.
- `client_information` that comes back as an empty object is treated as
  absent (Python's falsy, where JavaScript's truthiness would pass it
  through), so the flow registers or exchanges a code instead of sending
  `client_id` with no value.
- OAuth URL handling is `urllib.parse` throughout: a malformed
  `token_endpoint` that the WHATWG URL constructor would repair
  (`"https:host/path"`) is passed to the fetch as-is and rejected there.

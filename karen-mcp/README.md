# karen-mcp

A small, standalone Model Context Protocol client — a Python port of
[`@earendil-works/pi`](https://github.com/earendil-works/pi)'s `packages/mcp`
(pi commit `f29ea3d`).

Like pi's package it depends on nothing but the standard library (plus pydantic
for the typed protocol objects), and it does not wrap the official MCP SDK. The
package provides a transport-neutral client core, stdio and Streamable HTTP
transports, and an in-memory testing transport.

> Status: the client core, the stdio transport, and the in-memory transport are
> in place. The Streamable HTTP transport and OAuth are the next two slices, so
> `StreamableHttpTransport` and `karen_mcp.oauth` do not exist yet.

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

For a remote server, use `StreamableHttpTransport(url=..., headers=...)`.

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

"""MCP tool integration for the karen coding agent.

Connects to Model Context Protocol servers (via `karen_mcp`), wraps each of
their tools as an `AgentTool` the agent loop can call, and projects the
results into LLM content with `karen_mcp.to_llm_content`.

This is the application-layer half of the MCP client: it owns the connection
lifecycle and the tool-bridging; the protocol, transports, and OAuth live in
`karen_mcp`. A server that fails to connect or list its tools is skipped with
a stderr diagnostic rather than aborting the whole session, mirroring how pi
coding-agent tolerates an individual MCP server being down while the rest of
the tool set stays usable.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from karen_ai import AbortSignal, ImageContent, TextContent
from karen_agent.types import AgentTool, AgentToolResult

from karen_mcp import McpClient, McpClientOptions
from karen_mcp.protocol.content import to_llm_content
from karen_mcp.transports.stdio import StdioTransport, StdioTransportOptions
from karen_mcp.transports.streamable_http import (
    StreamableHttpTransport,
    StreamableHttpTransportOptions,
)


def _sanitized_name(server: str, tool_name: str) -> str:
    """`mcp_<server>_<tool>` flattened to a provider-safe [A-Za-z0-9_-] run,
    capped at 64 characters (pi's coding-agent tool-name rule)."""
    raw = f"mcp_{server}_{tool_name}"
    name = re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:64]
    return name or "mcp_tool"


@dataclass
class McpServerConfig:
    """One MCP server. Exactly one of `command` (stdio) or `url` (HTTP).

    `auth_provider` is an `AuthProvider` (from `karen_mcp.oauth`
    `adapt_oauth_provider`) for a Streamable HTTP server that needs a token.
    `transport` is a pre-built transport for testing; when set the server
    never starts a real stdio/HTTP connection.
    """

    name: str
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    auth_provider: Optional[Any] = None
    transport: Optional[Any] = None


def _make_transport(config: McpServerConfig) -> Any:
    if config.transport is not None:
        return config.transport
    if config.command:
        return StdioTransport(
            StdioTransportOptions(command=config.command, args=config.args or [])
        )
    if config.url:
        options = StreamableHttpTransportOptions(url=config.url, headers=config.headers or {})
        if config.auth_provider is not None:
            options.auth_provider = config.auth_provider
        return StreamableHttpTransport(options)
    raise ValueError(f"MCP server {config.name!r} has no command, url, or transport")


def _wrap_tool(client: McpClient, config: McpServerConfig, tool: Any) -> AgentTool:
    """One `AgentTool` that proxies to `client.call_tool` and projects the
    result through `to_llm_content` (text and images survive; everything else
    becomes a short placeholder, exactly as karen_mcp documents)."""
    bound_name = _sanitized_name(config.name, tool.name)

    async def execute(
        tool_call_id: str,
        args: Any,
        signal: Optional[AbortSignal],
        on_update: Any,
    ) -> AgentToolResult:
        result = await client.call_tool(tool.name, args)
        content: List[Any] = []
        for block in to_llm_content(result):
            kind = block.get("type")
            if kind == "text":
                content.append(TextContent(text=block.get("text", "")))
            elif kind == "image":
                content.append(ImageContent(data=block.get("data", ""), mime_type=block.get("mimeType", "")))
            else:
                content.append(TextContent(text=f"[unsupported MCP content {kind}]"))
        return AgentToolResult(content=content, details=None)

    return AgentTool(
        name=bound_name,
        label=f"{config.name}:{tool.name}",
        description=tool.description or tool.name,
        parameters=tool.input_schema or {"type": "object"},
        execute=execute,
    )


@dataclass
class McpToolManager:
    """Connect a set of MCP servers and expose their tools as `AgentTool`s.

    Call `connect()` before `tools()`, and `aclose()` when the session ends.
    A server that errors during connect is dropped from the set (its tools
    simply do not appear); a server that connects but whose `tools/list`
    fails is closed and dropped too, each with a stderr diagnostic.
    """

    configs: List[McpServerConfig] = field(default_factory=list)
    _clients: Dict[str, McpClient] = field(default_factory=dict, init=False)
    _tools: List[AgentTool] = field(default_factory=list, init=False)

    async def connect(self) -> None:
        """Connect every configured server and build the wrapped tool list."""
        for config in self.configs:
            transport = _make_transport(config)
            client = McpClient(
                McpClientOptions(
                    name=f"karen-mcp-{config.name}",
                    version="1.0.0",
                    roots=[{"uri": "file:///", "name": "workspace"}],
                )
            )
            try:
                await client.connect(transport)
            except BaseException as error:
                print(
                    f"[mcp: failed to connect to {config.name!r}: {error}]",
                    file=sys.stderr,
                )
                continue
            self._clients[config.name] = client
            try:
                tools = await client.list_tools()
            except BaseException as error:
                print(f"[mcp: tools/list failed for {config.name!r}: {error}]", file=sys.stderr)
                await client.close()
                self._clients.pop(config.name, None)
                continue
            for tool in tools:
                self._tools.append(_wrap_tool(client, config, tool))

    def tools(self) -> List[AgentTool]:
        """The wrapped tools, ready to merge into the agent's tool set."""
        return list(self._tools)

    async def aclose(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except BaseException:
                pass
        self._clients.clear()


__all__ = ["McpServerConfig", "McpToolManager", "_wrap_tool"]

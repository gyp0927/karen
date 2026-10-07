"""A standalone Model Context Protocol client.

Ported from `@earendil-works/pi`'s `packages/mcp` (pi commit f29ea3d): a
transport-neutral client core, stdio and Streamable HTTP transports, an
in-memory testing transport, and the MCP OAuth client subset.
"""

from .auth_provider import AuthProvider, McpFetch, UnauthorizedContext
from .client import DEFAULT_REQUEST_TIMEOUT_MS, MAX_LIST_PAGES, McpClient, McpClientOptions, RequestContext
from .protocol.content import CallToolResult, ContentBlock, LlmContent, to_llm_content
from .protocol.jsonrpc import (
    JSON_RPC_ERROR_CODES,
    JsonRpcId,
    McpAbortError,
    McpConnectionClosedError,
    McpError,
    McpTimeoutError,
    is_json_rpc_notification,
    is_json_rpc_request,
    is_json_rpc_response,
    parse_json_rpc_message,
)
from .protocol.types import (
    LATEST_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    CancelledNotification,
    ClientCapabilities,
    Implementation,
    InitializeResult,
    ListResourceTemplatesResult,
    ListResourcesResult,
    ListToolsResult,
    ProgressNotification,
    ReadResourceResult,
    Resource,
    ResourceContents,
    ResourceTemplate,
    Root,
    ServerCapabilities,
    Tool,
    ToolAnnotations,
    ToolExecution,
)
from .transports.in_memory import InMemoryTransport, create_in_memory_transport_pair
from .transports.stdio import StdioTransport, StdioTransportOptions
from .transports.transport import (
    DEFAULT_MAX_MESSAGE_BYTES,
    CloseListener,
    ErrorListener,
    McpTransport,
    MessageListener,
)

__all__ = [
    "DEFAULT_MAX_MESSAGE_BYTES",
    "DEFAULT_REQUEST_TIMEOUT_MS",
    "JSON_RPC_ERROR_CODES",
    "LATEST_PROTOCOL_VERSION",
    "MAX_LIST_PAGES",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "AuthProvider",
    "CallToolResult",
    "CancelledNotification",
    "ClientCapabilities",
    "CloseListener",
    "ContentBlock",
    "ErrorListener",
    "Implementation",
    "InMemoryTransport",
    "InitializeResult",
    "JsonRpcId",
    "ListResourceTemplatesResult",
    "ListResourcesResult",
    "ListToolsResult",
    "LlmContent",
    "McpAbortError",
    "McpClient",
    "McpClientOptions",
    "McpConnectionClosedError",
    "McpError",
    "McpFetch",
    "McpTimeoutError",
    "McpTransport",
    "MessageListener",
    "ProgressNotification",
    "ReadResourceResult",
    "RequestContext",
    "Resource",
    "ResourceContents",
    "ResourceTemplate",
    "Root",
    "ServerCapabilities",
    "StdioTransport",
    "StdioTransportOptions",
    "Tool",
    "ToolAnnotations",
    "ToolExecution",
    "UnauthorizedContext",
    "create_in_memory_transport_pair",
    "is_json_rpc_notification",
    "is_json_rpc_request",
    "is_json_rpc_response",
    "parse_json_rpc_message",
    "to_llm_content",
]

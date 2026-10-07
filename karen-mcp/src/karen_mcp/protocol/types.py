"""MCP protocol types (pi's `protocol/types.ts`).

Wire fields are camelCase, Python attributes snake_case — the same convention
karen-ai uses for pi-ai's models. Unknown members survive (`extra="allow"`),
because `_meta` and future fields have to reach the caller untouched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "LATEST_PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "CancelledNotification",
    "ClientCapabilities",
    "Implementation",
    "InitializeResult",
    "ListResourceTemplatesResult",
    "ListResourcesResult",
    "ListToolsResult",
    "ProgressNotification",
    "ReadResourceResult",
    "Resource",
    "ResourceContents",
    "ResourceTemplate",
    "Root",
    "ServerCapabilities",
    "Tool",
    "ToolAnnotations",
    "ToolExecution",
]

LATEST_PROTOCOL_VERSION = "2025-11-25"

#: Versions the client accepts from a server. Servers that do not support the
#: requested version answer with their own latest one, so older versions stay
#: accepted for servers built on older SDKs.
SUPPORTED_PROTOCOL_VERSIONS = (LATEST_PROTOCOL_VERSION, "2025-06-18", "2025-03-26", "2024-11-05")

_CONFIG = ConfigDict(populate_by_name=True, extra="allow")


class Implementation(BaseModel):
    model_config = _CONFIG

    name: str
    version: str
    title: Optional[str] = None


class Root(BaseModel):
    model_config = _CONFIG

    uri: str
    name: Optional[str] = None


class ClientCapabilities(BaseModel):
    model_config = _CONFIG

    experimental: Optional[Dict[str, Any]] = None
    roots: Optional[Dict[str, Any]] = None
    sampling: Optional[Dict[str, Any]] = None
    elicitation: Optional[Dict[str, Any]] = None


class ServerCapabilities(BaseModel):
    model_config = _CONFIG

    experimental: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    tools: Optional[Dict[str, Any]] = None
    completions: Optional[Dict[str, Any]] = None


class InitializeResult(BaseModel):
    model_config = _CONFIG

    protocol_version: str = Field(alias="protocolVersion")
    capabilities: ServerCapabilities
    server_info: Implementation = Field(alias="serverInfo")
    instructions: Optional[str] = None


class ProgressNotification(BaseModel):
    model_config = _CONFIG

    progress_token: Any = Field(alias="progressToken")
    progress: float
    total: Optional[float] = None
    message: Optional[str] = None


class CancelledNotification(BaseModel):
    model_config = _CONFIG

    request_id: Any = Field(alias="requestId")
    reason: Optional[str] = None


class ToolAnnotations(BaseModel):
    model_config = _CONFIG

    title: Optional[str] = None
    read_only_hint: Optional[bool] = Field(default=None, alias="readOnlyHint")
    destructive_hint: Optional[bool] = Field(default=None, alias="destructiveHint")
    idempotent_hint: Optional[bool] = Field(default=None, alias="idempotentHint")
    open_world_hint: Optional[bool] = Field(default=None, alias="openWorldHint")


class ToolExecution(BaseModel):
    model_config = _CONFIG

    task_support: Optional[str] = Field(default=None, alias="taskSupport")


class Tool(BaseModel):
    model_config = _CONFIG

    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(alias="inputSchema")
    output_schema: Optional[Dict[str, Any]] = Field(default=None, alias="outputSchema")
    annotations: Optional[ToolAnnotations] = None
    execution: Optional[ToolExecution] = None


class ListToolsResult(BaseModel):
    model_config = _CONFIG

    tools: List[Tool]
    next_cursor: Optional[str] = Field(default=None, alias="nextCursor")


class Resource(BaseModel):
    """A resource a server lists in `resources/list`."""

    model_config = _CONFIG

    uri: str
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    mime_type: Optional[str] = Field(default=None, alias="mimeType")
    size: Optional[int] = None


class ResourceTemplate(BaseModel):
    """A family of resources, addressed by an RFC 6570 URI template."""

    model_config = _CONFIG

    uri_template: str = Field(alias="uriTemplate")
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    mime_type: Optional[str] = Field(default=None, alias="mimeType")


class ListResourcesResult(BaseModel):
    model_config = _CONFIG

    resources: List[Resource]
    next_cursor: Optional[str] = Field(default=None, alias="nextCursor")


class ListResourceTemplatesResult(BaseModel):
    model_config = _CONFIG

    resource_templates: List[ResourceTemplate] = Field(alias="resourceTemplates")
    next_cursor: Optional[str] = Field(default=None, alias="nextCursor")


class ResourceContents(BaseModel):
    """One entry of `resources/read`. Either `text` or `blob` is set."""

    model_config = _CONFIG

    uri: str
    mime_type: Optional[str] = Field(default=None, alias="mimeType")
    text: Optional[str] = None
    blob: Optional[str] = None


class ReadResourceResult(BaseModel):
    model_config = _CONFIG

    contents: List[ResourceContents]

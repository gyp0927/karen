"""Tool-result content blocks and their LLM projection (pi's `protocol/content.ts`).

Blocks stay dicts: servers send block types this client does not know, and pi's
conversion is a switch over `type` that turns unknown ones into a placeholder
rather than a validation error — a block that cannot be modelled cannot be
passed through either. The result envelope around them is typed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Union

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["CallToolResult", "ContentBlock", "LlmContent", "to_llm_content"]

#: One entry of a `tools/call` result: text, image, audio, resource_link or an
#: embedded resource. Only `type` is guaranteed.
ContentBlock = Dict[str, Any]

#: Tool result content in the shape LLM APIs accept: text and base64 images,
#: matching karen-ai's `TextContent`/`ImageContent` wire fields.
LlmContent = Dict[str, Any]


class CallToolResult(BaseModel):
    """pi's `CallToolResult`. `content` is required by the spec, but servers
    that only return `structuredContent` omit it (the SDK defaults it too)."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    content: List[ContentBlock] = Field(default_factory=list)
    structured_content: Union[Dict[str, Any], None] = Field(default=None, alias="structuredContent")
    is_error: Union[bool, None] = Field(default=None, alias="isError")


def _block_to_llm_content(block: Mapping[str, Any]) -> LlmContent:
    kind = block.get("type")
    if kind == "text":
        return {"type": "text", "text": block.get("text", "")}
    if kind == "image":
        return {"type": "image", "data": block.get("data", ""), "mimeType": block.get("mimeType", "")}
    if kind == "audio":
        return {"type": "text", "text": f"[audio {block.get('mimeType')} omitted]"}
    if kind == "resource_link":
        return {"type": "text", "text": f"{block.get('name')}: {block.get('uri')}"}
    if kind == "resource":
        resource = block.get("resource") or {}
        if "text" in resource:
            return {"type": "text", "text": resource.get("text", "")}
        mime_type = resource.get("mimeType")
        if isinstance(mime_type, str) and mime_type.startswith("image/"):
            return {"type": "image", "data": resource.get("blob", ""), "mimeType": mime_type}
        return {
            "type": "text",
            "text": f"[binary resource {resource.get('uri')} ({mime_type or 'unknown type'}) omitted]",
        }
    return {"type": "text", "text": f"[unsupported MCP content {kind}]"}


def to_llm_content(result: Union[CallToolResult, Mapping[str, Any]]) -> List[LlmContent]:
    """Convert a tool result to text and image content for a model.

    Text and images pass through, embedded text resources become text, embedded
    image resources become images, and other blocks (audio, resource links,
    binary resources) become a short text placeholder. A result without content
    blocks but with `structuredContent` becomes its JSON, since servers should,
    but do not always, mirror structured results as text.
    """
    data: Mapping[str, Any] = result.model_dump(by_alias=True) if isinstance(result, CallToolResult) else result
    content = [_block_to_llm_content(block) for block in (data.get("content") or [])]
    structured = data.get("structuredContent")
    if not content and structured is not None:
        content.append({"type": "text", "text": json.dumps(structured, indent=2, ensure_ascii=False)})
    return content

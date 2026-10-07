"""Normalize image blocks returned by tool results (pi's `utils/tool-result-images.ts`).

Tools that produce images themselves (extensions, MCP bridges, screenshot
tools) hand back arbitrary base64 payloads that go straight into session
history and every subsequent provider request. Oversized images make the
provider reject the whole conversation, not just the offending turn, so
normalize them once as they enter history.

The `read` tool and `@file` attachments run their images through
`process_image` already; this pass covers images produced by tools.
"""

from __future__ import annotations

import base64
from typing import List, Optional, Sequence, Union

from karen_ai import ImageContent, TextContent

from .image_process import ImageResizeOptions, process_image

ToolResultContent = Union[TextContent, ImageContent]


async def normalize_tool_result_images(
    content: Sequence[ToolResultContent],
    auto_resize_images: bool = True,
    resize_options: Optional[ImageResizeOptions] = None,
) -> List[ToolResultContent]:
    """Normalize image blocks in a tool result.

    Returns the original list (same objects) when nothing changed so callers
    can skip rewriting the result.
    """
    if not any(getattr(block, "type", None) == "image" for block in content):
        return list(content)

    normalized: List[ToolResultContent] = []
    changed = False

    for block in content:
        if getattr(block, "type", None) != "image":
            normalized.append(block)
            continue

        processed = await process_image(
            base64.b64decode(block.data),
            block.mime_type,
            auto_resize_images=auto_resize_images,
            resize_options=resize_options,
        )
        if not processed.ok:
            # Unlike `read`, keep the original block. The tool already produced
            # this image and the failure may just be an unavailable image
            # backend, so passing it through preserves behavior instead of
            # silently deleting the tool's output.
            normalized.append(block)
            continue

        if processed.data == block.data and processed.mime_type == block.mime_type and not processed.hints:
            normalized.append(block)
            continue

        normalized.append(ImageContent(data=processed.data, mime_type=processed.mime_type))
        if processed.hints:
            normalized.append(TextContent(text="\n".join(processed.hints)))
        changed = True

    return normalized if changed else list(content)

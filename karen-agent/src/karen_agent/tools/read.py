"""The `read` tool (pi's `harness/tools/read.ts`).

karen adaptation: the factory takes `cwd` directly instead of pi's
ExecutionToolContext{env}; file reads use pathlib I/O with UTF-8 decoding
(replacement on error, like pi's TextDecoder).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from karen_ai import ImageContent, TextContent
from karen_ai.types import KarenBase
from pydantic import Field

from ..types import AgentTool, AgentToolResult
from ..utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head
from .image import detect_supported_image_mime_type, encode_base64
from .local_shell import format_js_number
from .path_utils import resolve_read_tool_path

READ_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
        "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
        "limit": {"type": "number", "description": "Maximum number of lines to read"},
    },
    "required": ["path"],
}

READ_DESCRIPTION = (
    "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). "
    "Images are sent as attachments. For text files, output is truncated to "
    f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
    "Use offset/limit for large files. When you need the full file, continue with offset until complete."
)


class ProcessedImage(KarenBase):
    """Successful image processor outcome (pi's ReadImageProcessorResult ok variant)."""

    data: str  # base64
    mime_type: str
    hints: List[str] = Field(default_factory=list)


class ImageProcessingFailed(KarenBase):
    """Failed image processor outcome (pi's ReadImageProcessorResult error variant)."""

    message: str


ReadImageProcessorResult = Union[ProcessedImage, ImageProcessingFailed]
#: Optional image conversion/resizing hook, e.g. to downscale large screenshots.
#: Receives (bytes, mime_type, auto_resize) and returns one of the result models.
ReadImageProcessor = Callable[[bytes, str, bool], Awaitable[ReadImageProcessorResult]]


def create_read_tool(
    cwd: Optional[str] = None,
    *,
    image_processor: Optional[ReadImageProcessor] = None,
    auto_resize_images: bool = True,
) -> AgentTool:
    """Create the `read` tool. `cwd` defaults to the process cwd at call time."""

    async def execute(tool_call_id: str, params: Dict[str, Any], signal, on_update) -> AgentToolResult:
        path = params["path"]
        offset = params.get("offset")
        limit = params.get("limit")

        absolute_path = resolve_read_tool_path(cwd or os.getcwd(), path)
        data = Path(absolute_path).read_bytes()
        mime_type = detect_supported_image_mime_type(data)

        if mime_type:
            if image_processor is not None:
                processed = await image_processor(data, mime_type, auto_resize_images)
                if isinstance(processed, ImageProcessingFailed):
                    return AgentToolResult(
                        content=[TextContent(text=f"Read image file [{mime_type}]\n{processed.message}")]
                    )
                hints = f"\n{chr(10).join(processed.hints)}" if processed.hints else ""
                return AgentToolResult(
                    content=[
                        TextContent(text=f"Read image file [{processed.mime_type}]{hints}"),
                        ImageContent(data=processed.data, mime_type=processed.mime_type),
                    ]
                )
            if mime_type == "image/bmp":
                return AgentToolResult(
                    content=[
                        TextContent(
                            text="Read image file [image/bmp]\n"
                            "[Image omitted: configure an imageProcessor to convert BMP images.]"
                        )
                    ]
                )
            return AgentToolResult(
                content=[
                    TextContent(text=f"Read image file [{mime_type}]"),
                    ImageContent(data=encode_base64(data), mime_type=mime_type),
                ]
            )

        text_content = data.decode("utf-8", errors="replace")
        all_lines = text_content.split("\n")
        total_file_lines = len(all_lines)
        start_line = max(0, int(offset) - 1) if offset else 0
        start_line_display = start_line + 1
        if start_line >= len(all_lines):
            raise ValueError(f"Offset {format_js_number(offset)} is beyond end of file ({len(all_lines)} lines total)")

        selected_content: str
        user_limited_lines: Optional[int] = None
        if limit is not None:
            end_line = min(start_line + int(limit), len(all_lines))
            selected_content = "\n".join(all_lines[start_line:end_line])
            user_limited_lines = end_line - start_line
        else:
            selected_content = "\n".join(all_lines[start_line:])

        truncation = truncate_head(selected_content)
        details: Optional[Dict[str, Any]] = None
        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(len(all_lines[start_line].encode("utf-8")))
            output_text = (
                f"[Line {start_line_display} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} "
                f"limit. Use bash: sed -n '{start_line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
            )
            details = {"truncation": truncation.model_dump(mode="json", by_alias=True)}
        elif truncation.truncated:
            end_line_display = start_line_display + truncation.output_lines - 1
            next_offset = end_line_display + 1
            output_text = truncation.content
            if truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                )
            details = {"truncation": truncation.model_dump(mode="json", by_alias=True)}
        elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
            remaining = len(all_lines) - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output_text = f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
        else:
            output_text = truncation.content

        return AgentToolResult(content=[TextContent(text=output_text)], details=details)

    return AgentTool(
        name="read",
        label="read",
        description=READ_DESCRIPTION,
        parameters=READ_SCHEMA,
        execute=execute,
    )


__all__ = [
    "ImageProcessingFailed",
    "ProcessedImage",
    "READ_DESCRIPTION",
    "READ_SCHEMA",
    "ReadImageProcessor",
    "ReadImageProcessorResult",
    "create_read_tool",
]

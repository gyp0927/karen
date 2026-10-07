"""Application-level tools for the karen coding agent (pi coding-agent's
`core/tools/`): find, grep, ls (in-process ports of pi's fd/rg wrappers) and
powershell (Windows only), layered on karen-agent's read/write/edit/bash.

`create_default_tools(cwd)` mirrors pi's default tool set and order:
read, bash, powershell (Windows), edit, write, grep, find, ls.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from karen_agent.tools import (
    create_bash_tool,
    create_edit_tool,
    create_read_tool,
    create_write_tool,
)
from karen_agent.tools.read import ImageProcessingFailed, ProcessedImage
from karen_agent.types import AgentTool

from ..utils.image_process import process_image
from .find import FIND_DESCRIPTION, FIND_SCHEMA, create_find_tool
from .grep import GREP_DESCRIPTION, GREP_SCHEMA, create_grep_tool
from .ls import LS_DESCRIPTION, LS_SCHEMA, create_ls_tool


def _make_read_image_processor(resize_options=None):
    """Build the read tool's image processor (pi's `read.ts` -> `processImage`).

    `resize_options` is the resize profile to use — or a zero-argument callable
    returning it, resolved per read. pi resolves the profile per call from the
    execution context's model (`ctx?.model?.inputLimits?.images?.resize`), so a
    callable keeps a mid-session model switch effective.
    """

    async def _processor(data: bytes, mime_type: str, auto_resize: bool):
        options = resize_options() if callable(resize_options) else resize_options
        processed = await process_image(
            data, mime_type, auto_resize_images=auto_resize, resize_options=options
        )
        if not processed.ok:
            return ImageProcessingFailed(message=processed.message)
        return ProcessedImage(data=processed.data, mime_type=processed.mime_type, hints=processed.hints)

    return _processor


def create_default_tools(cwd: Optional[str] = None, *, shell_path: Optional[str] = None,
                         shell_command_prefix: Optional[str] = None,
                         auto_resize_images: bool = True,
                         image_resize_options=None) -> List[AgentTool]:
    """karen's default tool set, in pi's order (powershell only on Windows).

    `shell_path`/`shell_command_prefix` customize the bash tool (pi's
    `shellPath`/`shellCommandPrefix` settings). `auto_resize_images` and
    `image_resize_options` (the current model's resize profile) wire the read
    tool's image processor (pi's `images.autoResize` + `inputLimits.images.resize`).
    """
    tools: List[AgentTool] = [
        create_read_tool(
            cwd,
            image_processor=_make_read_image_processor(image_resize_options),
            auto_resize_images=auto_resize_images,
        ),
        create_bash_tool(cwd, command_prefix=shell_command_prefix, shell_path=shell_path),
    ]
    if sys.platform == "win32":
        from .powershell import create_powershell_tool

        tools.append(create_powershell_tool(cwd))
    tools.extend(
        [
            create_edit_tool(cwd),
            create_write_tool(cwd),
            create_grep_tool(cwd),
            create_find_tool(cwd),
            create_ls_tool(cwd),
        ]
    )
    return tools


__all__ = [
    "FIND_DESCRIPTION",
    "FIND_SCHEMA",
    "GREP_DESCRIPTION",
    "GREP_SCHEMA",
    "LS_DESCRIPTION",
    "LS_SCHEMA",
    "create_default_tools",
    "create_find_tool",
    "create_grep_tool",
    "create_ls_tool",
]

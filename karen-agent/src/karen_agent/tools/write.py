"""The `write` tool (pi's `harness/tools/write.ts`)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from karen_ai import TextContent

from ..types import AgentTool, AgentToolResult
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_tool_path

WRITE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
        "content": {"type": "string", "description": "Content to write to the file"},
    },
    "required": ["path", "content"],
}

WRITE_DESCRIPTION = (
    "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
    "Automatically creates parent directories."
)


def create_write_tool(cwd: Optional[str] = None) -> AgentTool:
    """Create the `write` tool. `cwd` defaults to the process cwd at call time."""

    async def execute(tool_call_id: str, params: Dict[str, Any], signal, on_update) -> AgentToolResult:
        path = params["path"]
        content = params["content"]
        absolute_path = resolve_tool_path(cwd or os.getcwd(), path)

        async def write() -> AgentToolResult:
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")
            target = Path(absolute_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Raw UTF-8 bytes, no newline translation (pi's env.writeFile).
            target.write_bytes(content.encode("utf-8"))
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")
            return AgentToolResult(content=[TextContent(text=f"Successfully wrote to {path}")])

        return await with_file_mutation_queue(absolute_path, write)

    return AgentTool(
        name="write",
        label="write",
        description=WRITE_DESCRIPTION,
        parameters=WRITE_SCHEMA,
        execute=execute,
    )


__all__ = ["WRITE_DESCRIPTION", "WRITE_SCHEMA", "create_write_tool"]

"""The `ls` tool (pi coding-agent's `core/tools/ls.ts`; pure JS there too)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from karen_ai import TextContent
from karen_agent.types import AgentTool, AgentToolResult
from karen_agent.utils.truncate import DEFAULT_MAX_BYTES, format_size, truncate_head

DEFAULT_LIMIT = 500
#: pi passes Number.MAX_SAFE_INTEGER as the line cap (byte limit only).
NO_LINE_LIMIT = 9007199254740991

LS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Directory to list (default: current directory)"},
        "limit": {"type": "number", "description": "Maximum number of entries to return (default: 500)"},
    },
    "required": [],
}

LS_DESCRIPTION = (
    "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. "
    f"Includes dotfiles. Output is truncated to {DEFAULT_LIMIT} entries or "
    f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
)


def create_ls_tool(cwd: Optional[str] = None) -> AgentTool:
    """Create the `ls` tool. `cwd` defaults to the process cwd at call time."""

    async def execute(tool_call_id: str, params: Dict[str, Any], signal, on_update) -> AgentToolResult:
        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")
        path = params.get("path")
        limit = params.get("limit")
        effective_limit = int(limit) if limit is not None else DEFAULT_LIMIT

        base = Path(cwd or os.getcwd())
        dir_path = (base / path).resolve() if path else base
        if not dir_path.exists():
            raise RuntimeError(f"Path not found: {dir_path}")
        if not dir_path.is_dir():
            raise RuntimeError(f"Not a directory: {dir_path}")
        try:
            entries = sorted(os.listdir(dir_path), key=str.lower)
        except OSError as error:
            raise RuntimeError(f"Cannot read directory: {error}") from error

        results = []
        entry_limit_reached = False
        for entry in entries:
            if len(results) >= effective_limit:
                entry_limit_reached = True
                break
            try:
                is_dir = (dir_path / entry).is_dir()
            except OSError:
                continue  # skip entries we cannot stat
            results.append(entry + ("/" if is_dir else ""))

        if not results:
            return AgentToolResult(content=[TextContent(text="(empty directory)")], details=None)

        truncation = truncate_head("\n".join(results), max_lines=NO_LINE_LIMIT)
        output = truncation.content
        details: Dict[str, Any] = {}
        notices = []
        if entry_limit_reached:
            notices.append(f"{effective_limit} entries limit reached. Use limit={effective_limit * 2} for more")
            details["entryLimitReached"] = effective_limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details["truncation"] = truncation.model_dump(mode="json", by_alias=True)
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"
        return AgentToolResult(
            content=[TextContent(text=output)],
            details=details or None,
        )

    return AgentTool(name="ls", label="ls", description=LS_DESCRIPTION, parameters=LS_SCHEMA, execute=execute)


__all__ = ["DEFAULT_LIMIT", "LS_DESCRIPTION", "LS_SCHEMA", "create_ls_tool"]

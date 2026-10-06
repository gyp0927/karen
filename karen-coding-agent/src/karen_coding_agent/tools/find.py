"""The `find` tool (pi coding-agent's `core/tools/find.ts`).

pi shells out to the `fd` binary (downloaded on demand); karen walks the tree
in-process via `walk.py` + `globs.py` with fd's matching semantics. Unlike pi,
results are sorted case-insensitively before the limit is applied — fd's
output order is unspecified, so this pins down what pi leaves open.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from karen_ai import TextContent
from karen_agent.types import AgentTool, AgentToolResult
from karen_agent.utils.truncate import DEFAULT_MAX_BYTES, format_size, truncate_head

from .globs import compile_find_matcher
from .walk import walk

DEFAULT_LIMIT = 1000
#: pi passes Number.MAX_SAFE_INTEGER as the line cap (byte limit only).
NO_LINE_LIMIT = 9007199254740991

FIND_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'",
        },
        "path": {"type": "string", "description": "Directory to search in (default: current directory)"},
        "limit": {"type": "number", "description": "Maximum number of results (default: 1000)"},
    },
    "required": ["pattern"],
}

FIND_DESCRIPTION = (
    "Search for files by glob pattern. Returns matching file paths relative to the search directory. "
    f"Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} results or "
    f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
)


def create_find_tool(cwd: Optional[str] = None) -> AgentTool:
    """Create the `find` tool. `cwd` defaults to the process cwd at call time."""

    async def execute(tool_call_id: str, params: Dict[str, Any], signal, on_update) -> AgentToolResult:
        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")
        pattern = params["pattern"]
        search_dir = params.get("path")
        limit = params.get("limit")
        effective_limit = int(limit) if limit is not None else DEFAULT_LIMIT

        base = Path(cwd or os.getcwd())
        search_path = (base / search_dir).resolve() if search_dir else base
        if not search_path.exists():
            raise RuntimeError(f"Path not found: {search_path}")
        if not search_path.is_dir():
            raise RuntimeError(f"Not a directory: {search_path}")

        matcher = compile_find_matcher(pattern)
        matches = []
        for entry in walk(search_path, include_dirs=True):
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")
            if matcher(entry.rel_path, entry.is_dir):
                matches.append(entry.rel_path + ("/" if entry.is_dir else ""))
        matches.sort(key=str.lower)

        if not matches:
            return AgentToolResult(content=[TextContent(text="No files found matching pattern")], details=None)

        result_limit_reached = len(matches) >= effective_limit
        matches = matches[:effective_limit]
        truncation = truncate_head("\n".join(matches), max_lines=NO_LINE_LIMIT)
        output = truncation.content
        details: Dict[str, Any] = {}
        notices = []
        if result_limit_reached:
            notices.append(
                f"{effective_limit} results limit reached. "
                f"Use limit={effective_limit * 2} for more, or refine pattern"
            )
            details["resultLimitReached"] = effective_limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details["truncation"] = truncation.model_dump(mode="json", by_alias=True)
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"
        return AgentToolResult(
            content=[TextContent(text=output)],
            details=details or None,
        )

    return AgentTool(name="find", label="find", description=FIND_DESCRIPTION, parameters=FIND_SCHEMA, execute=execute)


__all__ = ["DEFAULT_LIMIT", "FIND_DESCRIPTION", "FIND_SCHEMA", "create_find_tool"]

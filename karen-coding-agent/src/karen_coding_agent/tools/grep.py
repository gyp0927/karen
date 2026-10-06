"""The `grep` tool (pi coding-agent's `core/tools/grep.ts`).

pi shells out to the `rg` binary (downloaded on demand) and parses its `--json`
event stream; karen searches in-process via `walk.py` + Python `re`. Semantics
preserved: gitignore-aware walk, one match per matching line, match limit
counts matching lines, binary files are searched only up to the first NUL
byte (rg's binary detection), lone/CRLF `\r` normalized away, context blocks
may overlap (pi does not dedupe), and the phantom line after a trailing
newline can appear as context but never matches (rg reports no match for it).
Rust-regex-only syntax differences surface as Python `re` errors.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from karen_ai import TextContent
from karen_agent.types import AgentTool, AgentToolResult
from karen_agent.utils.truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    format_size,
    truncate_head,
    truncate_line,
)

from .globs import compile_grep_glob_matcher
from .walk import walk

DEFAULT_LIMIT = 100
#: pi passes Number.MAX_SAFE_INTEGER as the line cap (byte limit only).
NO_LINE_LIMIT = 9007199254740991

GREP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
        "path": {"type": "string", "description": "Directory or file to search (default: current directory)"},
        "glob": {"type": "string", "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'"},
        "ignoreCase": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
        "literal": {"type": "boolean", "description": "Treat pattern as literal string instead of regex (default: false)"},
        "context": {"type": "number", "description": "Number of lines to show before and after each match (default: 0)"},
        "limit": {"type": "number", "description": "Maximum number of matches to return (default: 100)"},
    },
    "required": ["pattern"],
}

GREP_DESCRIPTION = (
    "Search file contents for a pattern. Returns matching lines with file paths and line numbers. "
    f"Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} matches or "
    f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
    f"Long lines are truncated to {GREP_MAX_LINE_LENGTH} chars."
)


def _read_search_lines(path: Path) -> Optional[List[str]]:
    """Decode a candidate file to display lines; None when unreadable.

    Splits like pi's getFileLines (`\r\n`/`\r` normalized to `\n`, trailing
    phantom line kept for context rendering); matching skips the phantom.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    # rg binary detection: the file is searched only up to the first NUL byte.
    data = data.split(b"\x00", 1)[0]
    text = data.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def create_grep_tool(cwd: Optional[str] = None) -> AgentTool:
    """Create the `grep` tool. `cwd` defaults to the process cwd at call time."""

    async def execute(tool_call_id: str, params: Dict[str, Any], signal, on_update) -> AgentToolResult:
        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")
        pattern = params["pattern"]
        glob_filter = params.get("glob")
        context = params.get("context") or 0
        context_value = int(context) if context > 0 else 0
        limit = params.get("limit")
        effective_limit = max(1, int(limit)) if limit is not None else DEFAULT_LIMIT

        source = re.escape(pattern) if params.get("literal") else pattern
        flags = re.IGNORECASE if params.get("ignoreCase") else 0
        try:
            regex = re.compile(source, flags)
        except re.error as error:
            raise RuntimeError(f"Invalid regular expression: {error}") from error

        base = Path(cwd or os.getcwd())
        search_dir = params.get("path")
        search_path = (base / search_dir).resolve() if search_dir else base
        if not search_path.exists():
            raise RuntimeError(f"Path not found: {search_path}")
        is_directory = search_path.is_dir()
        file_matcher = compile_grep_glob_matcher(glob_filter) if glob_filter else None

        candidates: List[Tuple[Path, str]] = []
        if is_directory:
            for entry in walk(search_path):
                if file_matcher is None or file_matcher(entry.rel_path):
                    candidates.append((entry.path, entry.rel_path))
        elif file_matcher is None or file_matcher(search_path.name):
            candidates.append((search_path, search_path.name))

        # (display path, file lines, hit line numbers) — file lines are kept
        # for context rendering, like pi's fileCache.
        matched_files: List[Tuple[str, List[str], List[int]]] = []
        match_count = 0
        match_limit_reached = False
        for path, rel_path in candidates:
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")
            lines = _read_search_lines(path)
            if not lines:
                continue
            # The phantom element after a trailing newline is never a match.
            searchable = len(lines) - 1 if lines[-1] == "" else len(lines)
            hits: List[int] = []
            for index in range(searchable):
                if regex.search(lines[index]):
                    hits.append(index + 1)
                    match_count += 1
                    if match_count >= effective_limit:
                        match_limit_reached = True
                        break
            if hits:
                matched_files.append((rel_path, lines, hits))
            if match_limit_reached:
                break

        if match_count == 0:
            return AgentToolResult(content=[TextContent(text="No matches found")], details=None)

        output_lines: List[str] = []
        lines_truncated = False
        for rel_path, lines, hits in matched_files:
            for line_number in hits:
                start = max(1, line_number - context_value)
                end = min(len(lines), line_number + context_value)
                for current in range(start, end + 1):
                    text, was_truncated = truncate_line(lines[current - 1])
                    if was_truncated:
                        lines_truncated = True
                    if current == line_number:
                        output_lines.append(f"{rel_path}:{current}: {text}")
                    else:
                        output_lines.append(f"{rel_path}-{current}- {text}")

        truncation = truncate_head("\n".join(output_lines), max_lines=NO_LINE_LIMIT)
        output = truncation.content
        details: Dict[str, Any] = {}
        notices = []
        if match_limit_reached:
            notices.append(
                f"{effective_limit} matches limit reached. "
                f"Use limit={effective_limit * 2} for more, or refine pattern"
            )
            details["matchLimitReached"] = effective_limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details["truncation"] = truncation.model_dump(mode="json", by_alias=True)
        if lines_truncated:
            notices.append(
                f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines"
            )
            details["linesTruncated"] = True
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"
        return AgentToolResult(
            content=[TextContent(text=output)],
            details=details or None,
        )

    return AgentTool(name="grep", label="grep", description=GREP_DESCRIPTION, parameters=GREP_SCHEMA, execute=execute)


__all__ = ["DEFAULT_LIMIT", "GREP_DESCRIPTION", "GREP_SCHEMA", "create_grep_tool"]

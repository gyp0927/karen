"""Shared truncation utilities for tool outputs (pi's `harness/utils/truncate.ts`).

Truncation is based on two independent limits — whichever is hit first wins:
- Line limit (default: 2000 lines)
- Byte limit (default: 50KB, counted as UTF-8 bytes)

Never returns partial lines (except the bash tail-truncation edge case).
"""

from __future__ import annotations

from typing import List, Literal, Optional

from karen_ai.types import KarenBase

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB
GREP_MAX_LINE_LENGTH = 500  # Max chars per grep match line


class TruncationResult(KarenBase):
    """Outcome of a head/tail truncation; serialized into tool `details`."""

    #: The truncated content.
    content: str
    #: Whether truncation occurred.
    truncated: bool
    #: Which limit was hit: "lines", "bytes", or None if not truncated.
    truncated_by: Optional[Literal["lines", "bytes"]] = None
    #: Total number of lines in the original content.
    total_lines: int
    #: Total number of UTF-8 bytes in the original content.
    total_bytes: int
    #: Number of complete lines in the truncated output.
    output_lines: int
    #: Number of UTF-8 bytes in the truncated output.
    output_bytes: int
    #: Whether the last line was partially truncated (tail truncation edge case).
    last_line_partial: bool
    #: Whether the first line exceeded the byte limit (head truncation).
    first_line_exceeds_limit: bool
    #: The max lines limit that was applied.
    max_lines: int
    #: The max bytes limit that was applied.
    max_bytes: int


def utf8_byte_length(content: str) -> int:
    """Byte length of `content` encoded as UTF-8 (pi's Buffer.byteLength)."""
    return len(content.encode("utf-8", errors="replace"))


def _split_lines_for_counting(content: str) -> List[str]:
    if len(content) == 0:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def format_size(num_bytes: int) -> str:
    """Format bytes as human-readable size (pi's formatSize)."""
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def truncate_head(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult:
    """Truncate content from the head (keep first N lines/bytes).

    Suitable for file reads where you want to see the beginning. Never returns
    partial lines; if the first line alone exceeds the byte limit, returns empty
    content with `first_line_exceeds_limit=True`.
    """
    total_bytes = utf8_byte_length(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    first_line_bytes = utf8_byte_length(lines[0])
    if first_line_bytes > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            last_line_partial=False,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines_arr: List[str] = []
    output_bytes_count = 0
    truncated_by: Literal["lines", "bytes"] = "lines"

    for i, line in enumerate(lines):
        if i >= max_lines:
            break
        line_bytes = utf8_byte_length(line) + (1 if i > 0 else 0)  # +1 for newline
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines_arr.append(line)
        output_bytes_count += line_bytes

    if len(output_lines_arr) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines_arr)
    final_output_bytes = utf8_byte_length(output_content)

    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines_arr),
        output_bytes=final_output_bytes,
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_tail(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult:
    """Truncate content from the tail (keep last N lines/bytes).

    Suitable for bash output where you want to see the end (errors, final
    results). May return a partial first line if the last line of the original
    content exceeds the byte limit.
    """
    total_bytes = utf8_byte_length(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines_arr: List[str] = []
    output_bytes_count = 0
    truncated_by: Literal["lines", "bytes"] = "lines"
    last_line_partial = False

    for i in range(len(lines) - 1, -1, -1):
        if len(output_lines_arr) >= max_lines:
            break
        line = lines[i]
        line_bytes = utf8_byte_length(line) + (1 if output_lines_arr else 0)  # +1 for newline
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            # Edge case: if we haven't added ANY lines yet and this line exceeds
            # maxBytes, take the end of the line (partial).
            if not output_lines_arr:
                truncated_line = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines_arr.insert(0, truncated_line)
                output_bytes_count = utf8_byte_length(truncated_line)
                last_line_partial = True
            break
        output_lines_arr.insert(0, line)
        output_bytes_count += line_bytes

    if len(output_lines_arr) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines_arr)
    final_output_bytes = utf8_byte_length(output_content)

    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines_arr),
        output_bytes=final_output_bytes,
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    """Truncate a string to fit within a byte limit (from the end), UTF-8 safe."""
    if max_bytes <= 0:
        return ""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    start = len(raw) - max_bytes
    while start < len(raw) and (raw[start] & 0xC0) == 0x80:  # skip mid-codepoint
        start += 1
    return raw[start:].decode("utf-8", errors="replace")


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    """Truncate a single line to max characters, adding a [truncated] suffix.

    Returns (text, was_truncated). Used for grep match lines.
    """
    if len(line) <= max_chars:
        return line, False
    return f"{line[:max_chars]}... [truncated]", True

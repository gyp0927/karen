"""Bounded shell-output capture (pi's `harness/utils/output-capture.ts`).

Maintains and publishes one bounded output view. Writes received while
publication is rate-limited collapse into the latest view; the first update
after idle and an explicit final flush are immediate.

Deviation from pi: pi's ExecutionEnv boundary publishes replace/append/slide/
metadata *deltas* (`ShellOutputUpdate`) that callers reassemble with
`applyShellOutputUpdate`. karen's capture lives in-process, so it publishes
complete `ShellOutputView` snapshots directly — no delta protocol, no
`apply_shell_output_update` port.
"""

from __future__ import annotations

import codecs
import re
from typing import Callable, Literal, Optional

from karen_ai.types import KarenBase

from .adaptive_publisher import AdaptivePublisher
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    truncate_head,
    truncate_tail,
    utf8_byte_length,
)

OUTPUT_MIN_EMIT_INTERVAL_MS = 100
OUTPUT_TARGET_BYTES_PER_SECOND = 100 * 1024

INVALID_SHELL_OUTPUT = re.compile(r"[\x00-\x08\x0b-\x1f￹-￻]")


def sanitize_shell_output(text: str) -> str:
    """Strip control chars that would corrupt terminal rendering."""
    return INVALID_SHELL_OUTPUT.sub("", text)


#: Alias kept for parity with pi's `sanitizeBinaryOutput` re-export.
sanitize_binary_output = sanitize_shell_output


class ShellOutputTruncation(KarenBase):
    """Truncation metadata without a duplicate copy of the retained text
    (pi's `ShellOutputTruncation` = TruncationResult minus content)."""

    truncated: bool
    truncated_by: Optional[Literal["lines", "bytes"]] = None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int


class ShellOutputView(KarenBase):
    """Complete bounded shell output view (pi's `ShellOutputView`)."""

    text: str
    truncation: ShellOutputTruncation
    spill_path: Optional[str] = None
    last_line_bytes: Optional[int] = None


class OutputCapture:
    """One bounded, published shell-output view."""

    def __init__(
        self,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_lines: int = DEFAULT_MAX_LINES,
        retain: Literal["head", "tail"] = "tail",
        on_update: Optional[Callable[[ShellOutputView], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> None:
        if not isinstance(max_bytes, (int, float)) or max_bytes <= 0 or max_bytes == float("inf"):
            raise TypeError("Output maxBytes must be a positive finite number")
        if not isinstance(max_lines, int) or max_lines <= 0:
            raise TypeError("Output maxLines must be a positive integer")
        self._max_bytes = int(max_bytes)
        self._max_lines = max_lines
        self._retain = retain
        self._on_update = on_update

        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buffer = ""
        self._buffer_bytes = 0
        self._total_bytes = 0
        self._newlines = 0
        self._ends_with_newline = True
        self._current_line_bytes = 0
        self._spill_path: Optional[str] = None
        self._disposed = False
        self._publisher: AdaptivePublisher[ShellOutputView, ShellOutputView] = AdaptivePublisher(
            snapshot=self.snapshot,
            update=lambda _previous, current: current,
            measure=lambda view: utf8_byte_length(view.text),
            publish=self._publish,
            on_error=on_error,
            min_interval_ms=OUTPUT_MIN_EMIT_INTERVAL_MS,
            target_bytes_per_second=OUTPUT_TARGET_BYTES_PER_SECOND,
        )

    def _publish(self, view: ShellOutputView) -> None:
        if self._on_update is not None:
            self._on_update(view)

    @property
    def truncated(self) -> bool:
        return self._total_bytes > self._max_bytes or self._total_lines() > self._max_lines

    def push(self, chunk: "str | bytes") -> None:
        if self._disposed:
            return
        if isinstance(chunk, str):
            # pi flushes the byte decoder here (TextDecoder.decode() with no input).
            self._append_text(self._decoder.decode(b"", final=True))
            self._append_text(chunk)
            return
        self._append_text(self._decoder.decode(bytes(chunk)))

    def finish(self) -> None:
        if self._disposed:
            return
        self._append_text(self._decoder.decode(b"", final=True))

    def set_spill_path(self, path: str) -> None:
        if self._disposed or self._spill_path == path:
            return
        self._spill_path = path
        self._publisher.mark_dirty()
        self.flush()

    def snapshot(self) -> ShellOutputView:
        if self._retain == "head":
            retained = truncate_head(self._buffer, max_bytes=self._max_bytes, max_lines=self._max_lines)
        else:
            retained = truncate_tail(self._buffer, max_bytes=self._max_bytes, max_lines=self._max_lines)
        total_lines = self._total_lines()
        truncated = self.truncated
        truncation = ShellOutputTruncation(
            truncated=truncated,
            truncated_by=("lines" if total_lines > self._max_lines else "bytes") if truncated else None,
            total_lines=total_lines,
            total_bytes=self._total_bytes,
            output_lines=retained.output_lines,
            output_bytes=retained.output_bytes,
            last_line_partial=retained.last_line_partial,
            first_line_exceeds_limit=retained.first_line_exceeds_limit,
            max_lines=retained.max_lines,
            max_bytes=retained.max_bytes,
        )
        return ShellOutputView(
            text=sanitize_shell_output(retained.content),
            truncation=truncation,
            spill_path=self._spill_path,
            last_line_bytes=self._current_line_bytes if retained.last_line_partial else None,
        )

    def flush(self) -> None:
        if self._disposed:
            return
        self._publisher.flush(force=True)

    def dispose(self) -> None:
        self._publisher.dispose()
        self._disposed = True

    def _append_text(self, text: str) -> None:
        if text == "":
            return
        text_bytes = utf8_byte_length(text)
        self._total_bytes += text_bytes
        self._newlines += text.count("\n")
        self._ends_with_newline = text.endswith("\n")
        last_newline = text.rfind("\n")
        self._current_line_bytes = (
            self._current_line_bytes + text_bytes
            if last_newline == -1
            else utf8_byte_length(text[last_newline + 1 :])
        )
        self._buffer += text
        self._buffer_bytes += text_bytes

        guard = self._max_bytes * 2
        if self._buffer_bytes > guard * 2:
            self._buffer = (
                _trim_to_last_utf8_bytes(self._buffer, guard)
                if self._retain == "tail"
                else _trim_to_first_utf8_bytes(self._buffer, guard)
            )
            self._buffer_bytes = utf8_byte_length(self._buffer)
        self._publisher.mark_dirty()

    def _total_lines(self) -> int:
        return self._newlines + (0 if self._ends_with_newline or self._total_bytes == 0 else 1)


def _trim_to_last_utf8_bytes(text: str, max_bytes: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    start = len(raw) - max_bytes
    while start < len(raw) and (raw[start] & 0xC0) == 0x80:
        start += 1
    return raw[start:].decode("utf-8", errors="replace")


def _trim_to_first_utf8_bytes(text: str, max_bytes: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    end = max_bytes
    while end > 0 and end < len(raw) and (raw[end] & 0xC0) == 0x80:
        end -= 1
    return raw[:end].decode("utf-8", errors="replace")


__all__ = [
    "OUTPUT_MIN_EMIT_INTERVAL_MS",
    "OUTPUT_TARGET_BYTES_PER_SECOND",
    "OutputCapture",
    "ShellOutputTruncation",
    "ShellOutputView",
    "sanitize_binary_output",
    "sanitize_shell_output",
]

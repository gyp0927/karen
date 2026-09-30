"""Shared helpers for karen-agent (pi's `harness/utils/`)."""

from .output_capture import (
    OUTPUT_MIN_EMIT_INTERVAL_MS,
    OUTPUT_TARGET_BYTES_PER_SECOND,
    OutputCapture,
    ShellOutputTruncation,
    ShellOutputView,
    sanitize_binary_output,
    sanitize_shell_output,
)
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    GREP_MAX_LINE_LENGTH,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_line,
    truncate_tail,
    utf8_byte_length,
)
from .usage import add_usage, empty_usage

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "GREP_MAX_LINE_LENGTH",
    "OUTPUT_MIN_EMIT_INTERVAL_MS",
    "OUTPUT_TARGET_BYTES_PER_SECOND",
    "OutputCapture",
    "ShellOutputTruncation",
    "ShellOutputView",
    "TruncationResult",
    "add_usage",
    "empty_usage",
    "format_size",
    "sanitize_binary_output",
    "sanitize_shell_output",
    "truncate_head",
    "truncate_line",
    "truncate_tail",
    "utf8_byte_length",
]

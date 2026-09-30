"""Execution environment (pi's `harness/env/`): capability types + local impl."""

from .local import LocalExecutionEnv
from .types import (
    ExecutionEnv,
    ExecutionError,
    ExecutionErrorCode,
    FileError,
    FileErrorCode,
    FileInfo,
    FileKind,
    FileSystem,
    Shell,
    ShellExecOptions,
    ShellExecResult,
    ShellOutputCaptureOptions,
    ShellOutputLimits,
    ShellOutputMetadata,
    ShellOutputRetention,
    TextLine,
    TextLineReader,
)

__all__ = [
    "ExecutionEnv",
    "ExecutionError",
    "ExecutionErrorCode",
    "FileError",
    "FileErrorCode",
    "FileInfo",
    "FileKind",
    "FileSystem",
    "LocalExecutionEnv",
    "Shell",
    "ShellExecOptions",
    "ShellExecResult",
    "ShellOutputCaptureOptions",
    "ShellOutputLimits",
    "ShellOutputMetadata",
    "ShellOutputRetention",
    "TextLine",
    "TextLineReader",
]

"""Execution-environment types (the FileSystem/Shell half of pi's `harness/types.ts`).

pi models the harness's filesystem + process access as a capability interface
(`ExecutionEnv extends FileSystem, Shell`) whose methods never throw — failures
are `Result` values with stable, backend-independent error codes. karen kept
direct pathlib/subprocess I/O in M1–M3; this package ports the capability
layer itself so apps (and pi-faithful harness code) can use it.

Deviation kept from M2: `ShellExecOptions.on_update` receives complete
`ShellOutputView` snapshots, not pi's replace/append/slide `ShellOutputUpdate`
deltas — karen's OutputCapture lives in-process (see `utils/output_capture.py`).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, runtime_checkable

from karen_ai.types import KarenBase
from pydantic import ConfigDict

from ..result import Result
from ..tools.local_shell import ExecutionError, ExecutionErrorCode
from ..utils.output_capture import ShellOutputTruncation, ShellOutputView

__all__ = [
    "FileKind",
    "FileErrorCode",
    "FileError",
    "ExecutionErrorCode",
    "ExecutionError",
    "FileInfo",
    "TextLine",
    "TextLineReader",
    "ShellOutputRetention",
    "ShellOutputLimits",
    "ShellOutputCaptureOptions",
    "ShellOutputMetadata",
    "ShellOutputTruncation",
    "ShellOutputView",
    "ShellExecResult",
    "ShellExecOptions",
    "FileSystem",
    "Shell",
    "ExecutionEnv",
]

#: Kind of filesystem object. Symlinks are not followed automatically.
FileKind = Literal["file", "directory", "symlink"]

#: Stable, backend-independent file error codes.
FileErrorCode = Literal[
    "aborted",
    "not_found",
    "permission_denied",
    "not_directory",
    "is_directory",
    "invalid",
    "not_supported",
    "unknown",
]


class FileError(Exception):
    """Error returned by FileSystem operations (pi's `FileError`)."""

    def __init__(
        self,
        code: FileErrorCode,
        message: str,
        path: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.code: FileErrorCode = code
        self.path = path
        if cause is not None:
            self.__cause__ = cause


class FileInfo(KarenBase):
    """Metadata for one filesystem object."""

    #: Basename of `path`.
    name: str
    #: Absolute, syntactically normalized addressed path. Symlinks are not followed.
    path: str
    #: Object kind. Symlink targets are not followed; use `canonical_path` explicitly.
    kind: FileKind
    #: Size in bytes.
    size: int
    #: Modification time as milliseconds since Unix epoch.
    mtime_ms: float


class TextLine(KarenBase):
    """One UTF-8 line read from a text file."""

    text: str
    #: Whether the line ended with `\n`; callers use this to discard a torn final record.
    terminated: bool


class TextLineReader(Protocol):
    """Pull-based UTF-8 line reader that preserves final-line termination."""

    async def read_line(self) -> Result[Optional[TextLine], FileError]: ...

    async def close(self) -> None:
        """Release the open file. Must be best-effort and must not raise."""
        ...


#: Which portion of bounded output survives after the limit is crossed.
ShellOutputRetention = Literal["head", "tail"]


class ShellOutputLimits(KarenBase):
    """Source-side limits for one combined shell output view."""

    max_bytes: int
    max_lines: int
    retain: ShellOutputRetention = "tail"


class ShellOutputCaptureOptions(KarenBase):
    """Bounded shell capture requested by the caller."""

    limits: ShellOutputLimits
    #: Preserve complete output in a local file after the limits are crossed.
    spill: Optional[bool] = None


class ShellOutputMetadata(KarenBase):
    """Metadata accompanying a bounded shell output view."""

    truncation: ShellOutputTruncation
    spill_path: Optional[str] = None
    last_line_bytes: Optional[int] = None


class ShellExecResult(KarenBase):
    """Bounded shell completion. Output text is delivered through `ShellExecOptions.on_update`."""

    exit_code: int
    truncation: ShellOutputTruncation
    spill_path: Optional[str] = None
    last_line_bytes: Optional[int] = None


class ShellExecOptions(KarenBase):
    """Options for `Shell.exec`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: Working directory for the command; relative paths resolve against the env's cwd.
    cwd: Optional[str] = None
    #: Environment variables; values override inherited defaults when `inherit_env` is true.
    env: Optional[Dict[str, str]] = None
    #: Whether to inherit the default variables. Defaults to true.
    inherit_env: Optional[bool] = None
    #: Timeout in seconds. Defaults to no timeout.
    timeout: Optional[float] = None
    #: Source-side bounded capture. Output is discarded when this and `on_update` are both absent.
    capture: Optional[ShellOutputCaptureOptions] = None
    #: Called with bounded output changes. karen deviation: receives complete
    #: ShellOutputView snapshots, not pi's delta updates.
    on_update: Optional[Callable[[ShellOutputView], None]] = None


@runtime_checkable
class FileSystem(Protocol):
    """Filesystem capability used by the harness (pi's `FileSystem`).

    Operation methods must never raise. All failures, including unexpected
    backend failures, are encoded in the returned `Result`.
    """

    cwd: str

    async def absolute_path(self, path: str) -> Result[str, FileError]: ...
    async def join_path(self, parts: List[str]) -> Result[str, FileError]: ...
    async def read_text_file(self, path: str) -> Result[str, FileError]: ...
    async def open_text_line_reader(self, path: str) -> Result[TextLineReader, FileError]: ...
    async def read_text_lines(
        self, path: str, options: Optional[Dict[str, int]] = None
    ) -> Result[List[str], FileError]: ...
    async def read_binary_file(self, path: str) -> Result[bytes, FileError]: ...
    async def write_file(self, path: str, content: Any) -> Result[None, FileError]: ...
    async def append_file(self, path: str, content: Any) -> Result[None, FileError]: ...
    async def rename_file(self, source_path: str, destination_path: str) -> Result[None, FileError]: ...
    async def file_info(self, path: str) -> Result[FileInfo, FileError]: ...
    async def list_dir(self, path: str) -> Result[List[FileInfo], FileError]: ...
    async def canonical_path(self, path: str) -> Result[str, FileError]: ...
    async def exists(self, path: str) -> Result[bool, FileError]: ...
    async def create_dir(
        self, path: str, options: Optional[Dict[str, bool]] = None
    ) -> Result[None, FileError]: ...
    async def remove(self, path: str, options: Optional[Dict[str, bool]] = None) -> Result[None, FileError]: ...
    async def create_temp_dir(self, prefix: Optional[str] = None) -> Result[str, FileError]: ...
    async def create_temp_file(self, options: Optional[Dict[str, str]] = None) -> Result[str, FileError]: ...
    async def cleanup(self) -> None: ...


@runtime_checkable
class Shell(Protocol):
    """Shell execution capability used by the harness (pi's `Shell`)."""

    async def exec(
        self, command: str, options: Optional[ShellExecOptions] = None
    ) -> Result[ShellExecResult, ExecutionError]: ...

    async def cleanup(self) -> None: ...


@runtime_checkable
class ExecutionEnv(FileSystem, Shell, Protocol):
    """Filesystem and process execution environment used by the harness (pi's `ExecutionEnv`)."""

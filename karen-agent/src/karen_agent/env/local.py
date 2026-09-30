"""Local execution environment (pi's `harness/env/nodejs.ts` `NodeExecutionEnv`).

A Python-backed `ExecutionEnv`: filesystem operations via `os`/`pathlib` and
shell execution via `local_shell.run_shell_command` (which carries pi's bash
resolution, process-tree kills, timeout/abort handling, and spill logic).

Deviations from pi, consistent with earlier milestones:
- no chord `Context` parameter — an explicit `signal: Optional[AbortSignal]`
  keyword threads cancellation into every method;
- methods are `async` per the interface but run synchronous `os` calls inline
  (consistent with karen's sync-pathlib convention);
- `os.replace` implements `rename_file` ("replacing the destination when it
  exists") cross-platform;
- OS error text differs from Node's; the stable `FileErrorCode` mapping is exact.
"""

from __future__ import annotations

import codecs
import errno
import os
import shutil
import stat
import tempfile
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

from karen_ai import AbortSignal

from ..result import Err, Ok, Result, err, ok
from ..tools.local_shell import (
    ExecutionError,
    _kill_process_tree,
    resolve_shell_config,
    run_shell_command,
)
from ..utils.output_capture import OutputCapture
from .types import (
    FileError,
    FileErrorCode,
    FileInfo,
    FileKind,
    ShellExecOptions,
    ShellExecResult,
    TextLine,
    TextLineReader,
)

__all__ = ["LocalExecutionEnv"]


def _resolve_path(cwd: str, path: str) -> str:
    """pi's resolvePath: `~` expansion, `file://` URLs, then cwd-relative resolve."""
    normalized = path
    if normalized == "~":
        normalized = os.path.expanduser("~")
    elif normalized.startswith("~/") or (os.name == "nt" and normalized.startswith("~\\")):
        normalized = os.path.join(os.path.expanduser("~"), normalized[2:])
    elif normalized.startswith("file://"):
        try:
            normalized = urllib.request.url2pathname(normalized[7:])
        except Exception:  # noqa: BLE001 - keep malformed URLs as ordinary paths
            pass
    return normalized if os.path.isabs(normalized) else os.path.abspath(os.path.join(cwd, normalized))


def _aborted(signal: Optional[AbortSignal], path: Optional[str] = None) -> Optional[Err[Any, FileError]]:
    if signal is not None and signal.aborted:
        return err(FileError("aborted", "aborted", path))
    return None


def _to_file_error(error: BaseException, fallback_path: Optional[str] = None) -> FileError:
    """Map Python OSErrors onto pi's stable FileErrorCode set (Node errno mapping)."""
    if isinstance(error, FileError):
        return error
    if isinstance(error, OSError):
        path = error.filename if isinstance(error.filename, str) else fallback_path
        message = str(error)
        mapping: Dict[int, FileErrorCode] = {
            errno.ENOENT: "not_found",
            errno.EACCES: "permission_denied",
            errno.EPERM: "permission_denied",
            errno.ENOTDIR: "not_directory",
            errno.EISDIR: "is_directory",
            errno.EINVAL: "invalid",
        }
        code = mapping.get(error.errno)
        if code is not None:
            return FileError(code, message, path, error)
    return FileError("unknown", str(error), fallback_path, error)


def _file_kind_from_stat(mode: int) -> Optional[FileKind]:
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    return None


def _file_info_from_stat(path: str, st: os.stat_result) -> Result[FileInfo, FileError]:
    kind = _file_kind_from_stat(st.st_mode)
    if kind is None:
        return err(FileError("invalid", "Unsupported file type", path))
    return ok(
        FileInfo(
            name=os.path.basename(path),
            path=path,
            kind=kind,
            size=st.st_size,
            mtime_ms=st.st_mtime * 1000,
        )
    )


class _LocalTextLineReader(TextLineReader):
    """Strict LF reader with explicit byte offsets, so an aborted read can be
    retried without skipping bytes (pi's NodeTextLineReader)."""

    _CHUNK = 64 * 1024

    def __init__(self, path: str) -> None:
        self._file = open(path, "rb")
        self._path = path
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._byte_offset = 0
        self._buffered = ""
        self._ended = False
        self._closed = False

    async def read_line(self, signal: Optional[AbortSignal] = None) -> Result[Optional[TextLine], FileError]:
        aborted = _aborted(signal, self._path)
        if aborted is not None:
            return aborted
        if self._closed:
            return err(FileError("invalid", "Text line reader is closed", self._path))

        try:
            while True:
                newline = self._buffered.find("\n")
                if newline != -1:
                    text = self._buffered[:newline]
                    self._buffered = self._buffered[newline + 1 :]
                    return ok(TextLine(text=text, terminated=True))
                if self._ended:
                    if len(self._buffered) == 0:
                        return ok(None)
                    text = self._buffered
                    self._buffered = ""
                    return ok(TextLine(text=text, terminated=False))

                self._file.seek(self._byte_offset)
                chunk = self._file.read(self._CHUNK)
                aborted = _aborted(signal, self._path)
                if aborted is not None:
                    return aborted
                self._byte_offset += len(chunk)
                if len(chunk) == 0:
                    self._buffered += self._decoder.decode(b"", final=True)
                    self._ended = True
                else:
                    self._buffered += self._decoder.decode(chunk)
        except OSError as error:
            return err(_to_file_error(error, self._path))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._buffered = ""
        try:
            self._file.close()
        except Exception:  # noqa: BLE001 - closing is best-effort
            pass


class LocalExecutionEnv:
    """Python-backed ExecutionEnv (pi's NodeExecutionEnv)."""

    def __init__(
        self,
        cwd: str,
        shell_path: Optional[str] = None,
        shell_env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.cwd = cwd
        self._shell_path = shell_path
        self._shell_env = shell_env
        self._active_procs: Dict[int, Any] = {}

    # -- path primitives ------------------------------------------------------

    async def absolute_path(self, path: str, signal: Optional[AbortSignal] = None) -> Result[str, FileError]:
        return ok(_resolve_path(self.cwd, path))

    async def join_path(self, parts: List[str], signal: Optional[AbortSignal] = None) -> Result[str, FileError]:
        return ok(os.path.normpath(os.path.join(*parts)))

    # -- shell ----------------------------------------------------------------

    async def exec(
        self,
        command: str,
        options: Optional[ShellExecOptions] = None,
        signal: Optional[AbortSignal] = None,
    ) -> Result[ShellExecResult, ExecutionError]:
        if signal is not None and signal.aborted:
            return err(ExecutionError("aborted", "aborted"))

        cwd = _resolve_path(self.cwd, options.cwd) if options is not None and options.cwd else self.cwd
        inherit_env = options.inherit_env if options is not None and options.inherit_env is not None else True
        merged_env = (
            {**os.environ, **(self._shell_env or {}), **((options.env if options else None) or {})}
            if inherit_env
            else dict(((options.env if options else None) or {}))
        )
        capture_options = options.capture if options is not None else None
        limits = capture_options.limits if capture_options is not None else None
        capture_kwargs: Dict[str, Any] = {"on_update": options.on_update if options is not None else None}
        if limits is not None:
            capture_kwargs.update(max_bytes=limits.max_bytes, max_lines=limits.max_lines, retain=limits.retain)
        try:
            capture = OutputCapture(**capture_kwargs)
        except Exception as error:  # noqa: BLE001 - pi encodes constructor failures as "unknown"
            return err(ExecutionError("unknown", str(error), error))

        # resolve_shell_config runs before spawn so a missing shell is shell_unavailable.
        try:
            resolve_shell_config(self._shell_path)
        except ExecutionError as error:
            return err(error)

        active_proc: Dict[str, Any] = {}

        def on_spawn(proc: Any) -> None:
            active_proc["proc"] = proc
            if proc.pid is not None:
                self._active_procs[proc.pid] = proc

        try:
            exit_code = await run_shell_command(
                command,
                cwd=cwd,
                env=merged_env,
                inherit_env=False,
                timeout=options.timeout if options is not None else None,
                shell_path=self._shell_path,
                capture=capture,
                spill=bool(capture_options.spill) if capture_options is not None else False,
                signal=signal,
                on_spawn=on_spawn,
            )
        except ExecutionError as error:
            return err(error)
        finally:
            proc = active_proc.get("proc")
            if proc is not None and proc.pid is not None:
                self._active_procs.pop(proc.pid, None)
            capture.dispose()

        output = capture.snapshot()
        return ok(
            ShellExecResult(
                exit_code=exit_code,
                truncation=output.truncation,
                spill_path=output.spill_path,
                last_line_bytes=output.last_line_bytes,
            )
        )

    # -- file reads -------------------------------------------------------------

    async def open_text_line_reader(
        self, path: str, signal: Optional[AbortSignal] = None
    ) -> Result[TextLineReader, FileError]:
        resolved = _resolve_path(self.cwd, path)
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        try:
            reader = _LocalTextLineReader(resolved)
        except OSError as error:
            return err(_to_file_error(error, resolved))
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            await reader.close()
            return aborted
        return ok(reader)

    async def read_text_file(self, path: str, signal: Optional[AbortSignal] = None) -> Result[str, FileError]:
        resolved = _resolve_path(self.cwd, path)
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        try:
            with open(resolved, "r", encoding="utf-8", newline="") as file:
                return ok(file.read())
        except OSError as error:
            return err(_to_file_error(error, resolved))

    async def read_text_lines(
        self,
        path: str,
        options: Optional[Dict[str, int]] = None,
        signal: Optional[AbortSignal] = None,
    ) -> Result[List[str], FileError]:
        max_lines = options.get("max_lines") if options else None
        if max_lines is not None and max_lines <= 0:
            return ok([])
        opened = await self.open_text_line_reader(path, signal)
        if isinstance(opened, Err):
            return opened
        lines: List[str] = []
        try:
            while max_lines is None or len(lines) < max_lines:
                line = await opened.value.read_line(signal)
                if isinstance(line, Err):
                    return line
                if line.value is None:
                    break
                lines.append(line.value.text)
            return ok(lines)
        finally:
            await opened.value.close()

    async def read_binary_file(self, path: str, signal: Optional[AbortSignal] = None) -> Result[bytes, FileError]:
        resolved = _resolve_path(self.cwd, path)
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        try:
            with open(resolved, "rb") as file:
                return ok(file.read())
        except OSError as error:
            return err(_to_file_error(error, resolved))

    # -- file writes ------------------------------------------------------------

    async def write_file(
        self, path: str, content: Any, signal: Optional[AbortSignal] = None
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        try:
            os.makedirs(os.path.dirname(os.path.abspath(resolved)), exist_ok=True)
            aborted = _aborted(signal, resolved)
            if aborted is not None:
                return aborted
            if isinstance(content, str):
                with open(resolved, "w", encoding="utf-8", newline="") as file:
                    file.write(content)
            else:
                with open(resolved, "wb") as file:
                    file.write(bytes(content))
            return ok(None)
        except OSError as error:
            return err(_to_file_error(error, resolved))

    async def append_file(
        self, path: str, content: Any, signal: Optional[AbortSignal] = None
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        try:
            os.makedirs(os.path.dirname(os.path.abspath(resolved)), exist_ok=True)
            aborted = _aborted(signal, resolved)
            if aborted is not None:
                return aborted
            if isinstance(content, str):
                with open(resolved, "a", encoding="utf-8", newline="") as file:
                    file.write(content)
            else:
                with open(resolved, "ab") as file:
                    file.write(bytes(content))
            aborted = _aborted(signal, resolved)
            if aborted is not None:
                return aborted
            return ok(None)
        except OSError as error:
            return err(_to_file_error(error, resolved))

    async def rename_file(
        self, source_path: str, destination_path: str, signal: Optional[AbortSignal] = None
    ) -> Result[None, FileError]:
        source = _resolve_path(self.cwd, source_path)
        destination = _resolve_path(self.cwd, destination_path)
        aborted = _aborted(signal, destination)
        if aborted is not None:
            return aborted
        try:
            os.replace(source, destination)
            return ok(None)
        except OSError as error:
            return err(_to_file_error(error, source))

    # -- metadata -----------------------------------------------------------------

    async def file_info(self, path: str, signal: Optional[AbortSignal] = None) -> Result[FileInfo, FileError]:
        resolved = _resolve_path(self.cwd, path)
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        try:
            return _file_info_from_stat(resolved, os.lstat(resolved))
        except OSError as error:
            return err(_to_file_error(error, resolved))

    async def list_dir(self, path: str, signal: Optional[AbortSignal] = None) -> Result[List[FileInfo], FileError]:
        resolved = _resolve_path(self.cwd, path)
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        try:
            infos: List[FileInfo] = []
            with os.scandir(resolved) as entries:
                for entry in entries:
                    aborted = _aborted(signal, resolved)
                    if aborted is not None:
                        return aborted
                    try:
                        info = _file_info_from_stat(entry.path, os.lstat(entry.path))
                        if isinstance(info, Ok):
                            infos.append(info.value)
                    except OSError as error:
                        return err(_to_file_error(error, entry.path))
            return ok(infos)
        except OSError as error:
            return err(_to_file_error(error, resolved))

    async def canonical_path(self, path: str, signal: Optional[AbortSignal] = None) -> Result[str, FileError]:
        resolved = _resolve_path(self.cwd, path)
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        try:
            return ok(os.path.realpath(resolved, strict=True))
        except OSError as error:
            return err(_to_file_error(error, resolved))

    async def exists(self, path: str, signal: Optional[AbortSignal] = None) -> Result[bool, FileError]:
        result = await self.file_info(path, signal)
        if isinstance(result, Ok):
            return ok(True)
        if result.error.code == "not_found":
            return ok(False)
        return err(result.error)

    # -- mutations ------------------------------------------------------------------

    async def create_dir(
        self,
        path: str,
        options: Optional[Dict[str, bool]] = None,
        signal: Optional[AbortSignal] = None,
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        recursive = options.get("recursive", True) if options else True
        try:
            if recursive:
                os.makedirs(resolved, exist_ok=True)
            else:
                os.mkdir(resolved)
            return ok(None)
        except OSError as error:
            return err(_to_file_error(error, resolved))

    async def remove(
        self,
        path: str,
        options: Optional[Dict[str, bool]] = None,
        signal: Optional[AbortSignal] = None,
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)
        aborted = _aborted(signal, resolved)
        if aborted is not None:
            return aborted
        recursive = options.get("recursive", False) if options else False
        force = options.get("force", False) if options else False
        try:
            if recursive and os.path.isdir(resolved) and not os.path.islink(resolved):
                shutil.rmtree(resolved, ignore_errors=force)
            else:
                try:
                    os.remove(resolved)
                except FileNotFoundError:
                    if not force:
                        raise
            return ok(None)
        except OSError as error:
            return err(_to_file_error(error, resolved))

    async def create_temp_dir(
        self, prefix: Optional[str] = None, signal: Optional[AbortSignal] = None
    ) -> Result[str, FileError]:
        aborted = _aborted(signal)
        if aborted is not None:
            return aborted
        try:
            return ok(tempfile.mkdtemp(prefix=prefix or "tmp-"))
        except OSError as error:
            return err(_to_file_error(error))

    async def create_temp_file(
        self,
        options: Optional[Dict[str, str]] = None,
        signal: Optional[AbortSignal] = None,
    ) -> Result[str, FileError]:
        directory = await self.create_temp_dir("tmp-", signal)
        if isinstance(directory, Err):
            return directory
        prefix = options.get("prefix", "") if options else ""
        suffix = options.get("suffix", "") if options else ""
        file_path = os.path.join(directory.value, f"{prefix}{uuid.uuid4()}{suffix}")
        try:
            with open(file_path, "w"):
                pass
            return ok(file_path)
        except OSError as error:
            return err(_to_file_error(error, file_path))

    async def cleanup(self) -> None:
        """Kill every process still running from `exec`; best-effort, never raises."""
        for proc in list(self._active_procs.values()):
            _kill_process_tree(proc)
        self._active_procs.clear()

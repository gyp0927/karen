"""Newline-delimited JSON-RPC over a child process's stdio (pi's `transports/stdio.ts`).

Order of business on close is the spec's: close stdin and let the server exit,
then SIGTERM, then SIGKILL — applied to the server's whole process group, so a
wrapper like `npx` or `uvx` does not leave the real server behind.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..protocol.jsonrpc import McpConnectionClosedError, parse_json_rpc_message
from .transport import DEFAULT_MAX_MESSAGE_BYTES, McpTransport, TransportEvents

__all__ = ["StdioTransport", "StdioTransportOptions"]

DEFAULT_MAX_STDERR_BYTES = 64 * 1024
DEFAULT_CLOSE_TIMEOUT_MS = 2_000
#: How long a server gets to exit on its own after stdin closes, before it is
#: sent SIGTERM.
STDIN_CLOSE_GRACE_MS = 500
USE_PROCESS_GROUPS = os.name != "nt"
#: Windows has no SIGKILL — and no signals at all, since `_kill_process_tree`
#: goes through taskkill there, so the signal is only a parameter on POSIX.
FORCE_KILL = getattr(signal, "SIGKILL", signal.SIGTERM)

#: Process groups of running servers, killed if the host exits without closing
#: them (POSIX only; Windows has no process groups to signal).
_LIVE_PROCESS_GROUPS: set = set()
_EXIT_HOOK_INSTALLED = False


def _install_exit_hook() -> None:
    global _EXIT_HOOK_INSTALLED
    if _EXIT_HOOK_INSTALLED:
        return
    _EXIT_HOOK_INSTALLED = True

    def _kill_live_groups() -> None:
        for pid in list(_LIVE_PROCESS_GROUPS):
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                pass

    atexit.register(_kill_live_groups)


def _resolve_program(command: str) -> str:
    """Resolve `command` through PATH, so a bare `npx` finds `npx.cmd`.

    pi reaches for cross-spawn here, which for a `.cmd` shim also wraps the
    spawn in `cmd.exe /d /s /c`. Python does not need that wrapper —
    `create_subprocess_exec` passes the image as the command line with a null
    application name, and Windows runs a batch file itself — and the wrapper is
    worse than not having it: `cmd.exe /s /c` strips the quotes around a path
    that contains a space, so `C:\\Program Files\\nodejs\\npx.cmd` would break,
    while the direct route handles it.
    """
    found = shutil.which(command)
    return found or command


def _kill_process_tree(process: "asyncio.subprocess.Process", sig: int) -> None:
    pid = process.pid
    if os.name == "nt" and pid is not None:
        if process.returncode is not None:
            return
        # Windows has no graceful signals, and a `.cmd` shim runs through
        # cmd.exe; killing only it would leave the server running.
        try:
            subprocess.Popen(
                ["taskkill", "/pid", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            pass
        return
    if USE_PROCESS_GROUPS and pid is not None:
        try:
            # The whole group: wrappers like `npx` or `uvx` must not leave the
            # server behind.
            os.killpg(pid, sig)
            return
        except OSError:
            # The group is gone or was never created; fall back to the child.
            pass
    try:
        process.send_signal(sig)
    except (ProcessLookupError, OSError):
        pass


@dataclass
class StdioTransportOptions:
    """pi's `StdioTransportOptions`."""

    command: str
    args: Sequence[str] = ()
    cwd: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    inherit_env: bool = True
    stderr: str = "pipe"
    on_stderr: Optional[Callable[[str], None]] = None
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES
    max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES
    #: Time to wait for the server to exit after SIGTERM before sending SIGKILL.
    close_timeout_ms: int = DEFAULT_CLOSE_TIMEOUT_MS


class StdioTransport(TransportEvents, McpTransport):
    def __init__(self, options: StdioTransportOptions) -> None:
        super().__init__()
        self.options = options
        self._process: Optional["asyncio.subprocess.Process"] = None
        self._stdout_buffer = b""
        self._stderr_buffer = ""
        self._started = False
        self._closed = False
        self._readers: List["asyncio.Task[None]"] = []

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process is not None else None

    @property
    def stderr(self) -> str:
        return self._stderr_buffer

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("MCP stdio transport already started")
        if self._closed:
            raise McpConnectionClosedError()
        self._started = True
        env = dict(os.environ) if self.options.inherit_env else {}
        env.update(self.options.env)
        program = _resolve_program(self.options.command)
        argv = [program, *self.options.args]
        self._process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.options.cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None if self.options.stderr == "inherit" else asyncio.subprocess.PIPE,
            # Own process group, so closing the transport can terminate the
            # server's children too.
            start_new_session=USE_PROCESS_GROUPS,
        )
        process = self._process
        pid = process.pid
        if USE_PROCESS_GROUPS and pid is not None:
            _install_exit_hook()
            _LIVE_PROCESS_GROUPS.add(pid)
        self._readers = [
            asyncio.ensure_future(self._read_stdout(process)),
            asyncio.ensure_future(self._read_stderr(process)),
            asyncio.ensure_future(self._watch_exit(process)),
        ]

    async def send(self, message: Dict[str, Any]) -> None:
        process = self._process
        stdin = process.stdin if process is not None else None
        if not self._started or self._closed or stdin is None or stdin.is_closing():
            raise McpConnectionClosedError()
        payload = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        stdin.write(payload)
        await stdin.drain()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None:
            self.emit_close()
            return
        if process.returncode is not None:
            self._close_stdin(process)
            await self._drain_readers()
            return
        # Shutdown per the spec: close stdin and let the server exit, then
        # SIGTERM, then SIGKILL.
        loop = asyncio.get_running_loop()
        grace = min(STDIN_CLOSE_GRACE_MS, self.options.close_timeout_ms)
        timers = [
            loop.call_later(grace / 1000.0, _kill_process_tree, process, signal.SIGTERM),
            loop.call_later(
                (grace + self.options.close_timeout_ms) / 1000.0, _kill_process_tree, process, FORCE_KILL
            ),
        ]
        self._close_stdin(process)
        try:
            await process.wait()
        finally:
            for timer in timers:
                timer.cancel()
        # Children of the server that ignored stdin closing would otherwise
        # outlive it.
        _kill_process_tree(process, signal.SIGTERM)
        await self._drain_readers()

    # -- internals -----------------------------------------------------------

    def _close_stdin(self, process: "asyncio.subprocess.Process") -> None:
        stdin = process.stdin
        if stdin is None:
            return
        try:
            stdin.close()
        except (BrokenPipeError, OSError):
            pass

    async def _drain_readers(self) -> None:
        readers, self._readers = self._readers, []
        for reader in readers:
            if reader.done():
                continue
            try:
                await asyncio.wait_for(asyncio.shield(reader), 5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                reader.cancel()

    async def _read_stdout(self, process: "asyncio.subprocess.Process") -> None:
        stream = process.stdout
        if stream is None:
            return
        try:
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    return
                self._handle_stdout(chunk)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self.emit_error(error)

    async def _read_stderr(self, process: "asyncio.subprocess.Process") -> None:
        stream = process.stderr
        if stream is None:
            return
        try:
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    return
                self._handle_stderr(chunk)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self.emit_error(error)

    async def _watch_exit(self, process: "asyncio.subprocess.Process") -> None:
        pid = process.pid
        try:
            await process.wait()
        except asyncio.CancelledError:
            raise
        finally:
            if pid is not None:
                _LIVE_PROCESS_GROUPS.discard(pid)
        if self._process is process:
            self._process = None
        if self._stdout_buffer.decode("utf-8", "replace").strip():
            self.emit_error(Exception("MCP stdio server closed with an incomplete JSON-RPC message"))
        self._stdout_buffer = b""
        self.emit_close()

    def _handle_stdout(self, chunk: bytes) -> None:
        self._stdout_buffer += chunk
        while True:
            newline = self._stdout_buffer.find(b"\n")
            if newline < 0:
                if len(self._stdout_buffer) > self.options.max_message_bytes:
                    self._stdout_buffer = b""
                    self.emit_error(
                        Exception(f"MCP stdio message exceeds {self.options.max_message_bytes} bytes")
                    )
                return
            line = self._stdout_buffer[:newline]
            self._stdout_buffer = self._stdout_buffer[newline + 1 :]
            if len(line) > self.options.max_message_bytes:
                self.emit_error(
                    Exception(f"MCP stdio message exceeds {self.options.max_message_bytes} bytes")
                )
                continue
            text = line.decode("utf-8", "replace").removesuffix("\r")
            if not text.strip():
                continue
            try:
                self.emit_message(parse_json_rpc_message(json.loads(text)))
            except BaseException as error:
                self.emit_error(error)

    def _handle_stderr(self, chunk: bytes) -> None:
        text = chunk.decode("utf-8", "replace")
        self._stderr_buffer = (self._stderr_buffer + text)[-self.options.max_stderr_bytes :]
        if self.options.on_stderr is not None:
            self.options.on_stderr(text)

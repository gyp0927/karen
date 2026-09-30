"""Local shell execution (the Shell half of pi's `harness/env/nodejs.ts`).

karen drops pi's ExecutionEnv capability abstraction: commands run on the local
machine via asyncio subprocesses. Semantics ported from pi:

- bash resolution: custom path → Git Bash under %ProgramFiles% → bash on PATH
  (Windows); /bin/bash → bash on PATH → sh (POSIX). Legacy WSL bash gets the
  command via stdin ("-s") instead of argv ("-c").
- combined stdout+stderr stream (merged at OS level via stderr=STDOUT),
- timeout and abort both kill the whole process tree,
- optional spill of the complete raw output to a temp file once capture
  truncation kicks in (bounded view keeps flowing through the capture),
- settle order: callback error → timeout → aborted → spill error → exit code
  (signal kills map to 128 + signum, mirroring pi).
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from typing import Dict, List, Literal, Optional

from karen_ai import AbortSignal

from ..utils.output_capture import OutputCapture

ExecutionErrorCode = Literal["aborted", "timeout", "shell_unavailable", "spawn_error", "callback_error", "unknown"]

#: pi's bash schema cap: 2^31 - 1 milliseconds, in seconds.
MAX_TIMEOUT_SECONDS = 2_147_483_647 / 1000


class ExecutionError(Exception):
    """Backend-independent shell execution failure (pi's ExecutionError)."""

    def __init__(self, code: ExecutionErrorCode, message: str, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.code: ExecutionErrorCode = code
        if cause is not None:
            self.__cause__ = cause


class ShellConfig:
    def __init__(self, shell: str, args: List[str], transport: Literal["argv", "stdin"] = "argv") -> None:
        self.shell = shell
        self.args = args
        self.transport = transport


def _is_legacy_wsl_bash_path(path: str) -> bool:
    normalized = path.replace("/", "\\").lower()
    return re.match(r"^[a-z]:\\windows\\(?:system32|sysnative)\\bash\.exe$", normalized) is not None


def _bash_shell_config(shell: str) -> ShellConfig:
    if _is_legacy_wsl_bash_path(shell):
        return ShellConfig(shell, ["-s"], "stdin")
    return ShellConfig(shell, ["-c"])


def resolve_shell_config(shell_path: Optional[str] = None) -> ShellConfig:
    """Resolve the shell to execute commands with (pi's getShellConfig)."""
    if shell_path:
        if os.path.exists(shell_path):
            return _bash_shell_config(shell_path)
        raise ExecutionError("shell_unavailable", f"Custom shell path not found: {shell_path}")

    if sys.platform == "win32":
        candidates: List[str] = []
        program_files = os.environ.get("ProgramFiles")
        if program_files:
            candidates.append(f"{program_files}\\Git\\bin\\bash.exe")
        program_files_x86 = os.environ.get("ProgramFiles(x86)")
        if program_files_x86:
            candidates.append(f"{program_files_x86}\\Git\\bin\\bash.exe")
        for candidate in candidates:
            if os.path.exists(candidate):
                return _bash_shell_config(candidate)
        bash_on_path = shutil.which("bash")
        if bash_on_path:
            return _bash_shell_config(bash_on_path)
        searched = "\n".join(f"  {candidate}" for candidate in candidates)
        raise ExecutionError(
            "shell_unavailable",
            "No bash shell found. Options:\n"
            "  1. Install Git for Windows: https://git-scm.com/download/win\n"
            "  2. Add your bash to PATH (Cygwin, MSYS2, etc.)\n"
            "  3. Configure an explicit shellPath\n\n"
            f"Searched Git Bash in:\n{searched}",
        )

    if os.path.exists("/bin/bash"):
        return _bash_shell_config("/bin/bash")
    bash_on_path = shutil.which("bash")
    if bash_on_path:
        return _bash_shell_config(bash_on_path)
    return ShellConfig("sh", ["-c"])


def validate_timeout(timeout: Optional[float]) -> None:
    """pi's bash schema + env validation for the timeout argument."""
    if timeout is None:
        return
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid timeout: must be a finite number of seconds")
    if timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds")


def format_js_number(value: float) -> str:
    """Format a number the way JavaScript string interpolation would."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _kill_process_tree(proc: "asyncio.subprocess.Process") -> None:
    """Kill the shell and its children; best-effort, never raises."""
    if proc.returncode is not None or proc.pid is None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            try:
                import signal as signal_module

                os.killpg(proc.pid, signal_module.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
    except Exception:  # noqa: BLE001 — best effort
        pass


async def run_shell_command(
    command: str,
    *,
    cwd: str,
    env: Optional[Dict[str, str]] = None,
    inherit_env: bool = True,
    timeout: Optional[float] = None,
    shell_path: Optional[str] = None,
    capture: OutputCapture,
    spill: bool = False,
    signal: Optional[AbortSignal] = None,
) -> int:
    """Run `command` to completion, streaming decoded output into `capture`.

    Returns the exit code. Raises ExecutionError on timeout/abort/spawn/spill
    failures. Output (including output captured before a failure) is available
    via `capture.snapshot()`.
    """
    if signal is not None and signal.aborted:
        raise ExecutionError("aborted", "aborted")
    validate_timeout(timeout)

    cwd_abs = os.path.abspath(cwd)
    if not os.path.exists(cwd_abs):
        raise ExecutionError(
            "spawn_error", f"Working directory does not exist: {cwd_abs}\nCannot execute bash commands."
        )
    config = resolve_shell_config(shell_path)
    merged_env = {**os.environ, **(env or {})} if inherit_env else dict(env or {})

    use_stdin = config.transport == "stdin"
    argv = [config.shell, *config.args] if use_stdin else [config.shell, *config.args, command]
    popen_kwargs = {} if os.name == "nt" else {"start_new_session": True}
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd_abs,
            env=merged_env,
            stdin=asyncio.subprocess.PIPE if use_stdin else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **popen_kwargs,
        )
    except OSError as error:
        raise ExecutionError("spawn_error", str(error), error)

    if use_stdin and proc.stdin is not None:
        try:
            proc.stdin.write(command.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass

    loop = asyncio.get_running_loop()
    timed_out = False

    def _on_timeout() -> None:
        nonlocal timed_out
        timed_out = True
        _kill_process_tree(proc)

    timeout_handle = loop.call_later(timeout, _on_timeout) if timeout is not None else None

    abort_task: Optional[asyncio.Task] = None
    if signal is not None:

        async def _watch_abort() -> None:
            await signal.wait()
            _kill_process_tree(proc)

        abort_task = asyncio.create_task(_watch_abort())

    spill_prefix: List[bytes] = []
    spill_path: Optional[str] = None
    spill_file = None
    spill_error: Optional[ExecutionError] = None

    def start_spill(chunk: bytes) -> None:
        nonlocal spill_path, spill_file, spill_error
        if spill_error is not None:
            return
        try:
            if spill_file is None:
                spill_dir = tempfile.mkdtemp(prefix="tmp-")
                spill_path = os.path.join(spill_dir, f"karen-output-{uuid.uuid4()}.log")
                spill_file = open(spill_path, "ab")
                capture.set_spill_path(spill_path)
            if chunk:
                spill_file.write(chunk)
        except OSError as error:
            spill_error = ExecutionError("unknown", f"Failed to preserve complete shell output: {error}", error)
            _kill_process_tree(proc)

    async def _reader() -> None:
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                break
            was_truncated = capture.truncated
            capture.push(chunk)
            if not spill:
                continue
            if spill_path is not None or was_truncated:
                start_spill(chunk)
            elif capture.truncated:
                for prefix in spill_prefix:
                    start_spill(prefix)
                spill_prefix.clear()
                start_spill(chunk)
            else:
                spill_prefix.append(chunk)

    read_task = asyncio.create_task(_reader())
    try:
        await proc.wait()  # timeout/abort resolve via the process-tree kill
        try:
            await read_task  # drain remaining buffered output to EOF
        except Exception as error:  # noqa: BLE001 — pi routes publish errors to callback_error
            _kill_process_tree(proc)
            raise ExecutionError("callback_error", str(error), error)

        if spill_file is not None:
            spill_file.close()
        try:
            capture.finish()
            capture.flush()
        except Exception as error:  # noqa: BLE001 — pi's failCallback channel
            raise ExecutionError("callback_error", str(error), error)

        if timed_out:
            raise ExecutionError("timeout", f"timeout:{format_js_number(timeout)}")
        if signal is not None and signal.aborted:
            raise ExecutionError("aborted", "aborted")
        if spill_error is not None:
            raise spill_error

        exit_code = proc.returncode
        if exit_code is None:
            exit_code = 1
        elif exit_code < 0:
            # Killed by a signal: map to the conventional 128 + signum.
            exit_code = 128 + (-exit_code)
        return exit_code
    finally:
        if timeout_handle is not None:
            timeout_handle.cancel()
        if abort_task is not None:
            abort_task.cancel()
        if spill_file is not None and not spill_file.closed:
            spill_file.close()


__all__ = [
    "ExecutionError",
    "ExecutionErrorCode",
    "MAX_TIMEOUT_SECONDS",
    "ShellConfig",
    "format_js_number",
    "resolve_shell_config",
    "run_shell_command",
    "validate_timeout",
]

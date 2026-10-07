"""Bash command execution with streaming + cancellation (pi's `core/bash-executor.ts`).

pi's executor is a unified implementation behind `AgentSession.executeBash()`
for interactive and RPC modes: it runs the command, streams sanitized output
to a callback, spills the full output to a temp file once it crosses the
truncation threshold, and reports `{output, exitCode, cancelled, truncated,
fullOutputPath}`.

karen-agent's `run_shell_command` + `OutputCapture` already implement that
exact contract (bounded tail capture, spill file, abort, exit code), so this
module is a thin adapter producing pi's `BashResult` shape.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from karen_agent.tools.local_shell import ExecutionError, run_shell_command
from karen_agent.utils.output_capture import OutputCapture, sanitize_shell_output

__all__ = ["BashResult", "execute_bash"]


class _StreamingCapture(OutputCapture):
    """`OutputCapture` whose updates are the fresh chunks (pi's `onChunk`).

    The base class's `on_update` publishes throttled *snapshots* of its bounded
    tail window, while pi hands `onChunk` each sanitized chunk as the decoder
    produces it. karen forwards that callback as the RPC
    `bash_execution_update` `delta`, so a snapshot would make an appending
    client print everything it has already printed; and once the window starts
    dropping output, a snapshot could not be turned back into a delta at all.
    """

    def __init__(self, on_chunk: Optional[Callable[[str], None]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._on_chunk = on_chunk

    def _append_text(self, text: str) -> None:
        super()._append_text(text)
        if text and self._on_chunk is not None:
            # pi sanitizes in the same place (bash-executor.ts decodes, strips
            # ANSI and control bytes, then calls onChunk).
            self._on_chunk(sanitize_shell_output(text))


class BashResult:
    """pi's `BashResult`."""

    def __init__(
        self,
        output: str,
        exit_code: Optional[int],
        cancelled: bool,
        truncated: bool,
        full_output_path: Optional[str] = None,
    ) -> None:
        self.output = output
        self.exit_code = exit_code
        self.cancelled = cancelled
        self.truncated = truncated
        self.full_output_path = full_output_path

    def to_dict(self) -> dict:
        return {
            "output": self.output,
            "exitCode": self.exit_code,
            "cancelled": self.cancelled,
            "truncated": self.truncated,
            "fullOutputPath": self.full_output_path,
        }


async def execute_bash(
    command: str,
    cwd: str,
    *,
    shell_path: Optional[str] = None,
    shell_command_prefix: Optional[str] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
    signal: Optional[Any] = None,
) -> BashResult:
    """Run `command` to completion and return pi's `BashResult`.

    `on_chunk` receives each sanitized chunk as it is decoded — pi's `onChunk`,
    not a snapshot. `signal` is a karen-ai `AbortSignal`; when it fires the
    process tree is killed and the result is reported cancelled (exit code
    omitted), matching pi's abort branch.
    """
    full_command = f"{shell_command_prefix}\n{command}" if shell_command_prefix else command

    capture = _StreamingCapture(on_chunk, retain="tail")
    try:
        exit_code = await run_shell_command(
            full_command,
            cwd=cwd,
            shell_path=shell_path,
            capture=capture,
            spill=True,
            signal=signal,
        )
        view = capture.snapshot()
        return BashResult(
            output=view.text,
            exit_code=exit_code,
            cancelled=False,
            truncated=view.truncation.truncated,
            full_output_path=view.spill_path,
        )
    except ExecutionError as error:
        if error.code == "aborted":
            view = capture.snapshot()
            return BashResult(
                output=view.text,
                exit_code=None,
                cancelled=True,
                truncated=view.truncation.truncated,
                full_output_path=view.spill_path,
            )
        raise

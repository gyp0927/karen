"""The `bash` tool (pi's `harness/tools/bash.ts`).

karen adaptation: no ExecutionEnv — commands run via `local_shell.run_shell_command`
against an explicit cwd. pi's 2s durable-checkpoint throttle is dropped (karen
has no checkpoint channel on the update callback); every published view still
flows to `on_update` as a partial AgentToolResult.
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from karen_ai import TextContent
from karen_ai.types import KarenBase
from pydantic import Field

from ..types import AgentTool, AgentToolResult
from ..utils.output_capture import OutputCapture, ShellOutputView
from ..utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size
from .local_shell import (
    MAX_TIMEOUT_SECONDS,
    ExecutionError,
    format_js_number,
    run_shell_command,
    validate_timeout,
)

BASH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
    },
    "required": ["command"],
}

BASH_DESCRIPTION = (
    "Execute a bash command in the current working directory. Returns combined stdout and stderr. "
    f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
    "(whichever is hit first). If truncated, full output is saved to a temp file. "
    "Optionally provide a timeout in seconds."
)


class BashExecution(KarenBase):
    """Mutable execution plan passed to the `prepare` hook (pi's BashExecution)."""

    command: str
    cwd: str
    env: Dict[str, str] = Field(default_factory=dict)
    inherit_env: bool = True


#: Hook invoked with the BashExecution before the command runs; may mutate it.
BashPrepare = Callable[[BashExecution], Union[None, Awaitable[None]]]


def _details_from_view(view: ShellOutputView) -> Optional[Dict[str, Any]]:
    if not view.truncation.truncated:
        return None
    details: Dict[str, Any] = {"truncation": view.truncation.model_dump(mode="json", by_alias=True)}
    if view.spill_path is not None:
        details["fullOutputPath"] = view.spill_path
    return details


def create_bash_tool(
    cwd: Optional[str] = None,
    *,
    command_prefix: Optional[str] = None,
    prepare: Optional[BashPrepare] = None,
    shell_path: Optional[str] = None,
) -> AgentTool:
    """Create the `bash` tool. `cwd` defaults to the process cwd at call time."""

    async def execute(tool_call_id: str, params: Dict[str, Any], signal, on_update) -> AgentToolResult:
        command = params["command"]
        timeout = params.get("timeout")
        validate_timeout(timeout)

        execution = BashExecution(
            command=f"{command_prefix}\n{command}" if command_prefix else command,
            cwd=cwd or os.getcwd(),
        )
        if prepare is not None:
            prepared = prepare(execution)
            if inspect.isawaitable(prepared):
                await prepared

        latest_view: Optional[ShellOutputView] = None

        def publish(view: ShellOutputView) -> None:
            nonlocal latest_view
            latest_view = view
            if on_update is None:
                return
            details: Optional[Dict[str, Any]] = None
            if view.truncation.truncated:
                details = {"truncation": view.truncation.model_dump(mode="json", by_alias=True)}
                if view.spill_path is not None:
                    details["fullOutputPath"] = view.spill_path
            on_update(AgentToolResult(content=[TextContent(text=view.text)], details=details))

        capture = OutputCapture(
            max_bytes=DEFAULT_MAX_BYTES,
            max_lines=DEFAULT_MAX_LINES,
            retain="tail",
            on_update=publish,
        )
        if on_update is not None:
            on_update(AgentToolResult(content=[], details=None))

        try:
            exit_code = await run_shell_command(
                execution.command,
                cwd=execution.cwd,
                env=execution.env,
                inherit_env=execution.inherit_env,
                timeout=timeout,
                shell_path=shell_path,
                capture=capture,
                spill=True,
                signal=signal,
            )
        except ExecutionError as error:
            output_text = latest_view.text if latest_view is not None else ""
            if error.code == "timeout":
                status = f"Command timed out after {format_js_number(timeout)} seconds"
            elif error.code == "aborted":
                status = "Command aborted"
            else:
                status = str(error)
            raise RuntimeError(f"{output_text}\n\n{status}" if output_text else status) from error

        view = capture.snapshot()
        output_text = view.text
        details = _details_from_view(view)
        if view.truncation.truncated:
            truncation = view.truncation
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                last_line_size = format_size(view.last_line_bytes or truncation.output_bytes)
                output_text += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line} "
                    f"(line is {last_line_size}). Full output: {view.spill_path}]"
                )
            elif truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. "
                    f"Full output: {view.spill_path}]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {view.spill_path}]"
                )

        if exit_code != 0:
            raise RuntimeError(
                f"{output_text}\n\nCommand exited with code {exit_code}" if output_text else f"Command exited with code {exit_code}"
            )
        return AgentToolResult(
            content=[TextContent(text=output_text or "(no output)")],
            details=details,
        )

    return AgentTool(
        name="bash",
        label="bash",
        description=BASH_DESCRIPTION,
        parameters=BASH_SCHEMA,
        execute=execute,
    )


__all__ = [
    "BASH_DESCRIPTION",
    "BASH_SCHEMA",
    "BashExecution",
    "BashPrepare",
    "MAX_TIMEOUT_SECONDS",
    "create_bash_tool",
]

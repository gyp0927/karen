"""The `powershell` tool (pi coding-agent's `core/tools/powershell.ts`).

Windows only, like pi: a `create_shell_tool` over PowerShell with pi's
`-NoProfile -NonInteractive -ExecutionPolicy Bypass -Command` invocation and
the UTF-8 output prefix. pi's `exposeSessionEnvironment` (PI_* env vars) and
spawn hooks are not ported yet.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from karen_agent.tools import create_shell_tool
from karen_agent.tools.local_shell import powershell_shell_config
from karen_agent.types import AgentTool
from karen_agent.utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES

UTF8_OUTPUT_PREFIX = "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}\n"

POWERSHELL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "PowerShell command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
    },
    "required": ["command"],
}

POWERSHELL_DESCRIPTION = (
    "Execute a PowerShell command in the current working directory. Returns stdout and stderr. "
    f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
    "(whichever is hit first). If truncated, full output is saved to a temp file. "
    "Optionally provide a timeout in seconds."
)


def create_powershell_tool(cwd: Optional[str] = None) -> AgentTool:
    """Create the `powershell` tool.

    Windows only: raises `ExecutionError("shell_unavailable", ...)` on other
    platforms or when no PowerShell executable is on PATH.
    """
    return create_shell_tool(
        cwd,
        name="powershell",
        label="powershell",
        description=POWERSHELL_DESCRIPTION,
        parameters=POWERSHELL_SCHEMA,
        # create_shell_tool joins prefix and command with "\n" — strip the
        # trailing newline so the result is pi's exact `${PREFIX}${command}`.
        command_prefix=UTF8_OUTPUT_PREFIX.rstrip("\n"),
        shell_config=powershell_shell_config(),
    )


__all__ = [
    "POWERSHELL_DESCRIPTION",
    "POWERSHELL_SCHEMA",
    "UTF8_OUTPUT_PREFIX",
    "create_powershell_tool",
]

"""The bash tool against a real local shell (skipped when no bash is available)."""

import asyncio
import os
import sys
from pathlib import Path

import pytest
from karen_ai import AbortController

from karen_agent.tools import BashExecution, create_bash_tool, create_shell_tool
from karen_agent.tools.local_shell import (
    POWERSHELL_ARGS,
    ExecutionError,
    powershell_shell_config,
    resolve_shell_config,
)

try:
    _SHELL = resolve_shell_config()
    HAS_BASH = True
except Exception:
    HAS_BASH = False

pytestmark = pytest.mark.skipif(not HAS_BASH, reason="no bash shell available on this machine")


async def _run(tool, params, signal=None, on_update=None):
    return await tool.execute("call-1", params, signal, on_update)


async def test_bash_echo(tmp_path):
    result = await _run(create_bash_tool(str(tmp_path)), {"command": "echo hello"})
    assert result.content[0].text == "hello\n"
    assert result.details is None


async def test_bash_no_output(tmp_path):
    result = await _run(create_bash_tool(str(tmp_path)), {"command": "true"})
    assert result.content[0].text == "(no output)"


async def test_bash_combines_stdout_and_stderr(tmp_path):
    result = await _run(create_bash_tool(str(tmp_path)), {"command": "echo out; echo err >&2"})
    assert result.content[0].text == "out\nerr\n"


async def test_bash_exit_code_error(tmp_path):
    with pytest.raises(RuntimeError, match=r"^Command exited with code 3$"):
        await _run(create_bash_tool(str(tmp_path)), {"command": "exit 3"})


async def test_bash_exit_code_with_output(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        await _run(create_bash_tool(str(tmp_path)), {"command": "echo boom; exit 1"})
    assert str(excinfo.value) == "boom\n\n\nCommand exited with code 1"


async def test_bash_runs_in_cwd(tmp_path):
    await _run(create_bash_tool(str(tmp_path)), {"command": "mkdir -p sub && touch sub/marker.txt"})
    assert (tmp_path / "sub" / "marker.txt").exists()


async def test_bash_timeout(tmp_path):
    with pytest.raises(RuntimeError, match=r"Command timed out after 1 seconds$"):
        await _run(create_bash_tool(str(tmp_path)), {"command": "sleep 30", "timeout": 1})


async def test_bash_timeout_validation(tmp_path):
    with pytest.raises(ValueError, match="^Invalid timeout: must be a finite number of seconds$"):
        await _run(create_bash_tool(str(tmp_path)), {"command": "true", "timeout": 0})
    with pytest.raises(ValueError, match="^Invalid timeout: maximum is 2147483.647 seconds$"):
        await _run(create_bash_tool(str(tmp_path)), {"command": "true", "timeout": 3_000_000})


async def test_bash_abort(tmp_path):
    controller = AbortController()

    async def abort_soon():
        await asyncio.sleep(0.2)
        controller.abort()

    task = asyncio.create_task(abort_soon())
    with pytest.raises(RuntimeError, match=r"Command aborted$"):
        await _run(
            create_bash_tool(str(tmp_path)), {"command": "sleep 30"}, signal=controller.signal
        )
    await task


async def test_bash_truncation_and_spill(tmp_path):
    result = await _run(create_bash_tool(str(tmp_path)), {"command": "seq 1 5000"})
    text = result.content[0].text
    assert "\n\n[Showing lines 3001-5000 of 5000. Full output: " in text
    assert result.details["truncation"]["truncatedBy"] == "lines"
    spill_path = result.details["fullOutputPath"]
    assert os.path.exists(spill_path)
    with open(spill_path, "rb") as handle:
        spilled = handle.read().decode()
    assert spilled.startswith("1\n2\n")
    assert spilled.rstrip("\n").endswith("5000")
    assert len(spilled.rstrip("\n").split("\n")) == 5000


async def test_bash_partial_last_line(tmp_path):
    # One 60000-byte line with no newline: tail truncation with a partial first line.
    result = await _run(
        create_bash_tool(str(tmp_path)), {"command": "head -c 60000 /dev/zero | tr '\\0' 'x'"}
    )
    text = result.content[0].text
    assert "[Showing last 50.0KB of line 1 (line is 58.6KB). Full output: " in text
    assert result.details["truncation"]["lastLinePartial"] is True


async def test_bash_command_prefix(tmp_path):
    tool = create_bash_tool(str(tmp_path), command_prefix="export KAREN_PREFIX_TEST=from-prefix")
    result = await _run(tool, {"command": "echo $KAREN_PREFIX_TEST"})
    assert result.content[0].text == "from-prefix\n"


async def test_bash_prepare_hook(tmp_path):
    def prepare(execution: BashExecution) -> None:
        execution.env["KAREN_PREPARE_TEST"] = "prepared"
        execution.command = "echo $KAREN_PREPARE_TEST"

    result = await _run(create_bash_tool(str(tmp_path), prepare=prepare), {"command": "ignored"})
    assert result.content[0].text == "prepared\n"


async def test_bash_prepare_hook_async(tmp_path):
    async def prepare(execution: BashExecution) -> None:
        await asyncio.sleep(0)
        execution.env["KAREN_PREPARE_ASYNC"] = "async-ok"

    result = await _run(create_bash_tool(str(tmp_path), prepare=prepare), {"command": "echo $KAREN_PREPARE_ASYNC"})
    assert result.content[0].text == "async-ok\n"


async def test_bash_on_update_streams_partials(tmp_path):
    updates = []
    result = await _run(
        create_bash_tool(str(tmp_path)),
        {"command": "echo first; sleep 0.3; echo second"},
        on_update=lambda partial: updates.append(partial),
    )
    assert result.content[0].text == "first\nsecond\n"
    assert updates[0].content == []  # initial empty snapshot (pi behavior)
    texts = ["".join(c.text for c in u.content) for u in updates if u.content]
    assert any("first" in t for t in texts)
    assert texts[-1] == "first\nsecond\n"


async def test_bash_custom_shell_path(tmp_path):
    result = await _run(
        create_bash_tool(str(tmp_path), shell_path=_SHELL.shell), {"command": "echo custom-shell"}
    )
    assert result.content[0].text == "custom-shell\n"


async def test_bash_bad_shell_path(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        await _run(create_bash_tool(str(tmp_path), shell_path=str(tmp_path / "nope.exe")), {"command": "true"})
    assert "Custom shell path not found" in str(excinfo.value)


async def test_bash_missing_cwd(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        await _run(create_bash_tool(str(tmp_path / "missing")), {"command": "true"})
    assert "Cannot execute bash commands." in str(excinfo.value)


# ---------------------------------------------------------------------------
# create_shell_tool generalization + powershell_shell_config (karen-coding-agent M2)
# ---------------------------------------------------------------------------


async def test_create_shell_tool_custom_identity_and_parameters(tmp_path):
    schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "PowerShell command to execute"},
            "timeout": {"type": "number"},
        },
        "required": ["command"],
    }
    tool = create_shell_tool(
        str(tmp_path), name="powershell", label="powershell", description="Run PowerShell", parameters=schema
    )
    assert tool.name == "powershell"
    assert tool.label == "powershell"
    assert tool.description == "Run PowerShell"
    assert tool.parameters["properties"]["command"]["description"] == "PowerShell command to execute"
    # still runs on the default bash config when no shell_config is passed
    result = await _run(tool, {"command": "echo via-custom-shell"})
    assert result.content[0].text == "via-custom-shell\n"


@pytest.mark.skipif(sys.platform != "win32", reason="powershell_shell_config is Windows-only")
def test_powershell_shell_config_finds_executable():
    config = powershell_shell_config()
    assert config.args == POWERSHELL_ARGS
    assert Path(config.shell).name.lower() in {"pwsh.exe", "powershell.exe"}


@pytest.mark.skipif(sys.platform == "win32", reason="only raises off Windows")
def test_powershell_shell_config_raises_off_windows():
    with pytest.raises(ExecutionError):
        powershell_shell_config()

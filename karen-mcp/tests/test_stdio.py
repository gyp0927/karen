"""`StdioTransport` against real child processes — pi's `test/stdio.test.ts`."""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, List

import pytest

from karen_mcp import McpClient, McpClientOptions, StdioTransport, StdioTransportOptions, to_llm_content

FIXTURE = Path(__file__).parent / "fixtures" / "stdio_server.py"
STUBBORN = Path(__file__).parent / "fixtures" / "stubborn_server.py"


async def until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


def _is_alive(pid: int) -> bool:
    if os.name == "nt":
        # `os.kill(pid, 0)` is TerminateProcess on Windows, so it cannot be used
        # as a liveness probe.
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _client() -> McpClient:
    return McpClient(McpClientOptions(name="stdio-test", version="1.0.0"))


async def test_connects_to_a_newline_delimited_mcp_server_and_captures_stderr():
    stderr: List[str] = []
    transport = StdioTransport(
        StdioTransportOptions(command=sys.executable, args=[str(FIXTURE)], on_stderr=stderr.append)
    )
    client = _client()
    await client.connect(transport)
    try:
        tools = await client.list_tools()
        assert [tool.model_dump(by_alias=True, exclude_none=True) for tool in tools] == [
            {"name": "echo", "inputSchema": {"type": "object"}}
        ]
        result = await client.call_tool("echo", {"text": "hello"})
        assert to_llm_content(result) == [{"type": "text", "text": "hello"}]
        assert isinstance(transport.pid, int)
        assert await until(lambda: "stdio fixture ready" in "".join(stderr))
        assert "stdio fixture ready" in transport.stderr
    finally:
        await client.close()
    assert client.connection_state == "closed"


async def test_reports_unparseable_output_and_keeps_serving_requests():
    transport = StdioTransport(StdioTransportOptions(command=sys.executable, args=[str(FIXTURE)]))
    client = _client()
    await client.connect(transport)
    errors: List[BaseException] = []
    client.on_error(errors.append)
    try:
        result = await client.call_tool("noise")
        # The stray line is a line the client cannot parse; the answer behind it
        # still has to arrive.
        assert to_llm_content(result) == [{"type": "text", "text": "noisy"}]
        assert await until(lambda: len(errors) == 1)
        assert isinstance(errors[0], json.JSONDecodeError)
        assert client.connection_state == "connected"
        assert (await client.call_tool("echo", {"text": "still here"})).content == [
            {"type": "text", "text": "still here"}
        ]
    finally:
        await client.close()


@pytest.mark.skipif(os.name != "nt", reason="`.cmd` shims are a Windows thing")
async def test_resolves_a_cmd_shim_on_path_and_runs_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # `npx` and `uvx` are `.cmd` shims on Windows and are named without their
    # extension: CreateProcess only appends `.exe`, so the transport has to
    # resolve the name itself. The space in the directory stands for program
    # paths like `C:\Program Files\nodejs\npx.cmd`.
    shim_dir = tmp_path / "with space"
    shim_dir.mkdir()
    shim = shim_dir / "mcp-fixture.cmd"
    shim.write_text(f'@echo off\r\n"{sys.executable}" "{FIXTURE}" %*\r\n', encoding="utf-8")
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ['PATH']}")
    transport = StdioTransport(StdioTransportOptions(command="mcp-fixture"))
    client = _client()
    await client.connect(transport)
    try:
        assert (await client.call_tool("echo", {"text": "through a shim"})).content == [
            {"type": "text", "text": "through a shim"}
        ]
    finally:
        await client.close()
    assert client.connection_state == "closed"


async def test_kills_a_server_that_ignores_shutdown_including_its_children():
    transport = StdioTransport(
        StdioTransportOptions(command=sys.executable, args=[str(STUBBORN)], close_timeout_ms=100)
    )
    client = _client()
    await client.connect(transport)
    grandchild: Any = None
    deadline = time.monotonic() + 10
    while grandchild is None and time.monotonic() < deadline:
        match = re.search(r"grandchild (\d+)", transport.stderr)
        if match:
            grandchild = int(match.group(1))
        else:
            await asyncio.sleep(0.01)
    assert isinstance(grandchild, int)

    started = time.monotonic()
    await client.close()
    assert time.monotonic() - started < 5
    assert await until(lambda: not _is_alive(grandchild), timeout=5)

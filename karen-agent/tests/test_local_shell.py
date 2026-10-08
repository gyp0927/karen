"""`run_shell_command`'s cancellation contract: no shell outlives its caller.

pi's TypeScript has no task cancellation, so this hole has no upstream
counterpart. It is reachable through every Python caller that can be
cancelled: Ctrl+C in the TUI's `!command`, a `wait_for` deadline, a run torn
down while a tool call is in flight.
"""

from __future__ import annotations

import asyncio

import pytest

from karen_agent.tools.local_shell import ExecutionError, resolve_shell_config, run_shell_command
from karen_agent.utils.output_capture import OutputCapture

try:
    resolve_shell_config()
    HAS_SHELL = True
except ExecutionError:
    HAS_SHELL = False

needs_shell = pytest.mark.skipif(not HAS_SHELL, reason="no shell available on this machine")


@needs_shell
async def test_cancelling_the_caller_kills_the_shell(tmp_path):
    """The child writes a marker one second after it starts. Cancelling the
    caller must take the shell (and its `sleep`) with it, so the marker is
    never written — a shell that keeps running behind a dead app still would.
    """
    started = tmp_path / "started.txt"
    marker = tmp_path / "alive.txt"
    spawned = {}
    command = (
        f'echo started > "{started.as_posix()}"; '
        f'sleep 1; '
        f'echo alive > "{marker.as_posix()}"'
    )

    task = asyncio.create_task(
        run_shell_command(
            command,
            cwd=str(tmp_path),
            capture=OutputCapture(),
            on_spawn=lambda proc: spawned.update(proc=proc),
        )
    )
    # The child's own `started` file, not the spawn call, is the clock: bash
    # can take a while to come up on Windows, and the `sleep` only begins once
    # the shell is actually running the command.
    for _ in range(1500):
        if started.exists():
            break
        await asyncio.sleep(0.01)
    assert started.exists(), "the shell never ran the command"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Held on purpose: a collected `Process` closes its transport, and
    # CPython's `__del__` kills the child — which would let this test report
    # "killed" even if the cancellation path killed nothing.
    proc = spawned.get("proc")
    assert proc is not None and proc.pid is not None

    await asyncio.sleep(1.5)
    assert not marker.exists(), "the shell outlived the caller that was cancelled"

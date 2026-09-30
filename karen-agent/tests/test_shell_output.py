"""execute_shell_with_capture (pi's harness/utils/shell-output.ts)."""

from __future__ import annotations

import pytest
from karen_ai import AbortController

from karen_agent.env import LocalExecutionEnv, ShellExecOptions
from karen_agent.result import Err, Ok, err, ok
from karen_agent.tools.local_shell import ExecutionError, resolve_shell_config
from karen_agent.utils.shell_output import (
    ShellCaptureOptions,
    ShellCaptureResult,
    execute_shell_with_capture,
)
from karen_agent.utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES

try:
    _SHELL = resolve_shell_config()
    HAS_BASH = True
except ExecutionError:
    HAS_BASH = False

needs_bash = pytest.mark.skipif(not HAS_BASH, reason="no bash shell available on this machine")


class _StubEnv:
    """ExecutionEnv-shaped stub returning a canned exec result."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    async def exec(self, command, options=None, signal=None):
        self.calls.append((command, options))
        if options is not None and options.on_update is not None:
            from karen_agent.utils.output_capture import OutputCapture

            capture = OutputCapture()
            capture.push(b"partial output\n")
            capture.finish()
            capture.flush()
            options.on_update(capture.snapshot())
        return self._result

    async def cleanup(self):
        pass


@needs_bash
async def test_success_collects_output(tmp_path):
    env = LocalExecutionEnv(str(tmp_path))
    chunks = []
    result = await execute_shell_with_capture(
        env,
        "printf 'hello\\nworld\\n'",
        ShellCaptureOptions(on_chunk=lambda chunk, _progress: chunks.append(chunk)),
    )
    assert isinstance(result, Ok)
    value = result.value
    assert value.exit_code == 0
    assert value.cancelled is False
    assert value.truncated is False
    assert value.output == "hello\nworld\n"
    # chunk accumulation reconstructs the final view exactly
    assert "".join(chunks) == value.output
    assert value.truncation.max_bytes == DEFAULT_MAX_BYTES
    assert value.truncation.max_lines == DEFAULT_MAX_LINES


@needs_bash
async def test_nonzero_exit_is_still_ok(tmp_path):
    env = LocalExecutionEnv(str(tmp_path))
    result = await execute_shell_with_capture(env, "echo oops; exit 7")
    assert isinstance(result, Ok)
    assert result.value.exit_code == 7
    assert "oops" in result.value.output


@needs_bash
async def test_progress_callback_sees_current_view(tmp_path):
    env = LocalExecutionEnv(str(tmp_path))
    seen = []

    def on_chunk(chunk, get_progress):
        progress = get_progress()
        seen.append((chunk, progress.output, progress.last_line_bytes))

    result = await execute_shell_with_capture(env, "echo abc", ShellCaptureOptions(on_chunk=on_chunk))
    assert isinstance(result, Ok)
    assert seen, "expected at least one chunk"
    _chunk, output_at_chunk, last_line_bytes = seen[-1]
    assert output_at_chunk == result.value.output
    assert last_line_bytes == result.value.last_line_bytes


@needs_bash
async def test_abort_returns_cancelled_ok(tmp_path):
    env = LocalExecutionEnv(str(tmp_path))
    controller = AbortController()

    import asyncio

    async def abort_soon():
        await asyncio.sleep(0.2)
        controller.abort()

    task = asyncio.create_task(abort_soon())
    result = await execute_shell_with_capture(env, "sleep 30", signal=controller.signal)
    await task
    assert isinstance(result, Ok)
    assert result.value.cancelled is True
    assert result.value.exit_code is None


async def test_execution_error_returned_as_err_by_default():
    stub = _StubEnv(err(ExecutionError("timeout", "timeout:1")))
    result = await execute_shell_with_capture(stub, "whatever")
    assert isinstance(result, Err)
    assert result.error.code == "timeout"


async def test_return_execution_errors_folds_into_ok():
    error = ExecutionError("spawn_error", "boom")
    stub = _StubEnv(err(error))
    result = await execute_shell_with_capture(
        stub, "whatever", ShellCaptureOptions(return_execution_errors=True)
    )
    assert isinstance(result, Ok)
    value = result.value
    assert value.exit_code is None
    assert value.cancelled is False
    assert value.execution_error is error
    assert "partial output" in value.output


async def test_aborted_stub_env_marks_cancelled():
    controller = AbortController()
    controller.abort()
    stub = _StubEnv(err(ExecutionError("aborted", "aborted")))
    result = await execute_shell_with_capture(stub, "whatever", signal=controller.signal)
    assert isinstance(result, Ok)
    assert result.value.cancelled is True


async def test_empty_output_view_when_nothing_published():
    stub = _StubEnv(ok(_exec_result()))

    class _SilentEnv(_StubEnv):
        async def exec(self, command, options=None, signal=None):
            return self._result

    result = await execute_shell_with_capture(_SilentEnv(ok(_exec_result())), "whatever")
    assert isinstance(result, Ok)
    assert result.value.output == ""
    assert result.value.truncated is False
    assert result.value.exit_code == 0


def _exec_result():
    from karen_agent.env import ShellExecResult
    from karen_agent.utils.output_capture import ShellOutputTruncation

    return ShellExecResult(
        exit_code=0,
        truncation=ShellOutputTruncation(
            truncated=False,
            total_lines=0,
            total_bytes=0,
            output_lines=0,
            output_bytes=0,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=DEFAULT_MAX_LINES,
            max_bytes=DEFAULT_MAX_BYTES,
        ),
    )


@needs_bash
async def test_cwd_and_env_forwarded(tmp_path):
    (tmp_path / "marker.txt").write_text("x")
    env = LocalExecutionEnv(str(tmp_path))
    result = await execute_shell_with_capture(
        env,
        "cat marker.txt; echo $KAREN_CAPTURE_TEST",
        ShellCaptureOptions(env={"KAREN_CAPTURE_TEST": "yes"}),
    )
    assert isinstance(result, Ok)
    assert result.value.exit_code == 0
    assert "x" in result.value.output and "yes" in result.value.output

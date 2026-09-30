"""LocalExecutionEnv: filesystem ops + shell exec (pi's NodeExecutionEnv).

Shell tests run against the real local bash (skipped when none is available),
mirroring test_tools_bash.py.
"""

from __future__ import annotations

import os

import pytest
from karen_ai import AbortController

from karen_agent.env import (
    ExecutionEnv,
    FileSystem,
    LocalExecutionEnv,
    Shell,
    ShellExecOptions,
    ShellOutputCaptureOptions,
    ShellOutputLimits,
)
from karen_agent.result import Err, Ok
from karen_agent.tools.local_shell import ExecutionError, resolve_shell_config

try:
    _SHELL = resolve_shell_config()
    HAS_BASH = True
except ExecutionError:
    HAS_BASH = False

needs_bash = pytest.mark.skipif(not HAS_BASH, reason="no bash shell available on this machine")


@pytest.fixture
def env(tmp_path):
    return LocalExecutionEnv(str(tmp_path))


def test_conforms_to_protocols(env):
    assert isinstance(env, ExecutionEnv)
    assert isinstance(env, FileSystem)
    assert isinstance(env, Shell)


# -- path primitives ---------------------------------------------------------


async def test_absolute_path_resolves_against_cwd(env, tmp_path):
    result = await env.absolute_path("sub/file.txt")
    assert isinstance(result, Ok)
    assert result.value == os.path.join(str(tmp_path), "sub", "file.txt")


async def test_absolute_path_keeps_absolute(env, tmp_path):
    absolute = os.path.join(str(tmp_path), "already.txt")
    result = await env.absolute_path(absolute)
    assert result.value == absolute


async def test_absolute_path_expands_tilde(env):
    result = await env.absolute_path("~/x.txt")
    assert result.value == os.path.join(os.path.expanduser("~"), "x.txt")


async def test_join_path_normalizes(env):
    result = await env.join_path(["a", "b", "..", "c.txt"])
    assert result.value == os.path.normpath(os.path.join("a", "b", "..", "c.txt"))


# -- reads / writes ------------------------------------------------------------


async def test_write_and_read_text_file(env):
    assert isinstance(await env.write_file("sub/dir/note.txt", "héllo"), Ok)
    result = await env.read_text_file("sub/dir/note.txt")
    assert result.value == "héllo"


async def test_read_text_file_missing_maps_not_found(env):
    result = await env.read_text_file("nope.txt")
    assert isinstance(result, Err)
    assert result.error.code == "not_found"
    assert result.error.path and result.error.path.endswith("nope.txt")


async def test_append_file_creates_and_appends(env):
    await env.write_file("log.txt", "a")
    assert isinstance(await env.append_file("log.txt", "b"), Ok)
    assert isinstance(await env.append_file("new/deep/log.txt", "x"), Ok)
    assert (await env.read_text_file("log.txt")).value == "ab"
    assert (await env.read_text_file("new/deep/log.txt")).value == "x"


async def test_binary_roundtrip(env):
    payload = bytes(range(256))
    await env.write_file("bin.dat", payload)
    result = await env.read_binary_file("bin.dat")
    assert result.value == payload


async def test_rename_file_replaces_destination(env):
    await env.write_file("a.txt", "one")
    await env.write_file("b.txt", "two")
    assert isinstance(await env.rename_file("a.txt", "b.txt"), Ok)
    assert (await env.read_text_file("b.txt")).value == "one"
    assert (await env.exists("a.txt")).value is False


async def test_rename_missing_source_is_not_found(env):
    result = await env.rename_file("missing.txt", "b.txt")
    assert isinstance(result, Err)
    assert result.error.code == "not_found"


# -- metadata -----------------------------------------------------------------


async def test_file_info_file_and_directory(env, tmp_path):
    await env.write_file("data.txt", "12345")
    info = (await env.file_info("data.txt")).value
    assert info.name == "data.txt"
    assert info.kind == "file"
    assert info.size == 5
    assert info.mtime_ms > 0

    dir_info = (await env.file_info(".")).value
    assert dir_info.kind == "directory"


async def test_list_dir_skips_unsupported_and_aborts_cleanly(env):
    await env.write_file("b.txt", "")
    await env.write_file("a.txt", "")
    await env.create_dir("sub")
    infos = (await env.list_dir(".")).value
    names = {info.name for info in infos}
    assert names == {"a.txt", "b.txt", "sub"}
    kinds = {info.name: info.kind for info in infos}
    assert kinds["sub"] == "directory"

    missing = await env.list_dir("no-such-dir")
    assert isinstance(missing, Err)
    assert missing.error.code == "not_found"


async def test_canonical_path_resolves_dotdot(env, tmp_path):
    await env.create_dir("real")
    result = await env.canonical_path("real/../real")
    assert isinstance(result, Ok)
    assert result.value == os.path.realpath(os.path.join(str(tmp_path), "real"))

    missing = await env.canonical_path("ghost")
    assert isinstance(missing, Err)
    assert missing.error.code == "not_found"


async def test_exists_true_false_and_error_passthrough(env):
    await env.write_file("here.txt", "")
    assert (await env.exists("here.txt")).value is True
    assert (await env.exists("absent.txt")).value is False


# -- mutations ------------------------------------------------------------------


async def test_create_dir_recursive_and_plain(env):
    assert isinstance(await env.create_dir("deep/nested/dir"), Ok)
    assert (await env.file_info("deep/nested/dir")).value.kind == "directory"
    # Non-recursive with missing parents fails.
    result = await env.create_dir("other/missing-parent/dir", {"recursive": False})
    assert isinstance(result, Err)


async def test_remove_file_dir_and_force(env):
    await env.write_file("junk.txt", "x")
    assert isinstance(await env.remove("junk.txt"), Ok)
    assert (await env.exists("junk.txt")).value is False

    await env.write_file("tree/a/b.txt", "x")
    non_recursive = await env.remove("tree")
    assert isinstance(non_recursive, Err)
    assert isinstance(await env.remove("tree", {"recursive": True}), Ok)
    assert (await env.exists("tree")).value is False

    # force swallows missing paths
    assert isinstance(await env.remove("ghost.txt", {"force": True}), Ok)
    missing = await env.remove("ghost.txt")
    assert isinstance(missing, Err)
    assert missing.error.code == "not_found"


async def test_temp_dir_and_file(env):
    directory = (await env.create_temp_dir("karen-test-")).value
    assert os.path.isdir(directory)
    assert os.path.basename(directory).startswith("karen-test-")

    file_path = (await env.create_temp_file({"prefix": "pre-", "suffix": ".log"})).value
    assert os.path.isfile(file_path)
    name = os.path.basename(file_path)
    assert name.startswith("pre-") and name.endswith(".log")
    # defaults
    plain = (await env.create_temp_file()).value
    assert os.path.isfile(plain)


# -- abort propagation -------------------------------------------------------------


async def test_aborted_signal_short_circuits_fs_ops(env):
    controller = AbortController()
    controller.abort()
    for result in [
        await env.read_text_file("x.txt", signal=controller.signal),
        await env.write_file("x.txt", "y", signal=controller.signal),
        await env.file_info("x.txt", signal=controller.signal),
        await env.list_dir(".", signal=controller.signal),
        await env.create_dir("d", signal=controller.signal),
        await env.remove("x.txt", signal=controller.signal),
    ]:
        assert isinstance(result, Err)
        assert result.error.code == "aborted"


# -- text line reader -----------------------------------------------------------


async def test_text_line_reader_lines_and_torn_tail(env):
    await env.write_file("lines.txt", "one\ntwo\nthree")
    opened = await env.open_text_line_reader("lines.txt")
    assert isinstance(opened, Ok)
    reader = opened.value
    try:
        first = (await reader.read_line()).value
        assert first is not None and first.text == "one" and first.terminated is True
        second = (await reader.read_line()).value
        assert second is not None and second.text == "two" and second.terminated is True
        third = (await reader.read_line()).value
        assert third is not None and third.text == "three" and third.terminated is False
        assert (await reader.read_line()).value is None
    finally:
        await reader.close()


async def test_read_text_lines_max_lines(env):
    await env.write_file("many.txt", "a\nb\nc\nd\n")
    assert (await env.read_text_lines("many.txt")).value == ["a", "b", "c", "d"]
    assert (await env.read_text_lines("many.txt", {"max_lines": 2})).value == ["a", "b"]
    assert (await env.read_text_lines("many.txt", {"max_lines": 0})).value == []


async def test_open_text_line_reader_missing(env):
    result = await env.open_text_line_reader("nope.txt")
    assert isinstance(result, Err)
    assert result.error.code == "not_found"


async def test_text_line_reader_multibyte_and_empty_lines(env):
    await env.write_file("multi.txt", "héllo\n\nwörld\n")
    assert (await env.read_text_lines("multi.txt")).value == ["héllo", "", "wörld"]


# -- shell exec -----------------------------------------------------------------


@needs_bash
async def test_exec_echo_and_exit_code(env, tmp_path):
    updates = []
    result = await env.exec(
        "echo hello && echo world",
        ShellExecOptions(on_update=updates.append),
    )
    assert isinstance(result, Ok)
    assert result.value.exit_code == 0
    assert updates, "expected published output snapshots"
    assert "hello" in updates[-1].text and "world" in updates[-1].text

    failing = await env.exec("exit 3")
    assert isinstance(failing, Ok)
    assert failing.value.exit_code == 3


@needs_bash
async def test_exec_cwd_env_and_inherit(env, tmp_path, monkeypatch):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "marker.txt").write_text("x")
    result = await env.exec("ls marker.txt", ShellExecOptions(cwd="sub"))
    assert isinstance(result, Ok)
    assert result.value.exit_code == 0

    seen = []
    await env.exec("echo $KAREN_ENV_TEST", ShellExecOptions(env={"KAREN_ENV_TEST": "merged"}, on_update=seen.append))
    assert "merged" in seen[-1].text

    # inherit_env=False hides process env (custom probe var: Git Bash re-establishes
    # PATH via its own profile, so PATH itself can't prove inheritance)
    monkeypatch.setenv("KAREN_INHERIT_PROBE", "yes")
    hidden = []
    await env.exec(
        "echo ${KAREN_INHERIT_PROBE:+has-probe}",
        ShellExecOptions(env={}, inherit_env=False, on_update=hidden.append),
    )
    assert "has-probe" not in hidden[-1].text
    inherited = []
    await env.exec(
        "echo ${KAREN_INHERIT_PROBE:+has-probe}",
        ShellExecOptions(env={}, inherit_env=True, on_update=inherited.append),
    )
    assert "has-probe" in inherited[-1].text


@needs_bash
async def test_exec_capture_limits_and_spill(env):
    # 100 numbered lines with a tiny line cap: tail retention keeps the last lines
    # and spill preserves the complete output.
    command = "for i in $(seq 1 100); do echo line-$i; done"
    result = await env.exec(
        command,
        ShellExecOptions(
            capture=ShellOutputCaptureOptions(
                limits=ShellOutputLimits(max_bytes=1024 * 1024, max_lines=10, retain="tail"),
                spill=True,
            )
        ),
    )
    assert isinstance(result, Ok)
    assert result.value.truncation.truncated is True
    assert result.value.truncation.truncated_by == "lines"
    assert result.value.spill_path is not None
    with open(result.value.spill_path, "r", encoding="utf-8") as file:
        spilled = file.read()
    assert "line-1\n" in spilled and "line-100\n" in spilled


@needs_bash
async def test_exec_timeout_returns_execution_error(env):
    result = await env.exec("sleep 30", ShellExecOptions(timeout=0.2))
    assert isinstance(result, Err)
    assert result.error.code == "timeout"


@needs_bash
async def test_exec_abort_kills_process(env):
    controller = AbortController()

    async def abort_soon():
        import asyncio

        await asyncio.sleep(0.2)
        controller.abort()

    import asyncio

    task = asyncio.create_task(abort_soon())
    result = await env.exec("sleep 30", signal=controller.signal)
    await task
    assert isinstance(result, Err)
    assert result.error.code == "aborted"


async def test_exec_pre_aborted_signal(env):
    controller = AbortController()
    controller.abort()
    result = await env.exec("echo hi", signal=controller.signal)
    assert isinstance(result, Err)
    assert result.error.code == "aborted"


async def test_exec_bad_shell_path_is_shell_unavailable(tmp_path):
    broken = LocalExecutionEnv(str(tmp_path), shell_path=str(tmp_path / "no-such-bash.exe"))
    result = await broken.exec("echo hi")
    assert isinstance(result, Err)
    assert result.error.code == "shell_unavailable"


@needs_bash
async def test_exec_missing_cwd_is_spawn_error(tmp_path):
    env = LocalExecutionEnv(str(tmp_path / "does-not-exist"))
    result = await env.exec("echo hi")
    assert isinstance(result, Err)
    assert result.error.code == "spawn_error"

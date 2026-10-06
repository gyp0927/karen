"""RPC-mode protocol tests: command dispatch, event stream shaping, and
`new_session` rebinding — driven offline with the faux provider over
in-memory lines, plus one child-process test over real stdin/stdout pipes.
"""

import asyncio
import json
import sys

from karen_ai import create_models
from karen_ai.providers import faux_assistant_message, faux_model, register_faux_provider
from karen_coding_agent import AgentSession
from karen_coding_agent.rpc import RpcServer, RpcSession, run_rpc_mode


async def _make_session(tmp_path, responses, **kwargs):
    models = create_models()
    registration = register_faux_provider(responses=responses)
    models.set_provider(registration.provider)
    kwargs.setdefault("sessions_root", str(tmp_path / "sessions"))
    kwargs.setdefault("fresh", True)
    session = AgentSession(cwd=str(tmp_path), models=models, model=registration.get_model(), **kwargs)
    await session.open()
    return session


async def _wait_for_busy(session, max_loops: int = 5000):
    """Wait until the spawned prompt task is running (it sets `_active_run`).

    Unlike waiting for idle, this cannot return too early: it waits for the
    run to *appear*, and a started run only disappears again when done.
    """
    for _ in range(max_loops):
        if session.agent._active_run is not None:
            return
        await asyncio.sleep(0.001)
    raise AssertionError("prompt never became active")


def _response(lines, command_id=None, command=None):
    parsed = [json.loads(line) for line in lines]
    for obj in parsed:
        if obj.get("type") == "response" and (
            command_id is None or obj.get("id") == command_id
        ) and (command is None or obj.get("command") == command):
            return obj
    raise AssertionError(f"no response for id={command_id} command={command} in {lines}")


# ---------------------------------------------------------------------------
# basic prompt + state
# ---------------------------------------------------------------------------


async def test_rpc_prompt_then_get_state(tmp_path):
    session = await _make_session(tmp_path, [faux_assistant_message("hello!")])
    server = RpcServer(RpcSession(session), input_iter=[], emit=lambda _l: None)

    resp = await server.handle_command({"type": "prompt", "message": "hi", "id": "p1"})
    assert resp["success"] and resp["data"]["disposition"] == "started"
    await server.wait_for_idle()

    state = await server.handle_command({"type": "get_state", "id": "s1"})
    assert state["success"] is True
    data = state["data"]
    assert data["messageCount"] >= 3  # system + user + assistant
    assert data["sessionId"]
    assert data["isStreaming"] is False
    await session.close()


# ---------------------------------------------------------------------------
# busy prompt -> queued; steer/follow_up -> clear_queue
# ---------------------------------------------------------------------------


async def test_rpc_prompt_is_queued_when_busy(tmp_path):

    session = await _make_session(tmp_path, [faux_assistant_message("one"), faux_assistant_message("two")])
    server = RpcServer(RpcSession(session), input_iter=[], emit=lambda _l: None)

    first = await server.handle_command({"type": "prompt", "message": "one"})
    assert first["data"]["disposition"] == "started"
    await _wait_for_busy(session)
    second = await server.handle_command({"type": "prompt", "message": "two"})
    assert second["data"]["disposition"] == "queued"  # steered into the active run
    await server.wait_for_idle()
    # the steered message was folded into the running turn, not dropped
    user_texts = [m.content[0].text for m in session.agent.state.messages if m.role == "user"]
    assert user_texts == ["one", "two"]
    await session.close()


async def test_rpc_steer_follow_up_and_clear_queue(tmp_path):
    session = await _make_session(tmp_path, [faux_assistant_message("done")])
    server = RpcServer(RpcSession(session), input_iter=[], emit=lambda _l: None)

    assert (await server.handle_command({"type": "steer", "message": "steer-me"}))["data"]["disposition"] == "queued"
    assert (await server.handle_command({"type": "follow_up", "message": "follow-me"}))["data"]["disposition"] == "queued"
    cleared = await server.handle_command({"type": "clear_queue"})
    assert cleared["data"]["steering"] == ["steer-me"]
    assert cleared["data"]["followUp"] == ["follow-me"]
    empty = await server.handle_command({"type": "clear_queue"})
    assert empty["data"]["steering"] == [] and empty["data"]["followUp"] == []
    await session.close()


# ---------------------------------------------------------------------------
# unknown commands / malformed input
# ---------------------------------------------------------------------------


async def test_rpc_unknown_command_reports_error(tmp_path):
    session = await _make_session(tmp_path, [faux_assistant_message("x")])
    lines = []
    server = RpcServer(RpcSession(session), input_iter=[], emit=lines.append)
    await server.handle_command({"type": "does_not_exist", "id": "z"})
    error = _response(lines, "z", "does_not_exist")
    assert error["success"] is False
    assert "Unknown command" in error["error"]
    await session.close()


async def test_rpc_malformed_lines_are_reported(tmp_path):
    session = await _make_session(tmp_path, [faux_assistant_message("x")])
    lines = []

    async def input_lines():
        yield "{not json"
        yield "42"
        yield json.dumps({"type": "get_state"})

    exit_code = await run_rpc_mode(session, input_iter=input_lines(), emit=lines.append)
    assert exit_code == 0
    parsed = [json.loads(line) for line in lines]
    errors = [obj for obj in parsed if obj.get("type") == "response" and obj.get("success") is False]
    assert len(errors) == 2
    assert all(e["command"] == "parse" for e in errors)
    state = _response(lines, None, "get_state")
    assert state["success"] is True
    await session.close()


# ---------------------------------------------------------------------------
# the real stdio loop: child process, real pipes
# ---------------------------------------------------------------------------

#: Drives `run_rpc_mode` in a child process over its actual stdin/stdout, so
#: the blocking stdin reader and the flushing stdout emitter are exercised.
_RPC_CHILD_SCRIPT = """
import asyncio, tempfile, sys
from karen_ai import create_models
from karen_ai.providers import faux_assistant_message, register_faux_provider
from karen_coding_agent import AgentSession
from karen_coding_agent.rpc import run_rpc_mode


async def main():
    models = create_models()
    registration = register_faux_provider(responses=[faux_assistant_message("pong")])
    models.set_provider(registration.provider)
    tmp = tempfile.mkdtemp(prefix="karen-rpc-child-")
    session = AgentSession(
        cwd=tmp,
        models=models,
        model=registration.get_model(),
        sessions_root=tmp,
        fresh=True,
    )
    await session.open()
    try:
        return await run_rpc_mode(session)
    finally:
        await session.close()


sys.exit(asyncio.run(main()))
"""


async def test_rpc_stdio_loop_over_real_pipes():
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _RPC_CHILD_SCRIPT,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def read_line(timeout: float = 60) -> dict:
        raw = await asyncio.wait_for(proc.stdout.readline(), timeout)
        assert raw, f"child closed stdout (stderr: {await _drain_stderr(proc)})"
        return json.loads(raw.decode("utf-8"))

    async def command(obj: dict) -> dict:
        proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        await proc.stdin.drain()
        while True:
            line = await read_line()
            if line.get("type") == "response" and line.get("id") == obj.get("id"):
                return line

    try:
        header = await read_line()
        assert header.get("kind") == "header" and header.get("id")

        started = await command({"type": "prompt", "message": "ping", "id": "p1"})
        assert started["success"] and started["data"]["disposition"] == "started"

        # events stream between the response and the next command's response
        events = []
        while True:
            line = await read_line()
            events.append(line.get("type"))
            if line.get("type") == "agent_end":
                break
        assert "message_update" in events and events[-1] == "agent_end"

        text = await command({"type": "get_last_assistant_text", "id": "t1"})
        assert text["data"]["text"] == "pong"

        proc.stdin.close()
        assert await asyncio.wait_for(proc.wait(), 30) == 0
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


async def _drain_stderr(proc) -> str:
    try:
        return (await asyncio.wait_for(proc.stderr.read(), 5)).decode("utf-8", "replace")
    except asyncio.TimeoutError:  # pragma: no cover - diagnostics only
        return "<stderr read timed out>"


# ---------------------------------------------------------------------------
# entries / messages / last assistant text
# ---------------------------------------------------------------------------


async def test_rpc_get_entries_and_messages(tmp_path):
    session = await _make_session(tmp_path, [faux_assistant_message("hi there")])
    server = RpcServer(RpcSession(session), input_iter=[], emit=lambda _l: None)
    await server.handle_command({"type": "prompt", "message": "hi"})
    await server.wait_for_idle()

    entries = await server.handle_command({"type": "get_entries"})
    assert entries["success"] is True
    assert isinstance(entries["data"]["entries"], list) and entries["data"]["entries"]
    assert entries["data"]["leafId"] is not None

    messages = await server.handle_command({"type": "get_messages"})
    assert messages["success"] is True
    assert len(messages["data"]["messages"]) >= 3

    text = await server.handle_command({"type": "get_last_assistant_text"})
    assert text["data"]["text"] == "hi there"
    await session.close()


# ---------------------------------------------------------------------------
# compaction switches
# ---------------------------------------------------------------------------


async def test_rpc_set_auto_compaction_flags(tmp_path):
    session = await _make_session(tmp_path, [faux_assistant_message("x")])
    server = RpcServer(RpcSession(session), input_iter=[], emit=lambda _l: None)
    assert (await server.handle_command({"type": "set_auto_compaction", "enabled": False}))["success"]
    assert server.auto_compaction_enabled is False
    assert (await server.handle_command({"type": "set_auto_compaction", "enabled": True}))["success"]
    assert server.auto_compaction_enabled is True


async def test_rpc_manual_compact_reports_outcome(tmp_path):
    session = await _make_session(tmp_path, [faux_assistant_message("short")])
    server = RpcServer(RpcSession(session), input_iter=[], emit=lambda _l: None)
    await server.handle_command({"type": "prompt", "message": "short"})
    await server.wait_for_idle()
    result = await server.handle_command({"type": "compact"})
    assert result["success"] is True
    assert "compacted" in result["data"]
    await session.close()


# ---------------------------------------------------------------------------
# new_session rebind
# ---------------------------------------------------------------------------


async def test_rpc_new_session_rebinds_holder_and_server(tmp_path):
    session = await _make_session(tmp_path, [faux_assistant_message("one")])
    holder = RpcSession(session)
    server = RpcServer(holder, input_iter=[], emit=lambda _l: None)
    original_id = session.session.metadata.id

    await server.handle_command({"type": "prompt", "message": "one"})
    await server.wait_for_idle()
    result = await server.handle_command({"type": "new_session"})
    assert result["success"] and result["data"]["cancelled"] is False
    assert holder.agent_session is not session
    assert holder.agent_session.session.metadata.id != original_id
    assert server.session is holder.agent_session
    await holder.agent_session.close()


async def test_rpc_new_session_keeps_the_structured_prompt(tmp_path):
    sections = {"preamble": "You are karen.", "cwd": "<cwd>\n/tmp\n</cwd>"}
    session = await _make_session(
        tmp_path, [faux_assistant_message("x")], system_prompt_sections=sections
    )
    holder = RpcSession(session)
    server = RpcServer(holder, input_iter=[], emit=lambda _l: None)

    await server.handle_command({"type": "new_session"})

    assert holder.agent_session.system_prompt_sections == sections
    rebound = holder.agent_session.agent.state.messages[0]
    assert rebound.sections == sections and rebound.content == ""
    await holder.agent_session.close()


# ---------------------------------------------------------------------------
# model commands (faux provider carries the catalog)
# ---------------------------------------------------------------------------


async def test_rpc_set_model_and_available_models(tmp_path):
    models = create_models()
    registration = register_faux_provider(
        responses=[], models=[faux_model(), faux_model(id="faux-2")]
    )
    models.set_provider(registration.provider)
    session = AgentSession(
        cwd=str(tmp_path),
        models=models,
        model=registration.get_model(),
        sessions_root=str(tmp_path / "sessions"),
        fresh=True,
    )
    await session.open()
    server = RpcServer(RpcSession(session), input_iter=[], emit=lambda _l: None)

    available = await server.handle_command({"type": "get_available_models"})
    assert available["success"] is True
    ids = [m.get("id") for m in available["data"]["models"]]
    assert "faux-2" in ids

    missing = await server.handle_command({"type": "set_model", "provider": "faux", "modelId": "nope"})
    assert missing["success"] is False
    assert "Model not found" in missing["error"]

    switched = await server.handle_command({"type": "set_model", "provider": "faux", "modelId": "faux-2"})
    assert switched["success"] is True
    assert session.model.id == "faux-2"
    await session.close()

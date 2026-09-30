"""Built-in tools running end-to-end through the agent loop with a scripted stream fn."""

import asyncio
import time

import pytest
from karen_ai import DoneEvent, ErrorEvent, EventStream, TextContent, ToolCall, UserMessage
from karen_ai.providers import faux_assistant_message, faux_model

from karen_agent import AgentContext, AgentLoopConfig, agent_loop
from karen_agent.tools import create_builtin_tools
from karen_agent.tools.local_shell import resolve_shell_config

try:
    resolve_shell_config()
    HAS_BASH = True
except Exception:
    HAS_BASH = False

_DRAIN_TIMEOUT = 30


def _now() -> int:
    return int(time.time() * 1000)


def _assistant(content, stop_reason="stop"):
    message = faux_assistant_message(content, api="openai-responses", provider="openai", model="mock")
    message.stop_reason = stop_reason
    return message


def _stream_fn(script):
    calls = []

    def fn(model, context, options=None):
        calls.append((model, context, options))
        stream = EventStream(
            lambda e: e.type in ("done", "error"),
            lambda e: e.message if e.type == "done" else e.error,
        )
        if len(calls) > len(script):
            failure = _assistant([], stop_reason="error")
            failure.error_message = "test script exhausted"
            stream.push(ErrorEvent(reason="error", error=failure))
            return stream
        for event in script[len(calls) - 1]:
            stream.push(event)
        return stream

    return fn, calls


async def _drain(stream):
    return await asyncio.wait_for(stream.result(), timeout=_DRAIN_TIMEOUT)


def _identity_convert(messages):
    return [m for m in messages if getattr(m, "role", None) in ("system", "user", "assistant", "toolResult")]


@pytest.mark.skipif(not HAS_BASH, reason="no bash shell available on this machine")
async def test_builtin_tools_round_trip_through_loop(tmp_path):
    """write → read → bash, all driven by scripted assistant tool calls."""
    fn, _calls = _stream_fn(
        [
            [DoneEvent(reason="stop", message=_assistant([
                ToolCall(id="c1", name="write", arguments={"path": "note.txt", "content": "from-the-loop\n"}),
            ]))],
            [DoneEvent(reason="stop", message=_assistant([
                ToolCall(id="c2", name="read", arguments={"path": "note.txt"}),
                ToolCall(id="c3", name="bash", arguments={"command": "cat note.txt"}),
            ]))],
            [DoneEvent(reason="stop", message=_assistant([TextContent(text="done")]))],
        ]
    )
    context = AgentContext(messages=[], tools=create_builtin_tools(str(tmp_path)))
    config = AgentLoopConfig(model=faux_model(provider="openai", api="openai-responses"), convert_to_llm=_identity_convert)
    stream = agent_loop([_user_msg()], context, config, None, fn)
    messages = await _drain(stream)

    results = [m for m in messages if getattr(m, "role", None) == "toolResult"]
    assert [r.tool_name for r in results] == ["write", "read", "bash"]
    assert results[0].content[0].text == "Successfully wrote to note.txt"
    assert results[1].content[0].text == "from-the-loop\n"
    assert results[2].content[0].text == "from-the-loop\n"
    assert (tmp_path / "note.txt").read_bytes() == b"from-the-loop\n"
    # Second LLM call must have seen the write result in its context.
    assert _calls[1][1].messages[-1].role == "toolResult"


async def test_loop_surfaces_tool_errors_as_error_results(tmp_path):
    fn, _calls = _stream_fn(
        [
            [DoneEvent(reason="stop", message=_assistant([
                ToolCall(id="c1", name="edit", arguments={"path": "missing.txt", "edits": [{"oldText": "a", "newText": "b"}]}),
            ]))],
            [DoneEvent(reason="stop", message=_assistant([TextContent(text="handled")]))],
        ]
    )
    context = AgentContext(messages=[], tools=create_builtin_tools(str(tmp_path)))
    config = AgentLoopConfig(model=faux_model(provider="openai", api="openai-responses"), convert_to_llm=_identity_convert)
    stream = agent_loop([_user_msg()], context, config, None, fn)
    messages = await _drain(stream)

    result = next(m for m in messages if getattr(m, "role", None) == "toolResult")
    assert result.is_error is True
    assert "Could not edit file: missing.txt. Error code: ENOENT." in result.content[0].text


def _user_msg():
    return UserMessage(content="go", timestamp=_now())

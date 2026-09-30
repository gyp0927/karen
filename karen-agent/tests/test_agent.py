"""Tests for `karen_agent.agent` (port of pi's `agent.ts` Agent class).

Streams are scripted synchronously: karen-ai's EventStream buffers pushes, so the
consumer sees the full event sequence even though the producer ran to completion
before iteration started.
"""

import asyncio
import time

import pytest

from karen_ai import (
    DoneEvent,
    ErrorEvent,
    EventStream,
    ImageContent,
    SystemMessage,
    TextContent,
    ToolCall,
    UserMessage,
)
from karen_ai.providers import faux_assistant_message, faux_model
from karen_agent import (
    Agent,
    AgentInitialState,
    AgentLoopTurnUpdate,
    AgentTool,
    AgentToolResult,
    default_convert_to_llm,
)

_TIMEOUT = 10


def _now() -> int:
    return int(time.time() * 1000)


def _user(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=_now())


def _assistant(content, stop_reason="stop"):
    message = faux_assistant_message(content, api="openai-responses", provider="openai", model="mock")
    message.stop_reason = stop_reason
    return message


def _tool_call(name, arguments, id="call-1"):
    return ToolCall(id=id, name=name, arguments=arguments)


def _mock_assistant_stream():
    return EventStream(
        lambda e: e.type in ("done", "error"),
        lambda e: e.message if e.type == "done" else e.error,
    )


def _done(message):
    return DoneEvent(reason=message.stop_reason, message=message)


def _stream_fn(script):
    """A StreamFn that plays back one event list per call, in order."""
    calls = []

    def fn(model, context, options=None):
        calls.append((model, context, options))
        stream = _mock_assistant_stream()
        if len(calls) > len(script):
            failure = _assistant([], stop_reason="error")
            failure.error_message = "test script exhausted: unexpected extra LLM call"
            stream.push(ErrorEvent(reason="error", error=failure))
            return stream
        for event in script[len(calls) - 1]:
            stream.push(event)
        return stream

    return fn, calls


def _no_stream(*args, **kwargs):
    raise AssertionError("stream must not run")


def _echo_tool(**overrides):
    async def execute(tool_call_id, params, signal, on_update):
        return AgentToolResult(content=[TextContent(text=str(params["value"]))], details={"echoed": params["value"]})

    options = dict(
        name="echo",
        description="Echo back the value",
        label="Echo",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        execute=execute,
    )
    options.update(overrides)
    return AgentTool(**options)


def _message_texts(messages):
    texts = []
    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(getattr(block, "text", "") for block in content if getattr(block, "text", None))
        elif isinstance(message, dict):
            texts.append(str(message))
    return texts


def _roles(messages):
    return [getattr(message, "role", message.get("role") if isinstance(message, dict) else None) for message in messages]


# ---------------------------------------------------------------------------
# initial state
# ---------------------------------------------------------------------------


def test_initial_state_seeds_system_message_and_tools():
    tool = _echo_tool()
    agent = Agent(
        stream_fn=_no_stream,
        initial_state=AgentInitialState(system_prompt="Be nice.", tools=[tool]),
    )
    state = agent.state
    assert _roles(state.messages) == ["system"]
    seeded = state.messages[0]
    assert seeded.content == "Be nice."
    assert [t.name for t in seeded.tools_added] == ["echo"]
    assert state.system_prompt == "Be nice."
    assert state.thinking_level == "off"
    assert state.model.id == "unknown"


def test_initial_messages_starting_with_system_are_kept():
    existing = SystemMessage(content="Already here.", timestamp=0)
    agent = Agent(
        stream_fn=_no_stream,
        initial_state=AgentInitialState(system_prompt="Ignored.", messages=[existing]),
    )
    assert agent.state.messages == [existing]
    assert agent.state.system_prompt == "Already here."


def test_state_setters_copy_top_level_lists():
    agent = Agent(stream_fn=_no_stream, initial_state=AgentInitialState())
    tools = [_echo_tool()]
    messages = [_user("hi")]
    agent.state.tools = tools
    agent.state.messages = messages
    tools.append(_echo_tool(name="second", label="Second"))
    messages.append(_user("extra"))
    assert len(agent.state.tools) == 1
    assert len(agent.state.messages) == 1


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


async def test_prompt_runs_loop_and_updates_state():
    fn, calls = _stream_fn([[_done(_assistant([TextContent(text="Hi there!")]))]])
    agent = Agent(stream_fn=fn)

    await agent.prompt("hello")
    await agent.wait_for_idle()

    state = agent.state
    assert _roles(state.messages) == ["user", "assistant"]
    assert "Hi there!" in _message_texts(state.messages)
    assert state.is_streaming is False
    assert state.streaming_message is None
    assert state.error_message is None
    assert state.pending_tool_calls == set()
    assert len(calls) == 1


async def test_prompt_accepts_images_and_message_lists():
    fn, calls = _stream_fn(
        [
            [_done(_assistant([TextContent(text="one")]))],
            [_done(_assistant([TextContent(text="two")]))],
            [_done(_assistant([TextContent(text="three")]))],
        ]
    )
    agent = Agent(stream_fn=fn)
    image = ImageContent(data="aGk=", mime_type="image/png")
    await agent.prompt("look", images=[image])

    first_content = agent.state.messages[0].content
    assert isinstance(first_content, list)
    assert first_content[0].text == "look"
    assert first_content[1] == image

    await agent.prompt(_user("second"))
    await agent.prompt([_user("third"), _user("fourth")])
    assert len(calls) == 3
    assert _roles(agent.state.messages)[:2] == ["user", "assistant"]


async def test_prompt_while_active_raises():
    stream = _mock_assistant_stream()
    agent = Agent(stream_fn=lambda model, context, options=None: stream)
    task = asyncio.create_task(agent.prompt("first"))
    await asyncio.sleep(0.05)
    assert agent.state.is_streaming is True
    assert agent.signal is not None and agent.signal.aborted is False

    with pytest.raises(RuntimeError, match="already processing a prompt"):
        await agent.prompt("second")
    with pytest.raises(RuntimeError, match="Wait for completion before continuing"):
        await agent.continue_()
    with pytest.raises(RuntimeError, match="Wait for completion before resetting"):
        agent.reset()

    stream.push(_done(_assistant([TextContent(text="late")])))
    await asyncio.wait_for(task, timeout=_TIMEOUT)
    assert agent.state.is_streaming is False


# ---------------------------------------------------------------------------
# steering and follow-up queues
# ---------------------------------------------------------------------------


async def test_steering_message_injected_between_turns():
    fn, calls = _stream_fn(
        [
            [_done(_assistant([_tool_call("echo", {"value": "1"})], stop_reason="toolUse"))],
            [_done(_assistant([TextContent(text="final")]))],
        ]
    )
    agent = Agent(stream_fn=fn, initial_state=AgentInitialState(tools=[_echo_tool()]))
    steered = _user("steer now")
    injected = []

    def on_event(event, signal):
        if event.type == "turn_end" and not injected:
            injected.append(True)
            agent.steer(steered)

    agent.subscribe(on_event)
    await agent.prompt("start")

    assert len(calls) == 2
    assert "steer now" in _message_texts(calls[1][1].messages)
    assert "steer now" in _message_texts(agent.state.messages)
    assert agent.has_queued_messages() is False


async def test_follow_up_message_runs_after_stop():
    fn, calls = _stream_fn(
        [
            [_done(_assistant([TextContent(text="first answer")]))],
            [_done(_assistant([TextContent(text="followed up")]))],
        ]
    )
    agent = Agent(stream_fn=fn)
    injected = []

    def on_event(event, signal):
        if event.type == "turn_end" and not injected:
            injected.append(True)
            agent.follow_up(_user("and another thing"))

    agent.subscribe(on_event)
    await agent.prompt("start")

    assert len(calls) == 2
    assert "and another thing" in _message_texts(calls[1][1].messages)
    assert "followed up" in _message_texts(agent.state.messages)


def test_queue_peek_modes_and_clearing():
    agent = Agent(stream_fn=_no_stream, initial_state=AgentInitialState())
    first, second = _user("one"), _user("two")
    agent.steer(first)
    agent.steer(second)
    assert agent.has_queued_messages() is True
    assert agent.peek_queued_messages() == [first]  # default one-at-a-time

    agent.steering_mode = "all"
    assert agent.peek_queued_messages() == [first, second]

    agent.follow_up(_user("follow"))
    agent.steering_mode = "one-at-a-time"
    assert agent.peek_queued_messages() == [first]  # steering wins over follow-ups

    agent.clear_steering_queue()
    remaining = agent.peek_queued_messages()
    assert len(remaining) == 1 and remaining[0].content == "follow"
    agent.clear_all_queues()
    assert agent.has_queued_messages() is False
    assert agent.peek_queued_messages() == []


# ---------------------------------------------------------------------------
# continue
# ---------------------------------------------------------------------------


async def test_continue_from_user_message():
    fn, calls = _stream_fn([[_done(_assistant([TextContent(text="continued")]))]])
    agent = Agent(stream_fn=fn)
    agent.state.messages = [_user("pending question")]

    await agent.continue_()
    assert len(calls) == 1
    assert _roles(agent.state.messages) == ["user", "assistant"]


async def test_continue_errors():
    agent = Agent(stream_fn=_no_stream, initial_state=AgentInitialState(system_prompt="prompt only"))
    with pytest.raises(RuntimeError, match="No messages to continue from"):
        await agent.continue_()

    agent.state.messages = [*agent.state.messages, _user("q"), _assistant([TextContent(text="a")])]
    agent.clear_all_queues()
    with pytest.raises(RuntimeError, match="Cannot continue from message role: assistant"):
        await agent.continue_()


async def test_continue_drains_queued_steering_after_assistant():
    fn, calls = _stream_fn(
        [
            [_done(_assistant([TextContent(text="first")]))],
            [_done(_assistant([TextContent(text="second")]))],
        ]
    )
    agent = Agent(stream_fn=fn)
    await agent.prompt("start")
    agent.steer(_user("queued steering"))

    await agent.continue_()
    assert len(calls) == 2
    assert "queued steering" in _message_texts(calls[1][1].messages)
    assert _roles(agent.state.messages) == ["user", "assistant", "user", "assistant"]


# ---------------------------------------------------------------------------
# abort and failure
# ---------------------------------------------------------------------------


async def test_abort_sets_signal_and_aborted_turn_records_error():
    stream = _mock_assistant_stream()
    agent = Agent(stream_fn=lambda model, context, options=None: stream)
    seen_signals = []
    agent.subscribe(lambda event, signal: seen_signals.append(signal))

    task = asyncio.create_task(agent.prompt("first"))
    await asyncio.sleep(0.05)
    agent.abort()
    assert agent.signal.aborted is True
    assert seen_signals[-1].aborted is True

    aborted = _assistant([], stop_reason="aborted")
    aborted.error_message = "aborted"
    stream.push(ErrorEvent(reason="aborted", error=aborted))
    await asyncio.wait_for(task, timeout=_TIMEOUT)
    assert agent.state.error_message == "aborted"
    assert agent.signal is None  # run finished


async def test_run_failure_becomes_error_assistant_message():
    fn, _calls = _stream_fn([[_done(_assistant([TextContent(text="never")]))]])

    def failing_convert(messages):
        raise RuntimeError("boom")

    agent = Agent(stream_fn=fn, convert_to_llm=failing_convert)
    events = []
    agent.subscribe(lambda event, signal: events.append(event.type))

    await agent.prompt("hello")  # must not raise
    state = agent.state
    last = state.messages[-1]
    assert getattr(last, "role") == "assistant"
    assert last.stop_reason == "error"
    assert last.error_message == "boom"
    assert state.error_message == "boom"
    assert state.is_streaming is False
    # The failure is surfaced as a normal (empty) turn, then the run ends.
    assert events[-4:] == ["message_start", "message_end", "turn_end", "agent_end"]


# ---------------------------------------------------------------------------
# listeners and idle
# ---------------------------------------------------------------------------


async def test_listeners_receive_events_in_order_and_unsubscribe():
    fn, _calls = _stream_fn(
        [
            [_done(_assistant([TextContent(text="one")]))],
            [_done(_assistant([TextContent(text="two")]))],
        ]
    )
    agent = Agent(stream_fn=fn)
    first_events = []
    second_events = []
    first_listener = lambda event, signal: first_events.append(event.type)
    unsubscribe = agent.subscribe(first_listener)
    agent.subscribe(first_listener)  # duplicate subscription is ignored, like pi's Set
    agent.subscribe(lambda event, signal: second_events.append(event.type))

    await agent.prompt("start")
    assert first_events == second_events
    assert first_events.count("agent_start") == 1
    assert first_events[-1] == "agent_end"

    unsubscribe()
    await agent.prompt("again")
    assert first_events.count("agent_start") == 1
    assert second_events.count("agent_start") == 2


async def test_wait_for_idle_waits_for_agent_end_listeners():
    fn, _calls = _stream_fn([[_done(_assistant([TextContent(text="hi")]))]])
    agent = Agent(stream_fn=fn)
    reached_agent_end = asyncio.Event()
    release = asyncio.Event()

    async def listener(event, signal):
        if event.type == "agent_end":
            reached_agent_end.set()
            await release.wait()

    agent.subscribe(listener)
    task = asyncio.create_task(agent.prompt("hello"))
    await asyncio.wait_for(reached_agent_end.wait(), timeout=_TIMEOUT)

    idle = asyncio.create_task(agent.wait_for_idle())
    await asyncio.sleep(0.02)
    assert not idle.done()  # listeners still running -> run not settled
    assert agent.state.is_streaming is True

    release.set()
    await asyncio.wait_for(idle, timeout=_TIMEOUT)
    await asyncio.wait_for(task, timeout=_TIMEOUT)
    assert agent.state.is_streaming is False


# ---------------------------------------------------------------------------
# reset and conversion
# ---------------------------------------------------------------------------


async def test_reset_keeps_baseline_and_clears_queues():
    fn, _calls = _stream_fn([[_done(_assistant([TextContent(text="hi")]))]])
    agent = Agent(
        stream_fn=fn,
        initial_state=AgentInitialState(system_prompt="Baseline.", tools=[_echo_tool()]),
    )
    await agent.prompt("hello")
    assert len(agent.state.messages) == 3
    agent.steer(_user("queued"))
    agent.follow_up(_user("queued follow-up"))

    agent.reset()
    assert len(agent.state.messages) == 1
    baseline = agent.state.messages[0]
    assert baseline.role == "system"
    assert baseline.content == "Baseline."
    assert [t.name for t in baseline.tools_added] == ["echo"]
    assert agent.has_queued_messages() is False
    assert agent.state.error_message is None


def test_default_convert_to_llm_keeps_system_and_drops_harness_roles():
    system = SystemMessage(content="prompt", timestamp=0)
    user = _user("hi")
    assistant = _assistant([TextContent(text="hello")])
    custom = {"role": "custom", "content": "harness only"}
    bash = {"role": "bashExecution", "content": "ls"}

    assert default_convert_to_llm([system, user, assistant, custom, bash]) == [system, user, assistant]


# ---------------------------------------------------------------------------
# turn preparation
# ---------------------------------------------------------------------------


async def test_prepare_next_turn_receives_context_and_signal():
    fn, _calls = _stream_fn(
        [
            [_done(_assistant([_tool_call("echo", {"value": "1"})], stop_reason="toolUse"))],
            [_done(_assistant([TextContent(text="final")]))],
        ]
    )
    seen = []

    def prepare_next_turn(signal):
        seen.append(signal)
        return AgentLoopTurnUpdate(model=faux_model(provider="openai", api="openai-responses"))

    agent = Agent(stream_fn=fn, initial_state=AgentInitialState(tools=[_echo_tool()]), prepare_next_turn=prepare_next_turn)
    await agent.prompt("start")

    assert len(seen) == 1
    assert seen[0] is not None and seen[0].aborted is False


async def test_prepare_next_turn_with_context_wins():
    fn, _calls = _stream_fn(
        [
            [_done(_assistant([_tool_call("echo", {"value": "1"})], stop_reason="toolUse"))],
            [_done(_assistant([TextContent(text="final")]))],
        ]
    )
    calls = []

    def legacy(signal):
        calls.append("legacy")
        return None

    def with_context(context, signal):
        calls.append("with_context")
        assert context.message.role == "assistant"
        assert len(context.tool_results) == 1
        assert context.new_messages
        assert signal is not None and signal.aborted is False
        return None

    agent = Agent(
        stream_fn=fn,
        initial_state=AgentInitialState(tools=[_echo_tool()]),
        prepare_next_turn=legacy,
        prepare_next_turn_with_context=with_context,
    )
    await agent.prompt("start")
    assert calls == ["with_context"]

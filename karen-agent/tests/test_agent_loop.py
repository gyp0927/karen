"""Agent loop tests (port of the core cases in pi's packages/agent/test/agent-loop.test.ts).

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
    StartEvent,
    SystemMessage,
    TextContent,
    TextDeltaEvent,
    ToolCall,
    UserMessage,
)
from karen_ai.providers import faux_assistant_message, faux_model
from karen_agent import (
    AfterToolCallResult,
    AgentContext,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentTool,
    AgentToolResult,
    AgentTurnDecision,
    BeforeToolCallResult,
    agent_loop,
    agent_loop_continue,
    set_default_stream_fn,
)

_DRAIN_TIMEOUT = 10


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


def _identity_convert(messages):
    return [m for m in messages if getattr(m, "role", None) in ("system", "user", "assistant", "toolResult")]


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
            # Fail fast instead of hanging the test when the loop calls more often than scripted.
            failure = _assistant([], stop_reason="error")
            failure.error_message = "test script exhausted: unexpected extra LLM call"
            stream.push(ErrorEvent(reason="error", error=failure))
            return stream
        for event in script[len(calls) - 1]:
            stream.push(event)
        return stream

    return fn, calls


async def _drain(stream, events=None):
    """Consume a stream to its result, with a timeout so a dead runner fails instead of hanging."""

    async def go():
        if events is not None:
            async for event in stream:
                events.append(event)
        return await stream.result()

    return await asyncio.wait_for(go(), timeout=_DRAIN_TIMEOUT)


def _config(**overrides):
    options = dict(model=faux_model(provider="openai", api="openai-responses"), convert_to_llm=_identity_convert)
    options.update(overrides)
    return AgentLoopConfig(**options)


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


# ---------------------------------------------------------------------------
# Basic runs
# ---------------------------------------------------------------------------


def test_emits_events_and_returns_messages():
    async def main():
        context = AgentContext(messages=[], tools=[])
        fn, calls = _stream_fn([[_done(_assistant([TextContent(text="Hi there!")]))]])
        config = _config()

        events = []
        messages = await _drain(agent_loop([_user("Hello")], context, config, None, fn), events)

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

        types = [e.type for e in events]
        for expected in ("agent_start", "turn_start", "message_start", "message_end", "turn_end", "agent_end"):
            assert expected in types
        assert len(calls) == 1

    asyncio.run(main())


def test_uses_configured_default_stream_fn_when_omitted():
    async def main():
        fn, calls = _stream_fn([[_done(_assistant([TextContent(text="fallback")]))]])
        set_default_stream_fn(fn)
        try:
            context = AgentContext(messages=[], tools=[])
            messages = await _drain(agent_loop([_user("Hello")], context, _config(), None))
            assert len(calls) == 1
            assert messages[-1].content[0].text == "fallback"
        finally:
            set_default_stream_fn(None)

    asyncio.run(main())


def test_provider_context_is_built_exclusively_from_transcript_messages():
    async def main():
        initial_system = SystemMessage(content="Transcript prompt", tools_added=[], timestamp=1)
        context = AgentContext(messages=[], tools=[])
        seen = []

        def fn(model, provider_context, options=None):
            seen.append(provider_context)
            stream = _mock_assistant_stream()
            stream.push(_done(_assistant([TextContent(text="done")])))
            return stream

        await _drain(agent_loop([initial_system, _user("Hello")], context, _config(), None, fn))

        assert type(seen[0]).__name__ == "TranscriptContext"
        assert seen[0].messages[0] is initial_system

    asyncio.run(main())


def test_custom_messages_are_filtered_by_convert_to_llm():
    async def main():
        class Notification:
            role = "notification"

            def __init__(self, text):
                self.text = text
                self.timestamp = _now()

        context = AgentContext(messages=[Notification("note")], tools=[])
        converted = []

        def convert(messages):
            nonlocal converted
            converted = [m for m in messages if getattr(m, "role", None) in ("user", "assistant", "toolResult")]
            return converted

        fn, calls = _stream_fn([[_done(_assistant([TextContent(text="Response")]))]])
        config = _config(convert_to_llm=convert)

        await _drain(agent_loop([_user("Hello")], context, config, None, fn))

        assert len(converted) == 1
        assert converted[0].role == "user"

    asyncio.run(main())


def test_transform_context_runs_before_convert():
    async def main():
        context = AgentContext(
            messages=[
                _user("old 1"),
                _assistant([TextContent(text="old r1")]),
                _user("old 2"),
                _assistant([TextContent(text="old r2")]),
            ],
            tools=[],
        )
        seen = {}

        async def transform(messages, signal=None):
            seen["transformed"] = messages[-2:]
            return seen["transformed"]

        def convert(messages):
            seen["converted"] = list(messages)
            return messages

        fn, calls = _stream_fn([[_done(_assistant([TextContent(text="Response")]))]])
        config = _config(transform_context=transform, convert_to_llm=convert)

        await _drain(agent_loop([_user("new")], context, config, None, fn))

        assert len(seen["transformed"]) == 2
        assert len(seen["converted"]) == 2

    asyncio.run(main())


def test_streaming_deltas_emit_message_update_events():
    async def main():
        message = _assistant([TextContent(text="Hi")])
        fn, calls = _stream_fn(
            [
                [
                    StartEvent(partial=_assistant([], stop_reason="pending")),
                    TextDeltaEvent(content_index=0, delta="Hi", partial=message),
                    _done(message),
                ]
            ]
        )
        context = AgentContext(messages=[], tools=[])

        events = []
        await _drain(agent_loop([_user("Hello")], context, _config(), None, fn), events)

        updates = [e for e in events if e.type == "message_update"]
        assert len(updates) == 1
        assert updates[0].assistant_message_event.delta == "Hi"

    asyncio.run(main())


def test_llm_error_ends_the_run_with_the_error_message():
    async def main():
        failure = _assistant([], stop_reason="error")
        failure.error_message = "boom"
        fn, calls = _stream_fn([[ErrorEvent(reason="error", error=failure)]])
        context = AgentContext(messages=[], tools=[])

        events = []
        messages = await _drain(agent_loop([_user("Hello")], context, _config(), None, fn), events)

        assert messages[-1].stop_reason == "error"
        assert messages[-1].error_message == "boom"
        assert events[-1].type == "agent_end"
        assert len(calls) == 1

    asyncio.run(main())


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def test_tool_call_round_trip():
    async def main():
        tool_message = _assistant([_tool_call("echo", {"value": "hello"})])
        fn, calls = _stream_fn(
            [
                [_done(tool_message)],
                [_done(_assistant([TextContent(text="done")]))],
            ]
        )
        context = AgentContext(messages=[], tools=[_echo_tool()])

        events = []
        messages = await _drain(agent_loop([_user("run echo")], context, _config(), None, fn), events)

        roles = [getattr(m, "role", None) for m in messages]
        # The context declares tools, so the loop announces the loadout with a system message.
        assert roles == ["system", "user", "assistant", "toolResult", "assistant"]
        result = messages[3]
        assert result.is_error is False
        assert result.content[0].text == "hello"
        assert result.details == {"echoed": "hello"}
        assert len(calls) == 2

        starts = [e for e in events if e.type == "tool_execution_start"]
        ends = [e for e in events if e.type == "tool_execution_end"]
        assert len(starts) == 1 and starts[0].tool_name == "echo"
        assert len(ends) == 1 and ends[0].is_error is False

    asyncio.run(main())


def test_tool_arguments_are_validated_and_coerced_before_execute():
    async def main():
        received = []

        async def execute(tool_call_id, params, signal, on_update):
            received.append(params)
            return AgentToolResult(content=[TextContent(text="ok")])

        tool = AgentTool(
            name="count",
            description="count",
            label="Count",
            parameters={"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"]},
            execute=execute,
        )
        tool_message = _assistant([_tool_call("count", {"value": "42"})])
        fn, calls = _stream_fn([[_done(tool_message)], [_done(_assistant([TextContent(text="done")]))]])
        context = AgentContext(messages=[], tools=[tool])

        messages = await _drain(agent_loop([_user("go")], context, _config(), None, fn))
        assert received == [{"value": 42}]
        assert messages[3].is_error is False

    asyncio.run(main())


def test_validation_failure_becomes_an_error_tool_result():
    async def main():
        # An object cannot coerce to a string, so validation must fail.
        tool_message = _assistant([_tool_call("echo", {"value": {"nested": True}})])
        fn, calls = _stream_fn([[_done(tool_message)], [_done(_assistant([TextContent(text="done")]))]])
        context = AgentContext(messages=[], tools=[_echo_tool()])

        messages = await _drain(agent_loop([_user("go")], context, _config(), None, fn))
        result = messages[3]
        assert result.is_error is True
        assert "Validation failed" in result.content[0].text

    asyncio.run(main())


def test_unknown_tool_becomes_an_error_tool_result():
    async def main():
        tool_message = _assistant([_tool_call("missing", {})])
        fn, calls = _stream_fn([[_done(tool_message)], [_done(_assistant([TextContent(text="done")]))]])
        context = AgentContext(messages=[], tools=[_echo_tool()])

        messages = await _drain(agent_loop([_user("go")], context, _config(), None, fn))
        assert messages[3].is_error is True
        assert "not found" in messages[3].content[0].text

    asyncio.run(main())


def test_truncated_message_tool_calls_fail_without_executing():
    async def main():
        executed = []

        async def execute(tool_call_id, params, signal, on_update):
            executed.append(params)
            return AgentToolResult(content=[TextContent(text="ok")])

        tool_message = _assistant(
            [_tool_call("echo", {"value": "x"}, id="call-1"), _tool_call("echo", {"value": "y"}, id="call-2")],
            stop_reason="length",
        )
        fn, calls = _stream_fn([[_done(tool_message)], [_done(_assistant([TextContent(text="done")]))]])
        tool = _echo_tool(execute=execute)
        context = AgentContext(messages=[], tools=[tool])

        messages = await _drain(agent_loop([_user("go")], context, _config(), None, fn))

        assert executed == []
        results = [m for m in messages if getattr(m, "role", None) == "toolResult"]
        assert len(results) == 2
        assert all(r.is_error for r in results)
        assert all("output token limit" in r.content[0].text for r in results)

    asyncio.run(main())


def test_prepare_arguments_shim_runs_before_validation():
    async def main():
        received = []

        async def execute(tool_call_id, params, signal, on_update):
            received.append(params)
            return AgentToolResult(content=[TextContent(text="ok")])

        tool = _echo_tool(
            prepare_arguments=lambda args: {"value": str(args.get("raw", ""))},
            execute=execute,
        )
        tool_message = _assistant([_tool_call("echo", {"raw": 123})])
        fn, calls = _stream_fn([[_done(tool_message)], [_done(_assistant([TextContent(text="done")]))]])
        context = AgentContext(messages=[], tools=[tool])

        messages = await _drain(agent_loop([_user("go")], context, _config(), None, fn))
        assert received == [{"value": "123"}]
        assert messages[3].is_error is False

    asyncio.run(main())


def test_before_tool_call_can_block_with_a_reason():
    async def main():
        async def before(ctx, signal=None):
            return BeforeToolCallResult(block=True, reason="nope")

        tool_message = _assistant([_tool_call("echo", {"value": "x"})])
        fn, calls = _stream_fn([[_done(tool_message)], [_done(_assistant([TextContent(text="done")]))]])
        context = AgentContext(messages=[], tools=[_echo_tool()])

        messages = await _drain(agent_loop([_user("go")], context, _config(before_tool_call=before), None, fn))
        result = messages[3]
        assert result.is_error is True
        assert result.content[0].text == "nope"

    asyncio.run(main())


def test_before_tool_call_mutation_reaches_execute_without_revalidation():
    async def main():
        received = []

        async def before(ctx, signal=None):
            ctx.args["value"] = "mutated"

        async def execute(tool_call_id, params, signal, on_update):
            received.append(dict(params))
            return AgentToolResult(content=[TextContent(text="ok")])

        tool_message = _assistant([_tool_call("echo", {"value": "original"})])
        fn, calls = _stream_fn([[_done(tool_message)], [_done(_assistant([TextContent(text="done")]))]])
        context = AgentContext(messages=[], tools=[_echo_tool(execute=execute)])

        await _drain(agent_loop([_user("go")], context, _config(before_tool_call=before), None, fn))
        assert received == [{"value": "mutated"}]

    asyncio.run(main())


def test_after_tool_call_overrides_fields():
    async def main():
        async def after(ctx, signal=None):
            return AfterToolCallResult(content=[TextContent(text="overridden")], terminate=True)

        tool_message = _assistant([_tool_call("echo", {"value": "x"})])
        fn, calls = _stream_fn([[_done(tool_message)]])
        context = AgentContext(messages=[], tools=[_echo_tool()])

        messages = await _drain(agent_loop([_user("go")], context, _config(after_tool_call=after), None, fn))
        assert messages[3].content[0].text == "overridden"
        # terminate=True on every result stops the batch — no second LLM call.
        assert len(calls) == 1

    asyncio.run(main())


def test_all_terminate_true_stops_after_the_batch():
    async def main():
        async def execute(tool_call_id, params, signal, on_update):
            return AgentToolResult(content=[TextContent(text="ok")], terminate=True)

        tool_message = _assistant(
            [_tool_call("echo", {"value": "a"}, id="call-1"), _tool_call("echo", {"value": "b"}, id="call-2")]
        )
        fn, calls = _stream_fn([[_done(tool_message)]])
        context = AgentContext(messages=[], tools=[_echo_tool(execute=execute)])

        messages = await _drain(agent_loop([_user("go")], context, _config(), None, fn))
        assert len(calls) == 1
        roles = [getattr(m, "role", None) for m in messages]
        assert roles == ["system", "user", "assistant", "toolResult", "toolResult"]

    asyncio.run(main())


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------


def test_tool_with_sequential_mode_serializes_execution():
    async def main():
        order = []

        def make_execute(name, delay):
            async def execute(tool_call_id, params, signal, on_update):
                order.append(f"{name}:start")
                await asyncio.sleep(delay)
                order.append(f"{name}:end")
                return AgentToolResult(content=[TextContent(text="ok")])

            return execute

        slow = _echo_tool(name="slow", label="Slow", execute=make_execute("slow", 0.05), execution_mode="sequential")
        fast = _echo_tool(name="fast", label="Fast", execute=make_execute("fast", 0.001))
        tool_message = _assistant(
            [_tool_call("slow", {"value": "a"}, id="call-1"), _tool_call("fast", {"value": "b"}, id="call-2")]
        )
        fn, calls = _stream_fn([[_done(tool_message)], [_done(_assistant([TextContent(text="done")]))]])
        context = AgentContext(messages=[], tools=[slow, fast])

        await _drain(agent_loop([_user("go")], context, _config(), None, fn))
        # One sequential tool forces the whole batch sequential: no interleaving.
        assert order == ["slow:start", "slow:end", "fast:start", "fast:end"]

    asyncio.run(main())


def test_parallel_emits_end_in_completion_order_but_results_in_source_order():
    async def main():
        def make_execute(name, delay):
            async def execute(tool_call_id, params, signal, on_update):
                await asyncio.sleep(delay)
                return AgentToolResult(content=[TextContent(text=name)], details={"tool": name})

            return execute

        slow = _echo_tool(name="slow", label="Slow", execute=make_execute("slow", 0.05))
        fast = _echo_tool(name="fast", label="Fast", execute=make_execute("fast", 0.001))
        tool_message = _assistant(
            [_tool_call("slow", {"value": "a"}, id="call-1"), _tool_call("fast", {"value": "b"}, id="call-2")]
        )
        fn, calls = _stream_fn([[_done(tool_message)], [_done(_assistant([TextContent(text="done")]))]])
        context = AgentContext(messages=[], tools=[slow, fast])

        events = []
        messages = await _drain(agent_loop([_user("go")], context, _config(), None, fn), events)

        completion_order = [e.tool_name for e in events if e.type == "tool_execution_end"]
        assert completion_order == ["fast", "slow"]

        results = [m for m in messages if getattr(m, "role", None) == "toolResult"]
        assert [r.tool_name for r in results] == ["slow", "fast"]

    asyncio.run(main())


# ---------------------------------------------------------------------------
# Steering / follow-up / finish_turn
# ---------------------------------------------------------------------------


def test_steering_messages_are_injected_after_tool_calls():
    async def main():
        steering = []

        async def get_steering():
            queued, steering[:] = steering[:], []
            return queued

        async def execute(tool_call_id, params, signal, on_update):
            steering.append(_user("steer"))
            return AgentToolResult(content=[TextContent(text="ok")])

        tool_message = _assistant([_tool_call("echo", {"value": "x"})])
        fn, calls = _stream_fn(
            [
                [_done(tool_message)],
                [_done(_assistant([TextContent(text="done")]))],
            ]
        )
        context = AgentContext(messages=[], tools=[_echo_tool(execute=execute)])

        messages = await _drain(
            agent_loop([_user("go")], context, _config(get_steering_messages=get_steering), None, fn)
        )

        roles = [getattr(m, "role", None) for m in messages]
        assert roles == ["system", "user", "assistant", "toolResult", "user", "assistant"]
        assert messages[4].content == "steer"
        assert len(calls) == 2

    asyncio.run(main())


def test_follow_up_messages_extend_the_run():
    async def main():
        follow_ups = [[_user("again")]]

        async def get_follow_up():
            return follow_ups.pop(0) if follow_ups else []

        fn, calls = _stream_fn(
            [
                [_done(_assistant([TextContent(text="first")]))],
                [_done(_assistant([TextContent(text="second")]))],
            ]
        )
        context = AgentContext(messages=[], tools=[])

        messages = await _drain(
            agent_loop([_user("go")], context, _config(get_follow_up_messages=get_follow_up), None, fn)
        )

        assert len(calls) == 2
        assert [getattr(m, "role", None) for m in messages] == ["user", "assistant", "user", "assistant"]
        assert messages[-1].content[0].text == "second"

    asyncio.run(main())


def test_finish_turn_end_stops_before_polling_queues():
    async def main():
        steering_calls = 0

        async def get_steering():
            nonlocal steering_calls
            steering_calls += 1
            return []

        async def finish(turn, signal=None):
            return AgentTurnDecision(action="end")

        fn, calls = _stream_fn([[_done(_assistant([TextContent(text="hi")]))]])
        context = AgentContext(messages=[], tools=[])

        events = []
        await _drain(
            agent_loop(
                [_user("go")], context, _config(get_steering_messages=get_steering, finish_turn=finish), None, fn
            ),
            events,
        )

        # Steering is polled once at loop start, never after the ended turn.
        assert steering_calls == 1
        assert events[-1].type == "agent_end"

    asyncio.run(main())


def test_prepare_next_turn_can_replace_model_and_thinking():
    async def main():
        seen = []

        def fn(model, context, options=None):
            seen.append((model.id, options.reasoning if options else None))
            label = "one" if len(seen) == 1 else "two"
            stream = _mock_assistant_stream()
            stream.push(_done(_assistant([TextContent(text=label)])))
            return stream

        other_model = faux_model(id="other", provider="openai", api="openai-responses")

        async def prepare_next(turn):
            if len(seen) == 1:
                return AgentLoopTurnUpdate(model=other_model, thinking_level="high")
            return None

        follow_ups = [[_user("second")]]

        async def get_follow_up():
            return follow_ups.pop(0) if follow_ups else []

        context = AgentContext(messages=[], tools=[])
        await _drain(
            agent_loop(
                [_user("go")],
                context,
                _config(prepare_next_turn=prepare_next, get_follow_up_messages=get_follow_up),
                None,
                fn,
            )
        )

        assert seen[0] == ("faux-1", None)
        assert seen[1] == ("other", "high")

    asyncio.run(main())


# ---------------------------------------------------------------------------
# agent_loop_continue
# ---------------------------------------------------------------------------


def test_continue_requires_messages():
    context = AgentContext(messages=[], tools=[])
    with pytest.raises(ValueError, match="no messages in context"):
        agent_loop_continue(context, _config())


def test_continue_rejects_assistant_as_last_message():
    context = AgentContext(messages=[_assistant([TextContent(text="hi")])], tools=[])
    with pytest.raises(ValueError, match="assistant"):
        agent_loop_continue(context, _config())


def test_continue_runs_without_reemitting_prompt_events():
    async def main():
        context = AgentContext(messages=[_user("retry")], tools=[])
        fn, calls = _stream_fn([[_done(_assistant([TextContent(text="ok")]))]])

        events = []
        messages = await _drain(agent_loop_continue(context, _config(), None, fn), events)

        assert len(messages) == 1
        assert messages[0].role == "assistant"
        # Only the assistant message gets message events; the pre-existing user message does not.
        user_message_events = [
            e for e in events if e.type in ("message_start", "message_end") and getattr(e.message, "role", None) == "user"
        ]
        assert user_message_events == []
        assert len(calls) == 1

    asyncio.run(main())

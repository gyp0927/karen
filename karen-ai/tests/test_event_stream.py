"""Tests for the push-based async event stream."""

import asyncio

import pytest

from karen_ai.event_stream import AssistantMessageEventStream, EventStream
from karen_ai.providers import faux_assistant_message


def test_event_stream_iterates_in_order_and_completes():
    async def main():
        stream: EventStream[int, int] = EventStream(lambda e: e == 3, lambda e: e * 10)
        stream.push(1)
        stream.push(2)
        stream.push(3)
        stream.push(4)  # ignored: stream completed at 3

        seen = [e async for e in stream]
        assert seen == [1, 2, 3]
        assert await stream.result() == 30

    asyncio.run(main())


def test_event_stream_buffers_before_consumer():
    async def main():
        stream: EventStream[str, str] = EventStream(lambda e: e == "end", lambda e: e)
        for value in ["a", "b", "end"]:
            stream.push(value)

        await asyncio.sleep(0)  # consumer starts after pushes
        assert [e async for e in stream] == ["a", "b", "end"]

    asyncio.run(main())


def test_event_stream_end_without_terminal_event():
    async def main():
        stream: EventStream[int, int] = EventStream(lambda e: False, lambda e: e)
        stream.push(1)
        stream.end(99)
        stream.push(2)  # ignored after end

        assert [e async for e in stream] == [1]
        assert await stream.result() == 99

    asyncio.run(main())


def test_assistant_message_event_stream_extracts_done_message():
    async def main():
        from karen_ai.types import DoneEvent, StartEvent

        stream = AssistantMessageEventStream()
        message = faux_assistant_message("hello")
        stream.push(StartEvent(partial=message))
        stream.push(DoneEvent(reason="stop", message=message))

        events = [e async for e in stream]
        assert [e.type for e in events] == ["start", "done"]
        assert (await stream.result()).stop_reason == "stop"

    asyncio.run(main())


def test_assistant_message_event_stream_extracts_error_message():
    async def main():
        from karen_ai.types import ErrorEvent

        stream = AssistantMessageEventStream()
        failure = faux_assistant_message([], stop_reason="error", error_message="boom")
        stream.push(ErrorEvent(reason="error", error=failure))

        assert [e.type async for e in stream] == ["error"]
        result = await stream.result()
        assert result.stop_reason == "error"
        assert result.error_message == "boom"

    asyncio.run(main())

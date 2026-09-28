"""Generic push-based async event stream, mirroring pi-ai's event-stream.ts.

Producers `push()` events and `end()` the stream; consumers iterate with
`async for` and/or `await result()` for the terminal value.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable, Generic, Optional, TypeVar

from .types import AssistantMessage, AssistantMessageEvent, DoneEvent, ErrorEvent

T = TypeVar("T")
R = TypeVar("R")

_SENTINEL: Any = object()
_UNSET: Any = object()


class EventStream(Generic[T, R]):
    def __init__(
        self,
        is_complete: Callable[[T], bool],
        extract_result: Callable[[T], R],
    ) -> None:
        self._is_complete = is_complete
        self._extract_result = extract_result
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._done = False
        self._finalized = False  # sentinel enqueued
        self._result_ready = asyncio.Event()
        self._result: Any = None

    # -- producer side -------------------------------------------------------

    def push(self, event: T) -> None:
        if self._done:
            return
        if self._is_complete(event):
            self._done = True
            self._set_result(self._extract_result(event))
        self._queue.put_nowait(event)
        if self._done:
            self._enqueue_sentinel()

    def end(self, result: Any = _UNSET) -> None:
        self._done = True
        if result is not _UNSET:
            self._set_result(result)
        self._enqueue_sentinel()

    # -- consumer side -------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[T]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[T]:
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                return
            yield item

    async def result(self) -> R:
        """Resolve with the terminal event's extracted result."""
        await self._result_ready.wait()
        return self._result  # type: ignore[return-value]

    # -- internals -----------------------------------------------------------

    def _set_result(self, result: R) -> None:
        if not self._result_ready.is_set():
            self._result = result
            self._result_ready.set()

    def _enqueue_sentinel(self) -> None:
        if not self._finalized:
            self._finalized = True
            self._queue.put_nowait(_SENTINEL)


def _is_terminal(event: AssistantMessageEvent) -> bool:
    return event.type in ("done", "error")


def _extract_message(event: AssistantMessageEvent) -> AssistantMessage:
    if isinstance(event, DoneEvent):
        return event.message
    if isinstance(event, ErrorEvent):
        return event.error
    raise ValueError("Unexpected event type for final result")


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    """The event protocol stream every chat API implementation returns.

    Successful streams emit `start` before partial updates and terminate with
    `done`. A stream may terminate directly with `error` when request setup
    fails before generation starts.
    """

    def __init__(self) -> None:
        super().__init__(_is_terminal, _extract_message)


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    """Factory function for AssistantMessageEventStream (for use in extensions)."""
    return AssistantMessageEventStream()

"""Lazy stream wiring, mirroring pi-ai's api/lazy.ts.

`lazy_stream()` returns a stream synchronously while running async setup (auth
resolution, provider dispatch) behind it. Setup failures terminate the stream
with an error event.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterable, Awaitable, Callable, Optional

from .event_stream import AssistantMessageEventStream
from .types import AssistantMessage, ErrorEvent, Model, Usage


def create_setup_error_message(model: Model, error: BaseException) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="error",
        error_message=str(error),
        timestamp=int(time.time() * 1000),
    )


async def _forward_stream(
    target: AssistantMessageEventStream,
    source: AsyncIterable[Any],
) -> None:
    async for event in source:
        target.push(event)
    result_fn = getattr(source, "result", None)
    if callable(result_fn):
        target.end(await result_fn())
    else:
        target.end()


def lazy_stream(
    model: Model,
    setup: Callable[[], Awaitable[AsyncIterable[Any]]],
) -> AssistantMessageEventStream:
    """Return a stream synchronously while running async setup behind it.

    Must be called from within a running event loop (as is every stream entry
    point in karen-ai).
    """

    outer = AssistantMessageEventStream()

    async def runner() -> None:
        try:
            inner = await setup()
            await _forward_stream(outer, inner)
        except Exception as error:
            message = create_setup_error_message(model, error)
            outer.push(ErrorEvent(reason="error", error=message))
            outer.end(message)

    asyncio.get_running_loop().create_task(runner())
    return outer


class ProviderStreams:
    """The uniform stream contract of an API implementation module.

    Every module under `karen_ai.api` provides `stream` and `stream_simple`;
    capable modules may also provide deferred-response methods. Provider
    factories pass these around as values.
    """

    def __init__(
        self,
        stream: Callable[..., AssistantMessageEventStream],
        stream_simple: Callable[..., AssistantMessageEventStream],
        fetch_deferred: Optional[Callable[..., AssistantMessageEventStream]] = None,
        cancel_deferred: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> None:
        self.stream = stream
        self.stream_simple = stream_simple
        self.fetch_deferred = fetch_deferred
        self.cancel_deferred = cancel_deferred

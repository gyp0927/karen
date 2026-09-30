"""Serializes complete read-modify-write jobs for one Session (pi's `mutation-line.ts`).

Operations run strictly in submission order. `seal()` rejects all future runs
(and runs already queued behind the current one) with the sealed error; close
paths use it to drain the line.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


async def _swallow(awaitable: Awaitable[Any]) -> None:
    try:
        await awaitable
    except Exception:
        pass


class MutationLine:
    def __init__(self) -> None:
        #: None means the line is idle (the previous tail settled).
        self._tail: Optional[asyncio.Future[None]] = None
        self._sealed_error: Optional[BaseException] = None

    def run(self, operation: Callable[[], Any]) -> "asyncio.Future[T]":
        """Queue `operation` behind the current tail; returns an awaitable of its result.

        Must be called inside a running event loop. If the line is sealed, the
        returned awaitable fails with the sealed error (pi returns a rejected
        promise rather than throwing synchronously).
        """
        loop = asyncio.get_running_loop()
        if self._sealed_error is not None:
            rejected: asyncio.Future[T] = loop.create_future()
            rejected.set_exception(self._sealed_error)
            return rejected

        tail = self._tail

        async def chained() -> T:
            if tail is not None:
                await _swallow(tail)
            if self._sealed_error is not None:
                raise self._sealed_error
            result = operation()
            if inspect.isawaitable(result):
                return await result
            return result

        task: asyncio.Task[T] = loop.create_task(chained())
        self._tail = loop.create_task(_swallow(task))
        return task

    async def seal(self, error: BaseException) -> None:
        """Reject future runs with `error`, then wait for admitted work to settle."""
        if self._sealed_error is None:
            self._sealed_error = error
        if self._tail is not None:
            await self._tail

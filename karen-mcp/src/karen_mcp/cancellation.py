"""The cancellation contract every `signal=` in this package accepts.

pi leans on the platform's `AbortController`/`AbortSignal`; Python has neither,
so the package ships the small shape it needs — and karen-ai's `AbortSignal`
already satisfies it, so an application can pass the same signal it uses to
stop a conversation.
"""

from __future__ import annotations

import asyncio
from typing import Any

__all__ = ["Signal"]


class Signal:
    """An abort flag with a waitable side, like `AbortSignal`.

    Aborting twice keeps the first reason: a signal is a latch, not a channel.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: Any = None

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> Any:
        return self._reason

    async def wait(self) -> None:
        await self._event.wait()

    def abort(self, reason: Any = None) -> None:
        if self._event.is_set():
            return
        self._reason = reason
        self._event.set()

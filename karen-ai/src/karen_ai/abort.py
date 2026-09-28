"""Cooperative cancellation, mirroring the AbortSignal/AbortController pattern.

Python has no built-in equivalent of the web's AbortSignal, so karen-ai
provides a minimal one. Provider adapters check `signal.aborted` /
`signal.throw_if_aborted()` at suspension points and `await signal.wait()`
when sleeping.
"""

from __future__ import annotations

import asyncio

from .errors import AbortError


class AbortSignal:
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: BaseException | None = None

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> BaseException:
        return self._reason if self._reason is not None else AbortError()

    def throw_if_aborted(self) -> None:
        if self.aborted:
            raise self.reason

    async def wait(self) -> None:
        await self._event.wait()

    def _abort(self, reason: BaseException | None = None) -> None:
        if not self._event.is_set():
            self._reason = reason
            self._event.set()


class AbortController:
    def __init__(self) -> None:
        self.signal = AbortSignal()

    def abort(self, reason: BaseException | None = None) -> None:
        self.signal._abort(reason)


def operation_signal(signal: AbortSignal | None) -> AbortSignal:
    """Always return a signal, so internal code never has to None-check."""
    return signal if signal is not None else AbortSignal()


async def abortable_sleep(delay: float, signal: AbortSignal | None = None) -> None:
    """Sleep `delay` seconds, raising AbortError early if the signal fires."""
    if signal is None:
        await asyncio.sleep(delay)
        return
    signal.throw_if_aborted()
    try:
        await asyncio.wait_for(signal.wait(), timeout=max(0.0, delay))
    except asyncio.TimeoutError:
        return
    raise signal.reason

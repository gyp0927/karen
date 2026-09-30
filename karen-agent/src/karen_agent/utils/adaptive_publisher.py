"""Rate-limited state publisher (pi's `harness/utils/adaptive-publisher.ts`).

Publishes the latest state without queuing intermediate mutations. The first
dirty state after idle is immediate. Each publication then buys a delay
proportional to its encoded size, with a minimum interval that also bounds
event count. A single trailing timer guarantees eventual publication.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Generic, Optional, TypeVar

TValue = TypeVar("TValue")
TUpdate = TypeVar("TUpdate")


def _now_ms() -> float:
    return time.monotonic() * 1000


class AdaptivePublisher(Generic[TValue, TUpdate]):
    def __init__(
        self,
        *,
        snapshot: Callable[[], TValue],
        update: Callable[[Optional[TValue], TValue], Optional[TUpdate]],
        measure: Callable[[TUpdate], int],
        publish: Callable[[TUpdate], None],
        on_error: Optional[Callable[[BaseException], None]] = None,
        min_interval_ms: float = 100,
        target_bytes_per_second: float = 100 * 1024,
    ) -> None:
        self._snapshot = snapshot
        self._update = update
        self._measure = measure
        self._publish = publish
        self._on_error = on_error
        self._min_interval_ms = min_interval_ms
        self._target_bytes_per_second = target_bytes_per_second
        self._published: Optional[TValue] = None
        self._dirty = False
        self._next_emit_at = 0.0
        self._timer: Optional[asyncio.TimerHandle] = None
        self._disposed = False

    def mark_dirty(self) -> None:
        if self._disposed:
            return
        self._dirty = True
        wait = self._next_emit_at - _now_ms()
        if wait <= 0:
            self.flush()
            return
        self._arm_timer(wait)

    def flush(self, force: bool = False) -> None:
        if self._disposed or not self._dirty:
            return
        now = _now_ms()
        if not force and now < self._next_emit_at:
            self._arm_timer(self._next_emit_at - now)
            return
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        current = self._snapshot()
        update = self._update(self._published, current)
        if update is None:
            self._published = current
            self._dirty = False
            return
        encoded_bytes = self._measure(update)
        self._published = current
        self._dirty = False
        self._next_emit_at = now + max(self._min_interval_ms, (encoded_bytes * 1000) / self._target_bytes_per_second)
        # Commit before delivery. A consumer may apply the update and then throw
        # or reenter the producer; retaining the old baseline would duplicate that delta.
        self._publish(update)

    def dispose(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._disposed = True

    def _arm_timer(self, wait_ms: float) -> None:
        if self._timer is not None:
            return

        def fire() -> None:
            self._timer = None
            try:
                self.flush()
            except Exception as error:  # noqa: BLE001 — mirrors pi's onError channel
                if self._on_error is not None:
                    self._on_error(error)

        self._timer = asyncio.get_running_loop().call_later(max(0.0, wait_ms / 1000), fire)


__all__ = ["AdaptivePublisher"]

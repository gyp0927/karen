"""Per-path serialization of file mutations (pi's `tools/file-mutation-queue.ts`).

pi keys its queues per ExecutionEnv on the canonical path; karen has no env
capability, so the registry is process-global (canonical path keys). Lock
entries are reference-counted and removed once idle.
"""

from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable, Dict, TypeVar

T = TypeVar("T")


class _QueueEntry:
    __slots__ = ("lock", "waiters")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.waiters = 0


_QUEUES: Dict[str, _QueueEntry] = {}
_GUARD = asyncio.Lock()


def _mutation_queue_key(path: str) -> str:
    absolute = os.path.abspath(path)
    try:
        return os.path.normcase(os.path.realpath(absolute))
    except OSError:
        # pi falls back to the absolute path on not_found/not_supported.
        return os.path.normcase(absolute)


async def with_file_mutation_queue(path: str, fn: Callable[[], Awaitable[T]]) -> T:
    """Run `fn` serialized against other mutations targeting the same path."""
    key = _mutation_queue_key(path)
    async with _GUARD:
        entry = _QUEUES.get(key)
        if entry is None:
            entry = _QUEUES[key] = _QueueEntry()
        entry.waiters += 1
    try:
        async with entry.lock:
            return await fn()
    finally:
        async with _GUARD:
            entry.waiters -= 1
            if entry.waiters == 0:
                _QUEUES.pop(key, None)


__all__ = ["with_file_mutation_queue"]

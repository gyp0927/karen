"""Session-scoped resource registry, mirroring pi-ai's `session-resources.ts`.

Transports that cache process-wide resources per session (the Codex WebSocket
transport) register a cleanup here so applications can drop a session's
resources on demand.
"""

from __future__ import annotations

import inspect
from typing import Awaitable, Callable, List, Optional, Set, Union

SessionResourceCleanup = Callable[[Optional[str]], Union[None, Awaitable[None]]]

_cleanups: Set[SessionResourceCleanup] = set()


def register_session_resource_cleanup(cleanup: SessionResourceCleanup) -> Callable[[], None]:
    """Registers a cleanup; returns a function that unregisters it."""
    _cleanups.add(cleanup)

    def unregister() -> None:
        _cleanups.discard(cleanup)

    return unregister


async def cleanup_session_resources(session_id: Optional[str] = None) -> None:
    """Runs every registered cleanup; failures are aggregated into one error."""
    errors: List[BaseException] = []
    for cleanup in list(_cleanups):
        try:
            result = cleanup(session_id)
            if inspect.isawaitable(result):
                await result
        except BaseException as error:  # noqa: BLE001 - aggregated and re-raised below
            errors.append(error)
    if errors:
        raise ExceptionGroup("Failed to cleanup session resources", errors)


__all__ = ["SessionResourceCleanup", "cleanup_session_resources", "register_session_resource_cleanup"]

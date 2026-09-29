"""Shared no-op telemetry context used when an application provides none."""

from __future__ import annotations

import inspect
from typing import Any, Optional

from .types import SpanAttributes, SpanCallback, SpanOptions, SpanStatus, T


class _NoopTelemetrySpan:
    """Inert span: records nothing, still runs the callback exactly once."""

    __slots__ = ()

    async def start_span(self, options: SpanOptions, callback: SpanCallback[T]) -> T:
        return await _admit(callback)

    def add_event(self, name: str, attributes: Optional[SpanAttributes] = None) -> None:
        return None

    def set_attributes(self, attributes: SpanAttributes) -> None:
        return None

    def set_status(self, status: SpanStatus) -> None:
        return None


async def _admit(callback: SpanCallback[T]) -> T:
    result = callback(_NOOP_TELEMETRY_SPAN)
    if inspect.isawaitable(result):
        return await result
    return result


_NOOP_TELEMETRY_SPAN = _NoopTelemetrySpan()


class _NoopTelemetryContext:
    """Runs work without recording anything; never inspects span payloads."""

    __slots__ = ()

    async def start_span(self, options: SpanOptions, callback: SpanCallback[T]) -> T:
        return await _admit(callback)


NOOP_TELEMETRY_CONTEXT: Any = _NoopTelemetryContext()
"""Shared telemetry context used when an application does not provide one."""

__all__ = ["NOOP_TELEMETRY_CONTEXT"]

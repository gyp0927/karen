"""Backend-neutral reference implementation that records spans in process memory.

Create a fresh instance to isolate tests or independent recording scopes.
Recording is passive: malformed or unreadable telemetry payloads are ignored
rather than propagated, and calls on a settled span are inert.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence as _Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .noop import NOOP_TELEMETRY_CONTEXT
from .types import SpanAttributes, SpanCallback, SpanError, SpanOptions, SpanStatus, T


@dataclass(frozen=True)
class RecordedTelemetryEvent:
    """A detached snapshot of one recorded span event."""

    name: str
    attributes: Dict[str, Any]


@dataclass(frozen=True)
class RecordedTelemetrySpan:
    """A detached snapshot of one recorded span."""

    id: int
    parent_id: Optional[int]
    name: str
    attributes: Dict[str, Any]
    events: List[RecordedTelemetryEvent]
    status: SpanStatus
    settled: bool
    end_sequence: Optional[int] = None


@dataclass
class _SpanRecord:
    id: int
    parent_id: Optional[int]
    name: str
    attributes: Dict[str, Any]
    events: List[RecordedTelemetryEvent]
    status: SpanStatus
    explicit_status: bool
    settled: bool
    end_sequence: Optional[int] = None


@dataclass
class _RecorderState:
    spans: List[_SpanRecord] = field(default_factory=list)
    next_span_id: int = 1
    next_end_sequence: int = 1


def _copy_attribute_value(value: Any) -> Any:
    if isinstance(value, (str, bytes)) or not isinstance(value, _Sequence):
        return value
    # Hostile array-likes raise here; the caller treats that as an unreadable payload.
    return list(value)


def _copy_attributes(attributes: Optional[SpanAttributes]) -> Dict[str, Any]:
    copy: Dict[str, Any] = {}
    if attributes is None:
        return copy
    for name, value in attributes.items():  # type: ignore[union-attr]
        if value is not None:
            copy[name] = _copy_attribute_value(value)
    return copy


def _merge_attributes(current: Dict[str, Any], attributes: Optional[SpanAttributes]) -> Dict[str, Any]:
    merged = _copy_attributes(current)
    for name, value in _copy_attributes(attributes).items():
        merged[name] = value
    return merged


def _copy_status(status: SpanStatus) -> SpanStatus:
    if status.status == "ok" or status.error is None:
        return SpanStatus(status="ok" if status.status == "ok" else "error")
    return SpanStatus(status="error", error=SpanError(name=status.error.name, message=status.error.message))


def _automatic_error_status(error: BaseException) -> SpanStatus:
    try:
        if isinstance(error, BaseException):
            return SpanStatus(status="error", error=SpanError(name=type(error).__name__, message=str(error)))
    except Exception:  # pragma: no cover - error inspection is passive
        pass
    return SpanStatus(status="error")


def _settle_span(state: _RecorderState, span: _SpanRecord, failed: bool, error: Optional[BaseException] = None) -> None:
    if span.settled:
        return
    if failed and not span.explicit_status:
        span.status = _automatic_error_status(error) if error is not None else SpanStatus(status="error")
    span.settled = True
    span.end_sequence = state.next_end_sequence
    state.next_end_sequence += 1


def _create_span(state: _RecorderState, parent: Optional[_SpanRecord], options: SpanOptions) -> _SpanRecord:
    name = getattr(options, "name")
    attributes = _copy_attributes(getattr(options, "attributes", None))
    span = _SpanRecord(
        id=state.next_span_id,
        parent_id=parent.id if parent is not None else None,
        name=name,
        attributes=attributes,
        events=[],
        status=SpanStatus(status="ok"),
        explicit_status=False,
        settled=False,
    )
    state.next_span_id += 1
    return span


class _LiveTelemetrySpan:
    """Recording span handed to the callback; inert once its span settles."""

    __slots__ = ("_state", "_record")

    def __init__(self, state: _RecorderState, record: _SpanRecord) -> None:
        self._state = state
        self._record = record

    async def start_span(self, options: SpanOptions, callback: SpanCallback[T]) -> T:
        return await _start_in_memory_span(self._state, self._record, options, callback)

    def add_event(self, name: str, attributes: Optional[SpanAttributes] = None) -> None:
        if self._record.settled:
            return
        try:
            event = RecordedTelemetryEvent(name=name, attributes=_copy_attributes(attributes))
        except Exception:
            return
        self._record.events.append(event)

    def set_attributes(self, attributes: SpanAttributes) -> None:
        if self._record.settled:
            return
        try:
            merged = _merge_attributes(self._record.attributes, attributes)
        except Exception:
            return
        self._record.attributes = merged

    def set_status(self, status: SpanStatus) -> None:
        if self._record.settled:
            return
        try:
            copied = _copy_status(status)
        except Exception:
            return
        self._record.status = copied
        self._record.explicit_status = True


async def _start_in_memory_span(
    state: _RecorderState,
    parent: Optional[_SpanRecord],
    options: SpanOptions,
    callback: SpanCallback[T],
) -> T:
    if parent is not None and parent.settled:
        return await NOOP_TELEMETRY_CONTEXT.start_span(options, callback)

    try:
        record = _create_span(state, parent, options)
        state.spans.append(record)
    except Exception:
        return await NOOP_TELEMETRY_CONTEXT.start_span(options, callback)

    span = _LiveTelemetrySpan(state, record)
    try:
        result = callback(span)
        if inspect.isawaitable(result):
            result = await result
    except BaseException as error:
        _settle_span(state, record, True, error)
        raise
    _settle_span(state, record, False)
    return result


class InMemoryTelemetryContext:
    """Records spans in process memory; `get_spans()` returns detached snapshots."""

    def __init__(self) -> None:
        self._state = _RecorderState()

    async def start_span(self, options: SpanOptions, callback: SpanCallback[T]) -> T:
        return await _start_in_memory_span(self._state, None, options, callback)

    def get_spans(self) -> List[RecordedTelemetrySpan]:
        """Returns detached snapshots in span-start order."""
        return [
            RecordedTelemetrySpan(
                id=span.id,
                parent_id=span.parent_id,
                name=span.name,
                attributes=_copy_attributes(span.attributes),
                events=[
                    RecordedTelemetryEvent(name=event.name, attributes=dict(event.attributes))
                    for event in span.events
                ],
                status=_copy_status(span.status),
                settled=span.settled,
                end_sequence=span.end_sequence,
            )
            for span in self._state.spans
        ]


__all__ = ["InMemoryTelemetryContext", "RecordedTelemetryEvent", "RecordedTelemetrySpan"]

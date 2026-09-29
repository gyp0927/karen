"""Telemetry contracts: spans, attributes, statuses, and the context protocol.

Mirrors the runtime (non-type-level) part of pi-telemetry's `index.ts`. The
TypeScript package's schema/typing machinery has no Python runtime equivalent;
`karen_ai.telemetry.schema` keeps the same call shapes for the bits that do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol, Sequence, TypeVar, Union, runtime_checkable

T = TypeVar("T")

#: Scalar or homogeneous list attribute payloads.
AttributeValue = Union[str, int, float, bool, Sequence[str], Sequence[int], Sequence[float], Sequence[bool]]

#: Span/event attributes. `None` values are ignored (never recorded).
SpanAttributes = Mapping[str, Optional[AttributeValue]]


@dataclass(frozen=True)
class SpanError:
    """Error detail attached to an error status."""

    name: str
    message: str


@dataclass(frozen=True)
class SpanStatus:
    """Outcome of a span: `ok`, or `error` with optional error detail."""

    status: str = "ok"
    error: Optional[SpanError] = None

    @classmethod
    def ok(cls) -> "SpanStatus":
        return cls(status="ok")

    @classmethod
    def failed(cls, name: Optional[str] = None, message: Optional[str] = None) -> "SpanStatus":
        """Builds an `error` status; detail is optional."""
        error = SpanError(name=name or "Error", message=message or "") if name or message else None
        return cls(status="error", error=error)

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class SpanOptions:
    """Start options for a new span."""

    name: str
    attributes: Optional[SpanAttributes] = None


SpanCallback = Callable[["TelemetrySpan"], Union[T, Awaitable[T]]]


@runtime_checkable
class TelemetrySpan(Protocol):
    """A span; also a context, so starting a span from it makes a child."""

    async def start_span(self, options: SpanOptions, callback: SpanCallback[T]) -> T: ...

    def add_event(self, name: str, attributes: Optional[SpanAttributes] = None) -> None: ...

    def set_attributes(self, attributes: SpanAttributes) -> None: ...

    def set_status(self, status: SpanStatus) -> None: ...


@runtime_checkable
class TelemetryContext(Protocol):
    """Explicit parent context for telemetry produced by one logical operation."""

    async def start_span(self, options: SpanOptions, callback: SpanCallback[T]) -> T: ...


__all__ = [
    "AttributeValue",
    "SpanAttributes",
    "SpanCallback",
    "SpanError",
    "SpanOptions",
    "SpanStatus",
    "TelemetryContext",
    "TelemetrySpan",
]

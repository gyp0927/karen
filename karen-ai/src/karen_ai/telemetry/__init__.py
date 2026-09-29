"""Vendor-neutral telemetry contracts, mirroring `@earendil-works/pi-telemetry`.

pi packages pass an explicit `TelemetryContext` through provider request options;
applications supply an implementation (or the shared no-op default). There is no
exporter, no global current-span state, and no backend dependency: spans are
started by the caller-provided context and telemetry never interferes with
provider behaviour.
"""

from .memory import InMemoryTelemetryContext, RecordedTelemetryEvent, RecordedTelemetrySpan
from .noop import NOOP_TELEMETRY_CONTEXT
from .schema import create_typed_span_starter, define_telemetry_schema
from .types import (
    AttributeValue,
    SpanAttributes,
    SpanError,
    SpanOptions,
    SpanStatus,
    TelemetryContext,
    TelemetrySpan,
)

__all__ = [
    "NOOP_TELEMETRY_CONTEXT",
    "AttributeValue",
    "InMemoryTelemetryContext",
    "RecordedTelemetryEvent",
    "RecordedTelemetrySpan",
    "SpanAttributes",
    "SpanError",
    "SpanOptions",
    "SpanStatus",
    "TelemetryContext",
    "TelemetrySpan",
    "create_typed_span_starter",
    "define_telemetry_schema",
]

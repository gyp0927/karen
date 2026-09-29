"""Serializable schema helpers, mirroring pi-telemetry's typed starter surface.

In pi-telemetry the schema values exist only for TypeScript inference: nothing
is validated at runtime. Python keeps the same call shapes so ported call sites
read identically — `define_telemetry_schema` is an identity helper and
`create_typed_span_starter` binds a parent context plus a span name vocabulary.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping, Optional, TypeVar, Union

from .types import SpanAttributes, SpanOptions, TelemetryContext, TelemetrySpan

T = TypeVar("T")

#: A span name and its (documentation-only) attribute metadata.
TelemetrySpanDefinition = Mapping[str, Any]
#: `{"version": int, "spans": {name: definition}}`.
TelemetrySchema = Mapping[str, Any]

TypedSpanCallback = Callable[["TypedSpanStarter", TelemetrySpan], Union[T, Awaitable[T]]]


def define_telemetry_schema(schema: TelemetrySchema) -> TelemetrySchema:
    """Typed identity helper for serializable telemetry schema data."""
    return schema


class TypedSpanStarter:
    """A span starter bound to one explicit parent context and schema vocabulary."""

    __slots__ = ("_context", "_schemas")

    def __init__(self, context: TelemetryContext, schemas: Iterable[TelemetrySchema] = ()) -> None:
        self._context = context
        self._schemas = tuple(schemas)

    @property
    def schemas(self) -> tuple[TelemetrySchema, ...]:
        """The schemas bound to this starter (used for inference upstream)."""
        return self._schemas

    async def start(
        self,
        name: str,
        attributes: Optional[SpanAttributes],
        callback: TypedSpanCallback[T],
    ) -> T:
        """Starts `name` under this starter's context, passing child nesting to the callback."""
        return await self._context.start_span(
            SpanOptions(name=name, attributes=attributes),
            lambda span: callback(TypedSpanStarter(span, self._schemas), span),
        )

    def __call__(
        self,
        name: str,
        attributes: Optional[SpanAttributes],
        callback: TypedSpanCallback[T],
    ) -> Any:
        return self.start(name, attributes, callback)


def create_typed_span_starter(
    telemetry_context: TelemetryContext,
    schemas: Iterable[TelemetrySchema] = (),
) -> TypedSpanStarter:
    """Binds an explicit parent context to the combined span vocabulary of the schemas."""
    return TypedSpanStarter(telemetry_context, schemas)


__all__ = [
    "TelemetrySchema",
    "TelemetrySpanDefinition",
    "TypedSpanStarter",
    "create_typed_span_starter",
    "define_telemetry_schema",
]

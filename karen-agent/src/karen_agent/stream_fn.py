"""Default stream function registry, mirroring pi's `stream-fn.ts`.

Hosts that provide a default model runtime install its stream function here so
callers of `agent_loop()` / `Agent` can omit `stream_fn`. `models_stream_fn()`
adapts a karen-ai `Models` registry to the loop's StreamFn shape — the standard
way to get auth resolution and telemetry for free.
"""

from __future__ import annotations

from typing import Optional

from karen_ai import Context, Models, TranscriptContext

from .types import StreamFn

_default_stream_fn: Optional[StreamFn] = None


def set_default_stream_fn(stream_fn: Optional[StreamFn]) -> None:
    global _default_stream_fn
    _default_stream_fn = stream_fn


def get_default_stream_fn() -> StreamFn:
    if _default_stream_fn is None:
        raise RuntimeError(
            "No default stream function configured. Pass stream_fn explicitly or call set_default_stream_fn()."
        )
    return _default_stream_fn


def models_stream_fn(models: Models) -> StreamFn:
    """Adapt a karen-ai `Models` registry to the agent loop's StreamFn.

    The loop passes a normalized TranscriptContext; Models.stream_simple takes a
    raw Context, so the messages pass through one more (no-op) normalization.
    """

    def stream_fn(model, context: TranscriptContext, options=None):
        return models.stream_simple(model, Context(messages=list(context.messages)), options)

    return stream_fn


__all__ = ["get_default_stream_fn", "models_stream_fn", "set_default_stream_fn"]

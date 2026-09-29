"""OpenCode's per-conversation routing header, mirroring opencode-headers.ts."""

from __future__ import annotations

from typing import Optional, TypeVar

from ..lazy import ProviderStreams
from ..types import ProviderHeaders, StreamOptions

OPENCODE_SESSION_HEADER = "x-opencode-session"

_Options = TypeVar("_Options", bound=StreamOptions)


def _has_header(headers: Optional[ProviderHeaders], name: str) -> bool:
    expected = name.lower()
    return any(key.lower() == expected for key in (headers or {}))


def _with_session_header(options: Optional[_Options]) -> Optional[_Options]:
    if options is None or not options.session_id or _has_header(options.headers, OPENCODE_SESSION_HEADER):
        return options
    return options.model_copy(update={"headers": {**(options.headers or {}), OPENCODE_SESSION_HEADER: options.session_id}})


def with_opencode_session_header(streams: ProviderStreams) -> ProviderStreams:
    """Adds OpenCode's required per-conversation routing header before API dispatch."""

    def stream(model, context, options):
        return streams.stream(model, context, _with_session_header(options))

    def stream_simple(model, context, options):
        return streams.stream_simple(model, context, _with_session_header(options))

    return ProviderStreams(
        stream=stream,
        stream_simple=stream_simple,
        fetch_deferred=streams.fetch_deferred,
        cancel_deferred=streams.cancel_deferred,
    )

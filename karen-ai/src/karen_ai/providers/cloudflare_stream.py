"""Cloudflare endpoint placeholder resolution, mirroring providers/cloudflare-stream.ts.

Catalog base URLs carry `{CLOUDFLARE_ACCOUNT_ID}` / `{CLOUDFLARE_GATEWAY_ID}`
placeholders; the wrappers materialize them from the resolved provider env
before dispatch.
"""

from __future__ import annotations

from typing import Optional, TypeVar

from ..lazy import ProviderStreams
from ..types import ProviderEnv

CLOUDFLARE_ACCOUNT_ID = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_GATEWAY_ID = "CLOUDFLARE_GATEWAY_ID"

_Model = TypeVar("_Model")


def resolve_cloudflare_model(model: _Model, env: Optional[ProviderEnv]) -> _Model:
    if not env:
        return model
    base_url = model.base_url  # type: ignore[attr-defined]
    resolved = base_url.replace(
        f"{{{CLOUDFLARE_ACCOUNT_ID}}}", env.get(CLOUDFLARE_ACCOUNT_ID, f"{{{CLOUDFLARE_ACCOUNT_ID}}}")
    ).replace(
        f"{{{CLOUDFLARE_GATEWAY_ID}}}", env.get(CLOUDFLARE_GATEWAY_ID, f"{{{CLOUDFLARE_GATEWAY_ID}}}")
    )
    if resolved == base_url:
        return model
    return model.model_copy(update={"base_url": resolved})  # type: ignore[attr-defined,return-value]


def cloudflare_streams(streams: ProviderStreams) -> ProviderStreams:
    """Wrap an API implementation so endpoint placeholders resolve from options.env."""

    def stream(model, context, options):
        return streams.stream(resolve_cloudflare_model(model, options.env if options else None), context, options)

    def stream_simple(model, context, options):
        return streams.stream_simple(
            resolve_cloudflare_model(model, options.env if options else None), context, options
        )

    return ProviderStreams(
        stream=stream,
        stream_simple=stream_simple,
        fetch_deferred=streams.fetch_deferred,
        cancel_deferred=streams.cancel_deferred,
    )


def cloudflare_classifier(classify):
    """Classifier counterpart of `cloudflare_streams` (classify callables)."""

    async def wrapped(model, context, options):
        return await classify(resolve_cloudflare_model(model, options.env if options else None), context, options)

    return wrapped

"""Meta provider (OpenAI responses + Muse subscription OAuth), mirroring meta.ts."""

from __future__ import annotations

from ..api import openai_responses_api
from ._catalog import catalog_provider


def meta_provider():
    from ..auth.oauth import meta_oauth

    return catalog_provider(
        id="meta",
        name="Meta",
        base_url="https://api.meta.ai/v1",
        key_name="Meta Model API key",
        env_vars=["META_API_KEY"],
        oauth=meta_oauth,
        api=openai_responses_api(),
    )

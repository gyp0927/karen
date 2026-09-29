"""xAI provider (OpenAI responses + SuperGrok/X subscription OAuth), mirroring xai.ts."""

from __future__ import annotations

from ..api import openai_responses_api
from ._catalog import catalog_provider


def xai_provider():
    from ..auth.oauth import xai_oauth

    return catalog_provider(
        id="xai",
        name="xAI",
        base_url="https://api.x.ai/v1",
        key_name="xAI API key",
        env_vars=["XAI_API_KEY"],
        oauth=xai_oauth,
        api=openai_responses_api(),
    )

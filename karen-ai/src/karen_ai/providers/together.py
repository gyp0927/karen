"""Together provider, mirroring pi-ai's providers/together.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def together_provider():
    return catalog_provider(
        id="together",
        name="Together",
        base_url="https://api.together.ai/v1",
        key_name="Together API key",
        env_vars=['TOGETHER_API_KEY'],
        api=openai_completions_api(),
    )

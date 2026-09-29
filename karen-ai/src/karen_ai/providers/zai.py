"""Z.AI provider, mirroring pi-ai's providers/zai.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def zai_provider():
    return catalog_provider(
        id="zai",
        name="Z.AI",
        base_url="https://api.z.ai/api/coding/paas/v4",
        key_name="Z.AI API key",
        env_vars=['ZAI_API_KEY'],
        api=openai_completions_api(),
    )

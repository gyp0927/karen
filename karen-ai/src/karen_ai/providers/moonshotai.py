"""Moonshot AI provider, mirroring pi-ai's providers/moonshotai.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def moonshotai_provider():
    return catalog_provider(
        id="moonshotai",
        name="Moonshot AI",
        base_url="https://api.moonshot.ai/v1",
        key_name="Moonshot AI API key",
        env_vars=['MOONSHOT_API_KEY'],
        api=openai_completions_api(),
    )

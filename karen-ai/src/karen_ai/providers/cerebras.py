"""Cerebras provider, mirroring pi-ai's providers/cerebras.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def cerebras_provider():
    return catalog_provider(
        id="cerebras",
        name="Cerebras",
        base_url="https://api.cerebras.ai/v1",
        key_name="Cerebras API key",
        env_vars=['CEREBRAS_API_KEY'],
        api=openai_completions_api(),
    )

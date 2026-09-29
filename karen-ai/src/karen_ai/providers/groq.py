"""Groq provider, mirroring pi-ai's providers/groq.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def groq_provider():
    return catalog_provider(
        id="groq",
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        key_name="Groq API key",
        env_vars=['GROQ_API_KEY'],
        api=openai_completions_api(),
    )

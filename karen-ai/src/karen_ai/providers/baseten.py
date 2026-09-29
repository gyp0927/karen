"""Baseten provider, mirroring pi-ai's providers/baseten.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def baseten_provider():
    return catalog_provider(
        id="baseten",
        name="Baseten",
        base_url="https://inference.baseten.co/v1",
        key_name="Baseten API key",
        env_vars=['BASETEN_API_KEY'],
        api=openai_completions_api(),
    )

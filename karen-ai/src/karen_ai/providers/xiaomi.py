"""Xiaomi provider, mirroring pi-ai's providers/xiaomi.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def xiaomi_provider():
    return catalog_provider(
        id="xiaomi",
        name="Xiaomi",
        base_url="https://api.xiaomimimo.com/v1",
        key_name="Xiaomi API key",
        env_vars=['XIAOMI_API_KEY'],
        api=openai_completions_api(),
    )

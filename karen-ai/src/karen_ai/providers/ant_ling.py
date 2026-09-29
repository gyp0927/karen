"""Ant Ling provider, mirroring pi-ai's providers/ant-ling.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def ant_ling_provider():
    return catalog_provider(
        id="ant-ling",
        name="Ant Ling",
        base_url="https://api.ant-ling.com/v1",
        key_name="Ant Ling API key",
        env_vars=['ANT_LING_API_KEY'],
        api=openai_completions_api(),
    )

"""Moonshot AI CN provider, mirroring pi-ai's providers/moonshotai-cn.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def moonshotai_cn_provider():
    return catalog_provider(
        id="moonshotai-cn",
        name="Moonshot AI CN",
        base_url="https://api.moonshot.cn/v1",
        key_name="Moonshot AI API key",
        env_vars=['MOONSHOT_API_KEY'],
        api=openai_completions_api(),
    )

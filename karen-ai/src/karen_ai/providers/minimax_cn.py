"""MiniMax CN provider, mirroring pi-ai's providers/minimax-cn.ts."""

from __future__ import annotations

from ..api import anthropic_messages_api
from ._catalog import catalog_provider


def minimax_cn_provider():
    return catalog_provider(
        id="minimax-cn",
        name="MiniMax CN",
        base_url="https://api.minimaxi.com/anthropic",
        key_name="MiniMax CN API key",
        env_vars=['MINIMAX_CN_API_KEY'],
        api=anthropic_messages_api(),
    )

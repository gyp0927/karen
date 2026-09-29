"""MiniMax provider, mirroring pi-ai's providers/minimax.ts."""

from __future__ import annotations

from ..api import anthropic_messages_api
from ._catalog import catalog_provider


def minimax_provider():
    return catalog_provider(
        id="minimax",
        name="MiniMax",
        base_url="https://api.minimax.io/anthropic",
        key_name="MiniMax API key",
        env_vars=['MINIMAX_API_KEY'],
        api=anthropic_messages_api(),
    )

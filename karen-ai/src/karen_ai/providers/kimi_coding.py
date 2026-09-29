"""Kimi For Coding provider (Anthropic messages + subscription OAuth), mirroring kimi-coding.ts."""

from __future__ import annotations

from ..api import anthropic_messages_api
from ._catalog import catalog_provider


def kimi_coding_provider():
    from ..auth.oauth import kimi_coding_oauth

    return catalog_provider(
        id="kimi-coding",
        name="Kimi For Coding",
        base_url="https://api.kimi.com/coding",
        key_name="Kimi API key",
        env_vars=["KIMI_API_KEY"],
        oauth=kimi_coding_oauth,
        api=anthropic_messages_api(),
    )

"""Fireworks provider (Anthropic messages + OpenAI completions), mirroring fireworks.ts."""

from __future__ import annotations

from ..api import anthropic_messages_api, openai_completions_api
from ._catalog import catalog_provider


def fireworks_provider():
    return catalog_provider(
        id="fireworks",
        name="Fireworks",
        base_url="https://api.fireworks.ai/inference",
        key_name="Fireworks API key",
        env_vars=["FIREWORKS_API_KEY"],
        api={
            "anthropic-messages": anthropic_messages_api(),
            "openai-completions": openai_completions_api(),
        },
    )

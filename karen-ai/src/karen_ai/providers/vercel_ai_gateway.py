"""Vercel AI Gateway provider, mirroring pi-ai's providers/vercel-ai-gateway.ts."""

from __future__ import annotations

from ..api import anthropic_messages_api
from ._catalog import catalog_provider


def vercel_ai_gateway_provider():
    return catalog_provider(
        id="vercel-ai-gateway",
        name="Vercel AI Gateway",
        base_url="https://ai-gateway.vercel.sh",
        key_name="Vercel AI Gateway API key",
        env_vars=['AI_GATEWAY_API_KEY'],
        api=anthropic_messages_api(),
    )

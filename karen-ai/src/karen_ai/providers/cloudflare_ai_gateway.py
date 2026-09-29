"""Cloudflare AI Gateway provider (multi-API gateway), mirroring cloudflare-ai-gateway.ts."""

from __future__ import annotations

from ..api import anthropic_messages_api, openai_completions_api, openai_responses_api
from ..auth.types import ProviderAuth
from ..model_catalog import flatten_chat_model_catalog
from ..models import CreateProviderOptions, create_provider
from .cloudflare_auth import cloudflare_ai_gateway_auth
from .cloudflare_stream import cloudflare_streams


def cloudflare_ai_gateway_provider():
    # The api map is pinned to all three APIs: models.dev's gateway catalog drops and
    # restores `workers-ai/*` (openai-completions) entries over time, and inference from
    # `models` alone would otherwise reject the openai-completions entry whenever the
    # generated catalog happens to contain none.
    return create_provider(
        CreateProviderOptions(
            id="cloudflare-ai-gateway",
            name="Cloudflare AI Gateway",
            auth=ProviderAuth(api_key=cloudflare_ai_gateway_auth()),
            models=list(flatten_chat_model_catalog("cloudflare-ai-gateway").values()),
            api={
                "anthropic-messages": cloudflare_streams(anthropic_messages_api()),
                "openai-completions": cloudflare_streams(openai_completions_api()),
                "openai-responses": cloudflare_streams(openai_responses_api()),
            },
        )
    )

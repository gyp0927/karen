"""OpenAI Codex provider (ChatGPT Plus/Pro subscription OAuth), mirroring openai-codex.ts."""

from __future__ import annotations

from ..api import openai_codex_responses_api
from ..auth.types import ProviderAuth
from ..model_catalog import flatten_chat_model_catalog
from ..models import CreateProviderOptions, create_provider

OPENAI_CODEX_BASE_URL = "https://chatgpt.com/backend-api"


def openai_codex_provider():
    from ..auth.oauth import openai_codex_oauth

    return create_provider(
        CreateProviderOptions(
            id="openai-codex",
            name="OpenAI Codex",
            base_url=OPENAI_CODEX_BASE_URL,
            auth=ProviderAuth(oauth=openai_codex_oauth),
            models=list(flatten_chat_model_catalog("openai-codex").values()),
            api=openai_codex_responses_api(),
        )
    )

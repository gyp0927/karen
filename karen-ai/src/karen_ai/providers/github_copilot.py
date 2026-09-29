"""GitHub Copilot provider (multi-API + subscription OAuth), mirroring github-copilot.ts."""

from __future__ import annotations

from typing import Optional, Sequence

from ..api import anthropic_messages_api, openai_completions_api, openai_responses_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import Credential, ProviderAuth
from ..model_catalog import flatten_chat_model_catalog
from ..models import CreateProviderOptions, create_provider
from ..types import Model


def _filter_models(models: Sequence[Model], credential: Optional[Credential]) -> Sequence[Model]:
    if credential is None or credential.type != "oauth":
        return models
    available_model_ids = (credential.model_extra or {}).get("availableModelIds")
    if not isinstance(available_model_ids, list) or not all(isinstance(id, str) for id in available_model_ids):
        return models
    available = set(available_model_ids)
    return [model for model in models if model.id in available]


def github_copilot_provider():
    from ..auth.oauth import github_copilot_oauth

    return create_provider(
        CreateProviderOptions(
            id="github-copilot",
            name="GitHub Copilot",
            base_url="https://api.individual.githubcopilot.com",
            auth=ProviderAuth(
                api_key=env_api_key_auth("GitHub Copilot token", ["COPILOT_GITHUB_TOKEN"]),
                oauth=github_copilot_oauth,
            ),
            models=list(flatten_chat_model_catalog("github-copilot").values()),
            filter_models=_filter_models,
            api={
                "anthropic-messages": anthropic_messages_api(),
                "openai-completions": openai_completions_api(),
                "openai-responses": openai_responses_api(),
            },
        )
    )

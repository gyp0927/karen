"""OpenRouter provider: unified access to many upstream models.

Mirroring pi-ai's providers/openrouter.ts: chat models from the vendored
catalog (anthropic-messages + openai-completions groups), image generation via
openrouter-images, and TypeSafe System One classification served at
/api/v1/systemone.
"""

from __future__ import annotations

from typing import Optional

from ..api import anthropic_messages_api, openai_completions_api
from ..auth.types import ApiKeyAuth, ApiKeyCredential, ApiKeyResolveInput, AuthResult, ModelAuth, ProviderAuth
from ..errors import AbortError
from ..model_catalog import flatten_all_model_catalog
from ..models import CreateProviderOptions, create_provider

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def _openrouter_api_key_auth() -> ApiKeyAuth:
    async def login(interaction):
        from ..auth.types import AuthPromptSecret

        if getattr(interaction, "signal", None) and interaction.signal.aborted:
            raise AbortError()
        key = await interaction.prompt(AuthPromptSecret(message="Enter OpenRouter API key"))
        return ApiKeyCredential(key=key)

    async def resolve(input: ApiKeyResolveInput) -> Optional[AuthResult]:
        input.signal.throw_if_aborted()
        if input.credential and input.credential.key:
            return AuthResult(
                auth=ModelAuth(api_key=input.credential.key), env=input.credential.env, source="stored credential"
            )
        api_key = await input.ctx.env(OPENROUTER_API_KEY_ENV)
        input.signal.throw_if_aborted()
        if api_key:
            return AuthResult(auth=ModelAuth(api_key=api_key), source=OPENROUTER_API_KEY_ENV)
        return None

    return ApiKeyAuth(name="OpenRouter API key", login=login, resolve=resolve)

def openrouter_provider():
    from ..api import openrouter_images, typesafe_system_one
    from ..auth.oauth import openrouter_oauth

    return create_provider(
        CreateProviderOptions(
            id="openrouter",
            name="OpenRouter",
            base_url=OPENROUTER_BASE_URL,
            auth=ProviderAuth(api_key=_openrouter_api_key_auth(), oauth=openrouter_oauth),
            models=list(flatten_all_model_catalog("openrouter").values()),
            api={
                "anthropic-messages": anthropic_messages_api(),
                "openai-completions": openai_completions_api(),
            },
            images={"openrouter-images": openrouter_images.generate_images},
            # OpenRouter serves TypeSafe's System One protocol at /api/v1/systemone.
            classifiers={"typesafe-system-one": typesafe_system_one.classify},
        )
    )

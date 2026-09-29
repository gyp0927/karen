"""DeepSeek provider (OpenAI-compatible with its own thinking format)."""

from __future__ import annotations

from typing import Optional

from ..api import openai_completions_api
from ..auth.types import ApiKeyAuth, ApiKeyCredential, ApiKeyResolveInput, AuthResult, ModelAuth, ProviderAuth
from ..errors import AbortError
from ..model_catalog import flatten_chat_model_catalog
from ..models import CreateProviderOptions, create_provider
from ..types import Model

DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

def _deepseek_api_key_auth() -> ApiKeyAuth:
    async def login(interaction):
        from ..auth.types import AuthPromptSecret

        if getattr(interaction, "signal", None) and interaction.signal.aborted:
            raise AbortError()
        key = await interaction.prompt(AuthPromptSecret(message="Enter DeepSeek API key"))
        return ApiKeyCredential(key=key)

    async def resolve(input: ApiKeyResolveInput) -> Optional[AuthResult]:
        input.signal.throw_if_aborted()
        if input.credential and input.credential.key:
            return AuthResult(
                auth=ModelAuth(api_key=input.credential.key), env=input.credential.env, source="stored credential"
            )
        api_key = await input.ctx.env(DEEPSEEK_API_KEY_ENV)
        input.signal.throw_if_aborted()
        if api_key:
            return AuthResult(auth=ModelAuth(api_key=api_key), source=DEEPSEEK_API_KEY_ENV)
        return None

    return ApiKeyAuth(name="DeepSeek API key", login=login, resolve=resolve)

def deepseek_provider():
    return create_provider(
        CreateProviderOptions(
            id="deepseek",
            name="DeepSeek",
            base_url=DEEPSEEK_BASE_URL,
            auth=ProviderAuth(api_key=_deepseek_api_key_auth()),
            models=list(flatten_chat_model_catalog("deepseek").values()),
            api=openai_completions_api(),
        )
    )

"""Google (Gemini Developer API) provider with a static flagship catalog."""

from __future__ import annotations

from typing import Optional

from ..api import google_generative_ai_api
from ..auth.types import ApiKeyAuth, ApiKeyCredential, ApiKeyResolveInput, AuthResult, ModelAuth, ProviderAuth
from ..errors import AbortError
from ..model_catalog import flatten_chat_model_catalog
from ..models import CreateProviderOptions, create_provider
from ..types import Model

GOOGLE_API_KEY_ENV = "GEMINI_API_KEY"
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

def _google_api_key_auth() -> ApiKeyAuth:
    async def login(interaction):
        from ..auth.types import AuthPromptSecret

        if getattr(interaction, "signal", None) and interaction.signal.aborted:
            raise AbortError()
        key = await interaction.prompt(AuthPromptSecret(message="Enter Gemini API key"))
        return ApiKeyCredential(key=key)

    async def resolve(input: ApiKeyResolveInput) -> Optional[AuthResult]:
        input.signal.throw_if_aborted()
        if input.credential and input.credential.key:
            return AuthResult(
                auth=ModelAuth(api_key=input.credential.key), env=input.credential.env, source="stored credential"
            )
        api_key = await input.ctx.env(GOOGLE_API_KEY_ENV)
        input.signal.throw_if_aborted()
        if api_key:
            return AuthResult(auth=ModelAuth(api_key=api_key), source=GOOGLE_API_KEY_ENV)
        return None

    return ApiKeyAuth(name="Gemini API key", login=login, resolve=resolve)

def google_provider():
    return create_provider(
        CreateProviderOptions(
            id="google",
            name="Google",
            base_url=GOOGLE_BASE_URL,
            auth=ProviderAuth(api_key=_google_api_key_auth()),
            models=list(flatten_chat_model_catalog("google").values()),
            api=google_generative_ai_api(),
        )
    )

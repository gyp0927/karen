"""Anthropic provider, mirroring pi-ai's providers/anthropic.ts (vendored catalog)."""

from __future__ import annotations

from typing import Optional

from ..api import anthropic_messages_api
from ..auth.types import ApiKeyAuth, ApiKeyResolveInput, AuthResult, ModelAuth, ProviderAuth
from ..models import create_provider
from ..models import CreateProviderOptions
from ..model_catalog import flatten_chat_model_catalog

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_OAUTH_TOKEN_ENV = "ANTHROPIC_OAUTH_TOKEN"

ANTHROPIC_BASE_URL = "https://api.anthropic.com"

def _anthropic_api_key_auth() -> ApiKeyAuth:
    async def login(interaction):
        from ..auth.types import ApiKeyCredential, AuthPromptSecret
        from ..errors import AbortError

        if getattr(interaction, "signal", None) and interaction.signal.aborted:
            raise AbortError()
        key = await interaction.prompt(AuthPromptSecret(message="Enter Anthropic API key"))
        return ApiKeyCredential(key=key)

    async def resolve(input: ApiKeyResolveInput) -> Optional[AuthResult]:
        input.signal.throw_if_aborted()
        if input.credential and input.credential.key:
            return AuthResult(
                auth=ModelAuth(api_key=input.credential.key), env=input.credential.env, source="stored credential"
            )

        auth_token = await input.ctx.env(ANTHROPIC_AUTH_TOKEN_ENV)
        input.signal.throw_if_aborted()
        if auth_token:
            return AuthResult(
                auth=ModelAuth(headers={"Authorization": f"Bearer {auth_token}"}),
                source=ANTHROPIC_AUTH_TOKEN_ENV,
            )

        for env_var in (ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV):
            api_key = await input.ctx.env(env_var)
            input.signal.throw_if_aborted()
            if api_key:
                return AuthResult(auth=ModelAuth(api_key=api_key), source=env_var)
        return None

    return ApiKeyAuth(name="Anthropic API key", login=login, resolve=resolve)

def anthropic_provider():
    from ..auth.oauth import anthropic_oauth

    return create_provider(
        CreateProviderOptions(
            id="anthropic",
            name="Anthropic",
            base_url=ANTHROPIC_BASE_URL,
            auth=ProviderAuth(api_key=_anthropic_api_key_auth(), oauth=anthropic_oauth),
            models=list(flatten_chat_model_catalog("anthropic").values()),
            api=anthropic_messages_api(),
        )
    )

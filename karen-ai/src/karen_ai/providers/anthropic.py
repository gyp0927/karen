"""Anthropic provider, mirroring pi-ai's providers/anthropic.ts (static catalog)."""

from __future__ import annotations

from typing import Optional

from ..api import anthropic_messages_api
from ..auth.types import ApiKeyAuth, ApiKeyResolveInput, AuthResult, ModelAuth, ProviderAuth
from ..models import create_provider
from ..models import CreateProviderOptions
from ..types import AnthropicMessagesCompat, Model, ModelCost

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_OAUTH_TOKEN_ENV = "ANTHROPIC_OAUTH_TOKEN"

ANTHROPIC_BASE_URL = "https://api.anthropic.com"


def _model(
    id: str,
    name: str,
    *,
    input_cost: float,
    output_cost: float,
    cache_read: float,
    cache_write: float,
    context_window: int = 200_000,
    max_tokens: int = 64_000,
    reasoning: bool = True,
    compat: Optional[AnthropicMessagesCompat] = None,
) -> Model:
    return Model(
        id=id,
        name=name,
        api="anthropic-messages",
        provider="anthropic",
        base_url=ANTHROPIC_BASE_URL,
        input=["text", "image"],
        cost=ModelCost(input=input_cost, output=output_cost, cache_read=cache_read, cache_write=cache_write),
        reasoning=reasoning,
        context_window=context_window,
        max_tokens=max_tokens,
        compat=compat,
    )


ANTHROPIC_MODELS = [
    _model(
        "claude-opus-5",
        "Claude Opus 5",
        input_cost=15.0,
        output_cost=75.0,
        cache_read=1.5,
        cache_write=18.75,
        max_tokens=128_000,
        compat=AnthropicMessagesCompat(force_adaptive_thinking=True, supports_temperature=False),
    ),
    _model(
        "claude-sonnet-5",
        "Claude Sonnet 5",
        input_cost=3.0,
        output_cost=15.0,
        cache_read=0.3,
        cache_write=3.75,
        max_tokens=128_000,
        compat=AnthropicMessagesCompat(force_adaptive_thinking=True),
    ),
    _model(
        "claude-haiku-4-5-20251001",
        "Claude Haiku 4.5",
        input_cost=1.0,
        output_cost=5.0,
        cache_read=0.1,
        cache_write=1.25,
        max_tokens=64_000,
    ),
    _model(
        "claude-sonnet-4-5-20250929",
        "Claude Sonnet 4.5",
        input_cost=3.0,
        output_cost=15.0,
        cache_read=0.3,
        cache_write=3.75,
    ),
    _model(
        "claude-opus-4-5",
        "Claude Opus 4.5",
        input_cost=15.0,
        output_cost=75.0,
        cache_read=1.5,
        cache_write=18.75,
    ),
]


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
    return create_provider(
        CreateProviderOptions(
            id="anthropic",
            name="Anthropic",
            base_url=ANTHROPIC_BASE_URL,
            auth=ProviderAuth(api_key=_anthropic_api_key_auth()),
            models=list(ANTHROPIC_MODELS),
            api=anthropic_messages_api(),
        )
    )

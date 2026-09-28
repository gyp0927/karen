"""DeepSeek provider (OpenAI-compatible with its own thinking format)."""

from __future__ import annotations

from typing import Optional

from ..api import openai_completions_api
from ..auth.types import ApiKeyAuth, ApiKeyCredential, ApiKeyResolveInput, AuthResult, ModelAuth, ProviderAuth
from ..errors import AbortError
from ..models import CreateProviderOptions, create_provider
from ..types import Model, ModelCost

DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _model(
    id: str,
    name: str,
    *,
    input_cost: float,
    output_cost: float,
    cache_read: float,
    reasoning: bool,
    context_window: int = 128_000,
    max_tokens: int = 64_000,
) -> Model:
    return Model(
        id=id,
        name=name,
        api="openai-completions",
        provider="deepseek",
        base_url=DEEPSEEK_BASE_URL,
        input=["text"],
        cost=ModelCost(input=input_cost, output=output_cost, cache_read=cache_read, cache_write=input_cost),
        reasoning=reasoning,
        context_window=context_window,
        max_tokens=max_tokens,
    )


DEEPSEEK_MODELS = [
    _model("deepseek-chat", "DeepSeek V3.2 Chat", input_cost=0.28, output_cost=0.42, cache_read=0.028, reasoning=False),
    _model("deepseek-reasoner", "DeepSeek V3.2 Reasoner", input_cost=0.28, output_cost=0.42, cache_read=0.028, reasoning=True),
]


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
            models=list(DEEPSEEK_MODELS),
            api=openai_completions_api(),
        )
    )

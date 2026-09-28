"""OpenAI provider with a static flagship catalog.

Matches pi-ai: the OpenAI provider speaks the Responses API for all models.
"""

from __future__ import annotations

from typing import Optional

from ..api import openai_responses_api
from ..auth.types import ApiKeyAuth, ApiKeyCredential, ApiKeyResolveInput, AuthResult, ModelAuth, ProviderAuth
from ..errors import AbortError
from ..models import CreateProviderOptions, create_provider
from ..types import Model, ModelCost

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL = "https://api.openai.com/v1"


def _model(
    id: str,
    name: str,
    *,
    input_cost: float,
    output_cost: float,
    cache_read: float = 0.0,
    cache_write: float = 0.0,
    context_window: int = 400_000,
    max_tokens: int = 128_000,
    reasoning: bool = True,
) -> Model:
    return Model(
        id=id,
        name=name,
        api="openai-responses",
        provider="openai",
        base_url=OPENAI_BASE_URL,
        input=["text", "image"],
        cost=ModelCost(input=input_cost, output=output_cost, cache_read=cache_read, cache_write=cache_write),
        reasoning=reasoning,
        context_window=context_window,
        max_tokens=max_tokens,
    )


OPENAI_MODELS = [
    _model("gpt-5.2", "GPT-5.2", input_cost=1.75, output_cost=14.0, cache_read=0.175),
    _model("gpt-5.1", "GPT-5.1", input_cost=1.25, output_cost=10.0, cache_read=0.125),
    _model("gpt-5", "GPT-5", input_cost=1.25, output_cost=10.0, cache_read=0.125),
    _model("gpt-5-mini", "GPT-5 mini", input_cost=0.25, output_cost=2.0, cache_read=0.025),
    _model("gpt-5-nano", "GPT-5 nano", input_cost=0.05, output_cost=0.4, cache_read=0.005),
    _model("o4-mini", "o4-mini", input_cost=1.1, output_cost=4.4, cache_read=0.275),
    _model("gpt-4.1", "GPT-4.1", input_cost=2.0, output_cost=8.0, cache_read=0.5, context_window=1_000_000, reasoning=False),
]


def _openai_api_key_auth() -> ApiKeyAuth:
    async def login(interaction):
        from ..auth.types import AuthPromptSecret

        if getattr(interaction, "signal", None) and interaction.signal.aborted:
            raise AbortError()
        key = await interaction.prompt(AuthPromptSecret(message="Enter OpenAI API key"))
        return ApiKeyCredential(key=key)

    async def resolve(input: ApiKeyResolveInput) -> Optional[AuthResult]:
        input.signal.throw_if_aborted()
        if input.credential and input.credential.key:
            return AuthResult(
                auth=ModelAuth(api_key=input.credential.key), env=input.credential.env, source="stored credential"
            )
        api_key = await input.ctx.env(OPENAI_API_KEY_ENV)
        input.signal.throw_if_aborted()
        if api_key:
            return AuthResult(auth=ModelAuth(api_key=api_key), source=OPENAI_API_KEY_ENV)
        return None

    return ApiKeyAuth(name="OpenAI API key", login=login, resolve=resolve)


def openai_provider():
    return create_provider(
        CreateProviderOptions(
            id="openai",
            name="OpenAI",
            base_url=OPENAI_BASE_URL,
            auth=ProviderAuth(api_key=_openai_api_key_auth()),
            models=list(OPENAI_MODELS),
            api=openai_responses_api(),
        )
    )

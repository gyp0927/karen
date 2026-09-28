"""Mistral provider with a static flagship catalog."""

from __future__ import annotations

from typing import Optional

from ..api import mistral_conversations_api
from ..auth.types import ApiKeyAuth, ApiKeyCredential, ApiKeyResolveInput, AuthResult, ModelAuth, ProviderAuth
from ..errors import AbortError
from ..models import CreateProviderOptions, create_provider
from ..types import Model, ModelCost

MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"
MISTRAL_BASE_URL = "https://api.mistral.ai"


def _model(
    id: str,
    name: str,
    *,
    input_cost: float,
    output_cost: float,
    context_window: int = 128_000,
    max_tokens: int = 32_000,
    reasoning: bool = False,
) -> Model:
    return Model(
        id=id,
        name=name,
        api="mistral-conversations",
        provider="mistral",
        base_url=MISTRAL_BASE_URL,
        input=["text", "image"],
        cost=ModelCost(input=input_cost, output=output_cost, cache_read=0.0, cache_write=0.0),
        reasoning=reasoning,
        context_window=context_window,
        max_tokens=max_tokens,
    )


MISTRAL_MODELS = [
    _model("mistral-large-latest", "Mistral Large", input_cost=0.5, output_cost=1.5, context_window=256_000, reasoning=True),
    _model("mistral-medium-latest", "Mistral Medium", input_cost=0.4, output_cost=2.0, reasoning=True),
    _model("mistral-small-latest", "Mistral Small", input_cost=0.1, output_cost=0.3, reasoning=True),
    _model("codestral-latest", "Codestral", input_cost=0.3, output_cost=0.9),
    _model("devstral-latest", "Devstral", input_cost=0.4, output_cost=2.0, reasoning=True),
]


def _mistral_api_key_auth() -> ApiKeyAuth:
    async def login(interaction):
        from ..auth.types import AuthPromptSecret

        if getattr(interaction, "signal", None) and interaction.signal.aborted:
            raise AbortError()
        key = await interaction.prompt(AuthPromptSecret(message="Enter Mistral API key"))
        return ApiKeyCredential(key=key)

    async def resolve(input: ApiKeyResolveInput) -> Optional[AuthResult]:
        input.signal.throw_if_aborted()
        if input.credential and input.credential.key:
            return AuthResult(
                auth=ModelAuth(api_key=input.credential.key), env=input.credential.env, source="stored credential"
            )
        api_key = await input.ctx.env(MISTRAL_API_KEY_ENV)
        input.signal.throw_if_aborted()
        if api_key:
            return AuthResult(auth=ModelAuth(api_key=api_key), source=MISTRAL_API_KEY_ENV)
        return None

    return ApiKeyAuth(name="Mistral API key", login=login, resolve=resolve)


def mistral_provider():
    return create_provider(
        CreateProviderOptions(
            id="mistral",
            name="Mistral",
            base_url=MISTRAL_BASE_URL,
            auth=ProviderAuth(api_key=_mistral_api_key_auth()),
            models=list(MISTRAL_MODELS),
            api=mistral_conversations_api(),
        )
    )

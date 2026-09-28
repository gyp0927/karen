"""Google (Gemini Developer API) provider with a static flagship catalog."""

from __future__ import annotations

from typing import Optional

from ..api import google_generative_ai_api
from ..auth.types import ApiKeyAuth, ApiKeyCredential, ApiKeyResolveInput, AuthResult, ModelAuth, ProviderAuth
from ..errors import AbortError
from ..models import CreateProviderOptions, create_provider
from ..types import Model, ModelCost

GOOGLE_API_KEY_ENV = "GEMINI_API_KEY"
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _model(
    id: str,
    name: str,
    *,
    input_cost: float,
    output_cost: float,
    cache_read: float = 0.0,
    cache_write: float = 0.0,
    context_window: int = 1_000_000,
    max_tokens: int = 65_536,
    reasoning: bool = True,
) -> Model:
    return Model(
        id=id,
        name=name,
        api="google-generative-ai",
        provider="google",
        base_url=GOOGLE_BASE_URL,
        input=["text", "image"],
        cost=ModelCost(input=input_cost, output=output_cost, cache_read=cache_read, cache_write=cache_write),
        reasoning=reasoning,
        context_window=context_window,
        max_tokens=max_tokens,
    )


GOOGLE_MODELS = [
    _model("gemini-3-pro-preview", "Gemini 3 Pro Preview", input_cost=2.0, output_cost=12.0, cache_read=0.2),
    _model("gemini-3-flash-preview", "Gemini 3 Flash Preview", input_cost=0.5, output_cost=3.0, cache_read=0.05),
    _model("gemini-2.5-pro", "Gemini 2.5 Pro", input_cost=1.25, output_cost=10.0, cache_read=0.125),
    _model("gemini-2.5-flash", "Gemini 2.5 Flash", input_cost=0.3, output_cost=2.5, cache_read=0.03),
    _model("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite", input_cost=0.1, output_cost=0.4, cache_read=0.01),
]


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
            models=list(GOOGLE_MODELS),
            api=google_generative_ai_api(),
        )
    )

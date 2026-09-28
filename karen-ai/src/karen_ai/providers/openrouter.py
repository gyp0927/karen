"""OpenRouter provider: unified access to many upstream models.

Static mini-catalog of popular models; call `openrouter_fetch_models` via a
dynamic overlay later, or extend the list by hand.
"""

from __future__ import annotations

from typing import Optional

from ..api import openai_completions_api
from ..auth.types import ApiKeyAuth, ApiKeyCredential, ApiKeyResolveInput, AuthResult, ModelAuth, ProviderAuth
from ..errors import AbortError
from ..models import CreateProviderOptions, create_provider
from ..types import ClassifierModel, ImageModel, Model, ModelCost

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _model(
    id: str,
    name: str,
    *,
    input_cost: float,
    output_cost: float,
    cache_read: float = 0.0,
    cache_write: float = 0.0,
    context_window: int = 200_000,
    max_tokens: int = 64_000,
    reasoning: bool = False,
    image_input: bool = True,
) -> Model:
    return Model(
        id=id,
        name=name,
        api="openai-completions",
        provider="openrouter",
        base_url=OPENROUTER_BASE_URL,
        input=["text", "image"] if image_input else ["text"],
        cost=ModelCost(input=input_cost, output=output_cost, cache_read=cache_read, cache_write=cache_write),
        reasoning=reasoning,
        context_window=context_window,
        max_tokens=max_tokens,
    )


OPENROUTER_MODELS = [
    _model(
        "anthropic/claude-sonnet-5",
        "Claude Sonnet 5 (OpenRouter)",
        input_cost=3.0,
        output_cost=15.0,
        cache_read=0.3,
        cache_write=3.75,
        reasoning=True,
    ),
    _model(
        "anthropic/claude-opus-5",
        "Claude Opus 5 (OpenRouter)",
        input_cost=15.0,
        output_cost=75.0,
        cache_read=1.5,
        cache_write=18.75,
        reasoning=True,
    ),
    _model(
        "openai/gpt-5.2",
        "GPT-5.2 (OpenRouter)",
        input_cost=1.75,
        output_cost=14.0,
        cache_read=0.175,
        context_window=400_000,
        reasoning=True,
    ),
    _model(
        "deepseek/deepseek-chat-v3.2",
        "DeepSeek V3.2 (OpenRouter)",
        input_cost=0.28,
        output_cost=0.42,
        image_input=False,
        reasoning=False,
    ),
    _model(
        "google/gemini-3-pro",
        "Gemini 3 Pro (OpenRouter)",
        input_cost=1.25,
        output_cost=10.0,
        context_window=1_000_000,
        reasoning=True,
    ),
    _model(
        "moonshotai/kimi-k2.5",
        "Kimi K2.5 (OpenRouter)",
        input_cost=0.6,
        output_cost=2.5,
        context_window=256_000,
        reasoning=True,
        image_input=False,
    ),
    _model(
        "zai/glm-5",
        "GLM-5 (OpenRouter)",
        input_cost=0.6,
        output_cost=2.2,
        context_window=128_000,
        reasoning=True,
        image_input=False,
    ),
]


OPENROUTER_IMAGE_MODELS = [
    ImageModel(
        id="google/gemini-3-pro-image-preview",
        name="Gemini 3 Pro Image (OpenRouter)",
        api="openrouter-images",
        provider="openrouter",
        base_url=OPENROUTER_BASE_URL,
        input=["text", "image"],
        output=["image", "text"],
        cost=ModelCost(input=2.0, output=12.0, cache_read=0.0, cache_write=0.0),
    ),
    ImageModel(
        id="google/gemini-2.5-flash-image",
        name="Gemini 2.5 Flash Image (OpenRouter)",
        api="openrouter-images",
        provider="openrouter",
        base_url=OPENROUTER_BASE_URL,
        input=["text", "image"],
        output=["image"],
        cost=ModelCost(input=0.3, output=2.5, cache_read=0.0, cache_write=0.0),
    ),
]


# OpenRouter serves TypeSafe's System One protocol at /api/v1/systemone.
OPENROUTER_CLASSIFIER_MODELS = [
    ClassifierModel(
        id="typesafe/jev",
        name="Jev (OpenRouter)",
        api="typesafe-system-one",
        provider="openrouter",
        base_url=OPENROUTER_BASE_URL,
        input=["text"],
        cost=ModelCost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
    ),
]


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

    return create_provider(
        CreateProviderOptions(
            id="openrouter",
            name="OpenRouter",
            base_url=OPENROUTER_BASE_URL,
            auth=ProviderAuth(api_key=_openrouter_api_key_auth()),
            models=[*OPENROUTER_MODELS, *OPENROUTER_IMAGE_MODELS, *OPENROUTER_CLASSIFIER_MODELS],
            api=openai_completions_api(),
            images={"openrouter-images": openrouter_images.generate_images},
            # OpenRouter serves TypeSafe's System One protocol at /api/v1/systemone.
            classifiers={"typesafe-system-one": typesafe_system_one.classify},
        )
    )

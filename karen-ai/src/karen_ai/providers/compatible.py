"""Generic OpenAI-compatible provider factory for custom/self-hosted endpoints
(vLLM, llama.cpp, SGLang, LiteLLM proxies, ...)."""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..api import openai_completions_api
from ..auth.types import ApiKeyAuth, ApiKeyResolveInput, AuthResult, ModelAuth, ProviderAuth
from ..models import CreateProviderOptions, create_provider
from ..types import Model, OpenAICompletionsCompat


def openai_compatible_provider(
    *,
    id: str,
    name: Optional[str] = None,
    base_url: str,
    models: Sequence[Model],
    api_key_env: Optional[str] = None,
    compat: Optional[OpenAICompletionsCompat] = None,
    headers: Optional[dict] = None,
):
    """Build a provider for any OpenAI Chat Completions compatible endpoint.

    - `api_key_env`: environment variable consulted for the key. When omitted,
      the provider is treated as keyless (local servers); auth resolves to an
      empty key and compat auto-detection still applies per model.
    - `compat`: applied to every listed model that has no explicit compat of its own.
    """
    prepared: List[Model] = []
    for model in models:
        update = {"provider": id, "base_url": model.base_url or base_url, "api": model.api or "openai-completions"}
        if compat is not None and model.compat is None:
            update["compat"] = compat
        prepared.append(model.model_copy(update=update))

    async def resolve(input: ApiKeyResolveInput) -> Optional[AuthResult]:
        input.signal.throw_if_aborted()
        if input.credential and input.credential.key:
            return AuthResult(
                auth=ModelAuth(api_key=input.credential.key), env=input.credential.env, source="stored credential"
            )
        if api_key_env:
            api_key = await input.ctx.env(api_key_env)
            input.signal.throw_if_aborted()
            if api_key:
                return AuthResult(auth=ModelAuth(api_key=api_key), source=api_key_env)
            return None
        # Keyless local server: always configured.
        return AuthResult(auth=ModelAuth(api_key="unused"), source="keyless")

    return create_provider(
        CreateProviderOptions(
            id=id,
            name=name or id,
            base_url=base_url,
            headers=headers,
            auth=ProviderAuth(api_key=ApiKeyAuth(name=f"{name or id} API key", resolve=resolve)),
            models=prepared,
            api=openai_completions_api(),
        )
    )

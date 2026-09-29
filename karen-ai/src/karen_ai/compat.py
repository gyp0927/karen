"""Deprecated ambient API surface, mirroring pi-ai's `compat.ts`.

Preserves the pre-`createModels()` global API: api-dispatch `stream()` /
`complete()` with environment API-key injection, the api registry, static
catalog reads (`get_model` / `get_models` / `get_providers`), and image
generation. New code should use `create_models()` and provider factories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from .api import (
    anthropic_messages_api,
    azure_openai_responses_api,
    bedrock_converse_stream_api,
    google_generative_ai_api,
    google_vertex_api,
    mistral_conversations_api,
    openai_codex_responses_api,
    openai_completions_api,
    openai_responses_api,
    pi_messages_api,
)
from .env_api_keys import AMBIENT_AUTH_MARKER, get_env_api_key
from .event_stream import AssistantMessageEventStream
from .lazy import ProviderStreams
from .providers import (
    builtin_models,
    get_builtin_model,
    get_builtin_models,
    get_builtin_providers,
)
from .providers.faux import FauxProviderRegistration, register_faux_provider as _create_faux_core
from .transcript import normalize_context
from .types import AssistantMessage, Context, Model, ProviderHeaders, SimpleStreamOptions, StreamOptions

#: @deprecated Static catalog read. Use `providers.get_builtin_model` or `Models.get_model()`.
get_model = get_builtin_model
#: @deprecated Static catalog read. Use `providers.get_builtin_models` or `Models.get_models()`.
get_models = get_builtin_models
#: @deprecated Static catalog read. Use `providers.get_builtin_providers` or `Models.get_providers()`.
get_providers = get_builtin_providers


@dataclass
class ApiProvider:
    """An api id bound to its stream implementations."""

    api: str
    stream: Callable[..., AssistantMessageEventStream]
    stream_simple: Callable[..., AssistantMessageEventStream]


_registry: Dict[str, tuple[ApiProvider, Optional[str]]] = {}


def register_api_provider(provider: ApiProvider, source_id: Optional[str] = None) -> None:
    """Registers (or replaces) the implementation for one api id."""

    def wrap(stream_fn: Callable[..., AssistantMessageEventStream], api: str):
        def wrapped(model: Model, context, options=None) -> AssistantMessageEventStream:
            if model.api != api:
                raise ValueError(f"Mismatched api: {model.api} expected {api}")
            return stream_fn(model, context, options)

        return wrapped

    _registry[provider.api] = (
        ApiProvider(
            api=provider.api,
            stream=wrap(provider.stream, provider.api),
            stream_simple=wrap(provider.stream_simple, provider.api),
        ),
        source_id,
    )


def get_api_provider(api: str) -> Optional[ApiProvider]:
    entry = _registry.get(api)
    return entry[0] if entry else None


def get_api_providers() -> List[ApiProvider]:
    return [provider for provider, _ in _registry.values()]


def unregister_api_providers(source_id: str) -> None:
    for api, (_, registered_source) in list(_registry.items()):
        if registered_source == source_id:
            del _registry[api]


_BUILTIN_APIS: List[tuple[str, ProviderStreams]] = [
    ("anthropic-messages", anthropic_messages_api()),
    ("openai-completions", openai_completions_api()),
    ("openai-responses", openai_responses_api()),
    ("openai-codex-responses", openai_codex_responses_api()),
    ("azure-openai-responses", azure_openai_responses_api()),
    ("google-generative-ai", google_generative_ai_api()),
    ("google-vertex", google_vertex_api()),
    ("mistral-conversations", mistral_conversations_api()),
    ("bedrock-converse-stream", bedrock_converse_stream_api()),
    ("pi-messages", pi_messages_api()),
]

_builtin_api_providers: Dict[str, Optional[ApiProvider]] = {}


def register_built_in_api_providers() -> None:
    """Registers the builtin api implementations without clobbering overrides."""
    for api, streams in _BUILTIN_APIS:
        if get_api_provider(api) is None:
            register_api_provider(ApiProvider(api=api, stream=streams.stream, stream_simple=streams.stream_simple))
        _builtin_api_providers[api] = get_api_provider(api)


def reset_api_providers() -> None:
    _registry.clear()
    _builtin_api_providers.clear()
    register_built_in_api_providers()


register_built_in_api_providers()

_compat_models_cache = None


def _compat_models():
    """Built-in `Models` registry used for cloudflare placeholder resolution."""
    global _compat_models_cache
    if _compat_models_cache is None:
        _compat_models_cache = builtin_models()
    return _compat_models_cache


def register_faux_provider(
    *,
    api: str = "faux",
    provider_id: str = "faux",
    models: Optional[List[Model]] = None,
    responses=None,
    token_delay: float = 0.0,
) -> FauxProviderRegistration:
    """Registers a scripted faux api implementation in the api registry."""
    registration = _create_faux_core(
        api=api,
        provider_id=provider_id,
        models=models,
        responses=responses,
        token_delay=token_delay,
    )
    provider = registration.provider
    source_id = f"faux-provider-{api}"
    register_api_provider(
        ApiProvider(api=api, stream=provider.stream, stream_simple=provider.stream_simple),
        source_id,
    )

    def unregister() -> None:
        unregister_api_providers(source_id)

    registration.unregister = unregister
    return registration


class AmbientOptions(StreamOptions):
    """Default options for the ambient surface: api-specific fields read as unset."""

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)  # type: ignore[misc]
        except AttributeError:
            return None


def _with_env_api_key(model: Model, options, fallback_cls=AmbientOptions):
    if options is not None and getattr(options, "api_key", None) and options.api_key.strip():
        return options
    api_key = get_env_api_key(model.provider, options.env if options is not None else None)
    if not api_key or api_key == AMBIENT_AUTH_MARKER:
        return options
    if options is None:
        return fallback_cls(api_key=api_key)
    if isinstance(options, BaseModel):
        return options.model_copy(update={"api_key": api_key})
    if isinstance(options, dict):
        return {**options, "apiKey": api_key}
    return options


def _has_resolved_cloudflare_auth(options) -> bool:
    if options is not None and options.api_key and options.api_key.strip():
        return True
    headers: Optional[ProviderHeaders] = options.headers if options is not None else None
    return isinstance((headers or {}).get("cf-aig-authorization"), str)


def _builtin_provider_for_model(model: Model):
    current = get_api_provider(model.api)
    if current is not _builtin_api_providers.get(model.api):
        return None
    provider = _compat_models().get_provider(model.provider)
    if provider is None or provider.stream is None:
        return None
    return provider if any(candidate.api == model.api for candidate in provider.get_models()) else None


def _resolve_api_provider(api: str) -> ApiProvider:
    provider = get_api_provider(api)
    if provider is None:
        raise ValueError(f"No API provider registered for api: {api}")
    return provider


def stream(model: Model, context: Context, options: Optional[StreamOptions] = None) -> AssistantMessageEventStream:
    """Ambient api-dispatch streaming with environment API-key injection."""
    transcript = normalize_context(context)
    builtin_provider = _builtin_provider_for_model(model)
    if builtin_provider is not None:
        if model.provider.startswith("cloudflare-") and not _has_resolved_cloudflare_auth(options):
            return _compat_models().stream(model, transcript, options)  # type: ignore[arg-type]
        return builtin_provider.stream(model, transcript, _with_env_api_key(model, options))  # type: ignore[arg-type]
    provider = _resolve_api_provider(model.api)
    return provider.stream(model, transcript, _with_env_api_key(model, options))


async def complete(model: Model, context: Context, options: Optional[StreamOptions] = None) -> AssistantMessage:
    return await stream(model, context, options).result()


def stream_simple(
    model: Model, context: Context, options: Optional[SimpleStreamOptions] = None
) -> AssistantMessageEventStream:
    """Ambient api-dispatch simple streaming with environment API-key injection."""
    transcript = normalize_context(context)
    builtin_provider = _builtin_provider_for_model(model)
    if builtin_provider is not None:
        if model.provider.startswith("cloudflare-") and not _has_resolved_cloudflare_auth(options):
            return _compat_models().stream_simple(model, transcript, options)  # type: ignore[arg-type]
        return builtin_provider.stream_simple(model, transcript, _with_env_api_key(model, options))  # type: ignore[arg-type]
    provider = _resolve_api_provider(model.api)
    return provider.stream_simple(model, transcript, _with_env_api_key(model, options))


async def complete_simple(
    model: Model, context: Context, options: Optional[SimpleStreamOptions] = None
) -> AssistantMessage:
    return await stream_simple(model, context, options).result()


__all__ = [
    "ApiProvider",
    "complete",
    "complete_simple",
    "get_api_provider",
    "get_api_providers",
    "get_model",
    "get_models",
    "get_providers",
    "register_api_provider",
    "register_built_in_api_providers",
    "register_faux_provider",
    "reset_api_providers",
    "stream",
    "stream_simple",
    "unregister_api_providers",
]

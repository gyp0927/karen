"""Provider/Models registry, mirroring pi-ai's models.ts.

A `Provider` is the concrete runtime unit: id/name, auth methods, model
listing, and the operations its models support. `Models` is a runtime
collection of providers plus auth application and request convenience.
"""

from __future__ import annotations

import asyncio
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from pydantic import ConfigDict

from .abort import AbortSignal, operation_signal
from .auth.context import AuthContext, default_provider_auth_context
from .auth.credential_store import InMemoryCredentialStore
from .auth.resolve import AuthResolutionOverrides, resolve_provider_auth
from .auth.types import (
    ApiKeyCredential,
    AuthCheck,
    AuthInteraction,
    AuthOperationOptions,
    AuthResult,
    AuthType,
    Credential,
    CredentialStore,
    OAuthCredential,
    ProviderAuth,
)
from .errors import ModelsError
from .event_stream import AssistantMessageEventStream
from .lazy import ProviderStreams, lazy_stream
from .models_store import InMemoryModelsStore, ModelsStore, ModelsStoreEntry, ModelsStoreOperationOptions
from .transcript import normalize_context
from .types import (
    AnyModel,
    Api,
    AssistantImages,
    AssistantMessage,
    ClassifierContext,
    ClassifierModel,
    ClassifierResult,
    Context,
    DeferredCancelOptions,
    DeferredFetchOptions,
    DeferredHandle,
    ImageModel,
    ImagesContext,
    KarenBase,
    Model,
    ModelCostRates,
    ModelThinkingLevel,
    ModelType,
    ProviderHeaders,
    ProviderRequestOptions,
    SimpleStreamOptions,
    StreamOptions,
    TranscriptContext,
    Usage,
    get_model_type,
    is_model_type,
)

_KNOWN_MODEL_TYPES = ("chat", "image", "classifier")


def _has_known_model_type(model: AnyModel) -> bool:
    return get_model_type(model) in _KNOWN_MODEL_TYPES


def _with_known_model_types(entry: ModelsStoreEntry) -> ModelsStoreEntry:
    return ModelsStoreEntry(
        models=[m for m in entry.models if _has_known_model_type(m)],
        last_modified=entry.last_modified,
        checked_at=entry.checked_at,
        etag=entry.etag,
    )


class ModelsPublication(KarenBase):
    """Provider-selected persisted catalog. `persist=None` leaves storage unchanged;
    pass `persist_null=True` semantics via the sentinel `DELETE` below."""

    persist: Any = None  # ModelsStoreEntry | _DELETE | None
    update: Optional[Callable[[], None]] = None


#: Sentinel for "delete the persisted catalog" in ModelsPublication.persist.
DELETE = object()


class RefreshModelsContext(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    credential: Optional[Credential] = None
    stored: Optional[ModelsStoreEntry] = None
    publish: Optional[Callable[[ModelsPublication], Awaitable[bool]]] = None
    allow_network: bool = False
    force: Optional[bool] = None
    signal: AbortSignal


class ModelsRefreshOptions(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    allow_network: Optional[bool] = None
    providers: Optional[List[str]] = None
    force: Optional[bool] = None
    signal: Optional[AbortSignal] = None


class ModelsRefreshResult(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    aborted: bool
    errors: Dict[str, Exception]


class ModelsRequestTransforms(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    transform_headers: Optional[Callable[[ProviderHeaders], Union[ProviderHeaders, Awaitable[ProviderHeaders]]]] = None


class Provider:
    """A provider: concrete runtime unit owning auth, model listing, and operations.

    Built by `create_provider()`; see that function for the parts.
    """

    def __init__(
        self,
        *,
        id: str,
        name: str,
        auth: ProviderAuth,
        base_url: Optional[str] = None,
        headers: Optional[ProviderHeaders] = None,
        get_models: Callable[[], List[Model]],
        get_all_models: Optional[Callable[[], List[AnyModel]]] = None,
        refresh_models: Optional[Callable[[RefreshModelsContext], Awaitable[None]]] = None,
        filter_models: Optional[Callable[[Sequence[Model], Optional[Credential]], Sequence[Model]]] = None,
        filter_all_models: Optional[Callable[[Sequence[AnyModel], Optional[Credential]], Sequence[AnyModel]]] = None,
        stream: Optional[Callable[[Model, TranscriptContext, Optional[StreamOptions]], AssistantMessageEventStream]] = None,
        stream_simple: Optional[
            Callable[[Model, TranscriptContext, Optional[SimpleStreamOptions]], AssistantMessageEventStream]
        ] = None,
        fetch_deferred: Optional[
            Callable[[Model, DeferredHandle, Optional[DeferredFetchOptions]], AssistantMessageEventStream]
        ] = None,
        cancel_deferred: Optional[Callable[[Model, DeferredHandle, Optional[DeferredCancelOptions]], Awaitable[None]]] = None,
        generate_images: Optional[Callable[..., Awaitable[AssistantImages]]] = None,
        classify: Optional[Callable[..., Awaitable[ClassifierResult]]] = None,
    ) -> None:
        self.id = id
        self.name = name
        self.base_url = base_url
        self.headers = headers
        self.auth = auth
        self.get_models = get_models
        self.get_all_models = get_all_models
        self.refresh_models = refresh_models
        self.filter_models = filter_models
        self.filter_all_models = filter_all_models
        self.stream = stream
        self.stream_simple = stream_simple
        self.fetch_deferred = fetch_deferred
        self.cancel_deferred = cancel_deferred
        self.generate_images = generate_images
        self.classify = classify


def _merge_headers(
    base: Optional[ProviderHeaders],
    override: Optional[ProviderHeaders],
) -> Optional[ProviderHeaders]:
    if not base and not override:
        return None
    merged: ProviderHeaders = dict(base or {})
    for name, value in (override or {}).items():
        lower = name.lower()
        for existing in list(merged.keys()):
            if existing.lower() == lower:
                del merged[existing]
        merged[name] = value
    return merged


class CreateModelsOptions(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    credentials: Optional[CredentialStore] = None
    models_store: Optional[ModelsStore] = None
    auth_context: Optional[AuthContext] = None


class Models:
    """Runtime collection of providers plus auth application and request convenience."""

    def __init__(self, options: Optional[CreateModelsOptions] = None) -> None:
        options = options or CreateModelsOptions()
        self._providers: Dict[str, Provider] = {}
        self._credentials: CredentialStore = options.credentials or InMemoryCredentialStore()
        self._models_store: ModelsStore = options.models_store or InMemoryModelsStore()
        self._auth_context: AuthContext = options.auth_context or default_provider_auth_context()
        self._refresh_locks: Dict[str, asyncio.Lock] = {}

    # -- provider management --------------------------------------------------

    def set_provider(self, provider: Provider) -> None:
        """Upsert/replace by provider.id. Provider ids are unique."""
        self._providers[provider.id] = provider

    def delete_provider(self, id: str) -> None:
        self._providers.pop(id, None)

    def clear_providers(self) -> None:
        self._providers.clear()

    def get_providers(self) -> List[Provider]:
        return list(self._providers.values())

    def get_provider(self, id: str) -> Optional[Provider]:
        return self._providers.get(id)

    # -- model reads ------------------------------------------------------------

    def get_models(self, provider: Optional[str] = None) -> List[Model]:
        """Sync read of last-known chat models from one provider or all providers.

        Best-effort: a provider whose get_models() raises yields no models.
        """
        if provider is not None:
            entry = self._providers.get(provider)
            if not entry:
                return []
            try:
                return list(entry.get_models())
            except Exception:
                return []

        models: List[Model] = []
        for entry in self._providers.values():
            try:
                models.extend(entry.get_models())
            except Exception:
                continue
        return models

    def get_all_models(self, provider: Optional[str] = None) -> List[AnyModel]:
        if provider is not None:
            entry = self._providers.get(provider)
            if not entry:
                return []
            try:
                return list(entry.get_all_models() if entry.get_all_models else entry.get_models())
            except Exception:
                return []

        models: List[AnyModel] = []
        for entry in self._providers.values():
            try:
                models.extend(entry.get_all_models() if entry.get_all_models else entry.get_models())
            except Exception:
                continue
        return models

    def get_models_of_type(self, model_type: ModelType, provider: Optional[str] = None) -> List[AnyModel]:
        return [m for m in self.get_all_models(provider) if is_model_type(m, model_type)]

    def get_model(self, provider: str, id: str) -> Optional[Model]:
        for model in self.get_models(provider):
            if model.id == id:
                return model
        return None

    def get_model_of_type(self, model_type: ModelType, provider: str, id: str) -> Optional[AnyModel]:
        for model in self.get_models_of_type(model_type, provider):
            if model.id == id:
                return model
        return None

    # -- refresh ----------------------------------------------------------------

    def _refresh_lock(self, provider_id: str) -> asyncio.Lock:
        return self._refresh_locks.setdefault(provider_id, asyncio.Lock())

    async def _publish_provider_models(
        self,
        provider_id: str,
        signal: AbortSignal,
        publication: ModelsPublication,
    ) -> bool:
        if signal.aborted:
            return False
        if publication.persist is DELETE:
            await self._models_store.delete(provider_id, ModelsStoreOperationOptions(signal=signal))
        elif publication.persist is not None:
            await self._models_store.write(
                provider_id, publication.persist, ModelsStoreOperationOptions(signal=signal)
            )
        if signal.aborted:
            return False
        if publication.update:
            publication.update()
        return True

    async def refresh(self, options: Optional[ModelsRefreshOptions] = None) -> ModelsRefreshResult:
        """Refresh selected configured dynamic providers concurrently (all when omitted).

        Provider errors and cancellation are returned without raising; static,
        unknown, and unconfigured providers are skipped.
        """
        options = options or ModelsRefreshOptions()
        allow_network = options.allow_network if options.allow_network is not None else True
        caller_signal = operation_signal(options.signal)
        errors: Dict[str, Exception] = {}
        if caller_signal.aborted:
            return ModelsRefreshResult(aborted=True, errors=errors)

        selected = set(options.providers) if options.providers else None
        refreshable = [
            p
            for p in self._providers.values()
            if p.refresh_models is not None and (selected is None or p.id in selected)
        ]

        async def refresh_one(provider: Provider) -> None:
            signal = caller_signal
            try:
                async with self._refresh_lock(provider.id):
                    try:
                        stored_credential = await self._read_credential(provider.id, signal)
                    except ModelsError:
                        stored_credential = None

                    stored = await self._models_store.read(provider.id, ModelsStoreOperationOptions(signal=signal))

                    async def publish(publication: ModelsPublication) -> bool:
                        return await self._publish_provider_models(provider.id, signal, publication)

                    # Phase 1: restore cached catalog without network.
                    await provider.refresh_models(
                        RefreshModelsContext(
                            credential=stored_credential,
                            stored=_with_known_model_types(stored) if stored else None,
                            publish=publish,
                            allow_network=False,
                            signal=signal,
                        )
                    )
                    if not allow_network or signal.aborted:
                        return

                    credential = await self._resolve_refresh_credential(provider, stored_credential, signal)
                    if credential is None:
                        return
                    # Phase 2: fetch with the effective credential.
                    await provider.refresh_models(
                        RefreshModelsContext(
                            credential=credential,
                            stored=_with_known_model_types(stored) if stored else None,
                            publish=publish,
                            allow_network=True,
                            force=options.force,
                            signal=signal,
                        )
                    )
            except Exception as error:
                if not signal.aborted:
                    errors[provider.id] = (
                        error
                        if isinstance(error, Exception)
                        else ModelsError("model_source", f"Model refresh failed for {provider.id}")
                    )

        await asyncio.gather(*(refresh_one(p) for p in refreshable))
        return ModelsRefreshResult(aborted=caller_signal.aborted, errors=errors)

    async def _resolve_refresh_credential(
        self,
        provider: Provider,
        stored: Optional[Credential],
        signal: AbortSignal,
    ) -> Optional[Credential]:
        if isinstance(stored, OAuthCredential):
            oauth = provider.auth.oauth
            if not oauth:
                return None
            if time.time() * 1000 < stored.expires:
                return stored
            if signal.aborted:
                return None

            async def refresh_if_needed(current: Optional[Credential]) -> Optional[Credential]:
                if not isinstance(current, OAuthCredential) or time.time() * 1000 < current.expires:
                    return None
                return await oauth.refresh(current, signal)

            post = await self._credentials.modify(provider.id, refresh_if_needed)
            return post if isinstance(post, OAuthCredential) else None

        api_key = provider.auth.api_key
        if not api_key:
            return None
        from .auth.types import ApiKeyResolveInput

        credential = stored if isinstance(stored, ApiKeyCredential) else None
        result = await api_key.resolve(ApiKeyResolveInput(ctx=self._auth_context, credential=credential, signal=signal))
        if not result:
            return None
        return ApiKeyCredential(key=result.auth.api_key, env=result.env)

    async def _read_credential(self, provider_id: str, signal: AbortSignal) -> Optional[Credential]:
        try:
            return await self._credentials.read(provider_id)
        except Exception as error:
            raise ModelsError("auth", f"Credential store read failed for {provider_id}", cause=error)

    # -- auth -------------------------------------------------------------------

    async def _check_provider_auth(
        self,
        provider: Provider,
        credential: Optional[Credential],
        signal: AbortSignal,
    ) -> Optional[AuthCheck]:
        if isinstance(credential, OAuthCredential):
            return AuthCheck(source="OAuth", type="oauth") if provider.auth.oauth else None
        api_key = provider.auth.api_key
        if not api_key:
            return None
        if api_key.check:
            from .auth.types import ApiKeyResolveInput

            try:
                return await api_key.check(
                    ApiKeyResolveInput(
                        ctx=self._auth_context,
                        credential=credential if isinstance(credential, ApiKeyCredential) else None,
                        signal=signal,
                    )
                )
            except Exception as error:
                raise ModelsError("auth", f"API key auth check failed for provider {provider.id}", cause=error)

        resolution = await resolve_provider_auth(provider.id, provider.auth, self._credentials, self._auth_context)
        return AuthCheck(source=resolution.source, type="api_key") if resolution else None

    async def check_auth(
        self, provider_id: str, options: Optional[AuthOperationOptions] = None
    ) -> Optional[AuthCheck]:
        """Check whether a provider has complete auth configuration without refreshing OAuth."""
        signal = operation_signal(options.signal if options else None)
        signal.throw_if_aborted()
        provider = self._providers.get(provider_id)
        if not provider:
            return None
        return await self._check_provider_auth(provider, await self._read_credential(provider_id, signal), signal)

    async def _get_authenticated_providers(
        self, provider_id: Optional[str], signal: AbortSignal
    ) -> List[tuple[Provider, Optional[Credential]]]:
        signal.throw_if_aborted()
        providers = (
            [p for p in [self._providers.get(provider_id)] if p is not None]
            if provider_id
            else self.get_providers()
        )

        async def check(provider: Provider):
            credential = await self._read_credential(provider.id, signal)
            auth = await self._check_provider_auth(provider, credential, signal)
            return (provider, credential) if auth else None

        results = await asyncio.gather(*(check(p) for p in providers))
        return [r for r in results if r is not None]

    async def get_available(
        self, provider_id: Optional[str] = None, options: Optional[AuthOperationOptions] = None
    ) -> List[Model]:
        """Return chat models whose providers have complete auth configuration."""
        signal = operation_signal(options.signal if options else None)
        providers = await self._get_authenticated_providers(provider_id, signal)
        available: List[Model] = []
        for provider, credential in providers:
            models = provider.get_models()
            available.extend(provider.filter_models(models, credential) if provider.filter_models else models)
        return available

    async def get_available_of_type(
        self,
        model_type: ModelType,
        provider_id: Optional[str] = None,
        options: Optional[AuthOperationOptions] = None,
    ) -> List[AnyModel]:
        return [m for m in await self.get_all_available(provider_id, options) if is_model_type(m, model_type)]

    async def get_all_available(
        self, provider_id: Optional[str] = None, options: Optional[AuthOperationOptions] = None
    ) -> List[AnyModel]:
        """Return models of every type whose providers have complete auth configuration."""
        signal = operation_signal(options.signal if options else None)
        providers = await self._get_authenticated_providers(provider_id, signal)
        available: List[AnyModel] = []
        for provider, credential in providers:
            models = provider.get_all_models() if provider.get_all_models else provider.get_models()
            if provider.filter_all_models:
                available.extend(provider.filter_all_models(models, credential))
            elif not provider.filter_models:
                available.extend(models)
            else:
                available_chat_ids = {m.id for m in provider.filter_models(provider.get_models(), credential)}
                available.extend(
                    m for m in models if not is_model_type(m, "chat") or m.id in available_chat_ids
                )
        return available

    async def get_auth(
        self,
        provider_or_model: Union[str, AnyModel],
        overrides: Optional[AuthResolutionOverrides] = None,
    ) -> Optional[AuthResult]:
        """Resolve provider-scoped auth by provider id, or provider auth plus static
        model headers when passed a model."""
        provider_id = provider_or_model if isinstance(provider_or_model, str) else provider_or_model.provider
        provider = self._providers.get(provider_id)
        if not provider:
            return None
        result = await resolve_provider_auth(provider_id, provider.auth, self._credentials, self._auth_context, overrides)
        if not result or isinstance(provider_or_model, str) or not provider_or_model.headers:
            return result
        return AuthResult(
            auth=result.auth.model_copy(update={"headers": _merge_headers(result.auth.headers, provider_or_model.headers)}),
            env=result.env,
            source=result.source,
        )

    async def login(self, provider_id: str, type: AuthType, interaction: AuthInteraction) -> Credential:
        """Run a provider-owned login flow and persist its returned credential."""
        signal = operation_signal(getattr(interaction, "signal", None))
        signal.throw_if_aborted()
        provider = self._providers.get(provider_id)
        if not provider:
            raise ModelsError("provider", f"Unknown provider: {provider_id}")
        method = provider.auth.oauth if type == "oauth" else provider.auth.api_key
        if method is None or method.login is None:
            raise ModelsError("auth", f"{provider.name} does not support {type} login")
        credential = await method.login(interaction)
        try:
            await self._credentials.modify(provider_id, lambda current: _return_credential(credential))
        except Exception as error:
            raise ModelsError("auth", f"Credential store modify failed for {provider_id}", cause=error)
        return credential

    async def logout(self, provider_id: str, options: Optional[AuthOperationOptions] = None) -> None:
        """Remove the stored credential for a provider."""
        signal = operation_signal(options.signal if options else None)
        try:
            await self._credentials.delete(provider_id)
        except Exception as error:
            signal.throw_if_aborted()
            raise ModelsError("auth", f"Credential store delete failed for {provider_id}", cause=error)

    # -- requests -----------------------------------------------------------------

    def _require_provider(self, model: AnyModel) -> Provider:
        provider = self._providers.get(model.provider)
        if not provider:
            raise ModelsError("provider", f"Unknown provider: {model.provider}")
        return provider

    def _require_chat_provider(self, model: Model) -> Provider:
        if not is_model_type(model, "chat"):
            raise ModelsError("provider", f"Model {model.id} is not a chat model")
        return self._require_provider(model)

    async def _apply_auth(
        self,
        model: AnyModel,
        options: Optional[ProviderRequestOptions],
        default_options: Optional[type] = None,
    ) -> tuple[AnyModel, Optional[ProviderRequestOptions]]:
        self._require_provider(model)
        resolution = await self.get_auth(
            model,
            AuthResolutionOverrides(
                api_key=options.api_key if options else None,
                env=options.env if options else None,
                signal=options.signal if options else None,
            ),
        )
        if not resolution:
            raise ModelsError("auth", f"Provider is not configured: {model.provider}")
        auth = resolution.auth

        # Explicit request options win per-field; the Models-only transform runs last.
        api_key = (options.api_key if options and options.api_key else None) or auth.api_key
        headers = _merge_headers(auth.headers, options.headers if options else None)
        transform_headers = getattr(options, "transform_headers", None) if options else None
        if transform_headers:
            maybe = transform_headers(headers or {})
            headers = await maybe if asyncio.iscoroutine(maybe) else maybe
        env = {**(resolution.env or {}), **(options.env or {})} if (resolution.env or (options and options.env)) else None
        request_model = model.model_copy(update={"base_url": auth.base_url}) if auth.base_url else model

        # A caller that passes no options still gets the option type its API expects:
        # adapters read plain fields off it (temperature, reasoning, ...) rather than
        # guarding every access, so a bare ProviderRequestOptions would raise there.
        if options is None:
            request_options = (default_options or ProviderRequestOptions)(api_key=api_key, headers=headers, env=env)
        else:
            request_options = options.model_copy(update={"api_key": api_key, "headers": headers, "env": env})
        return request_model, request_options

    def stream(
        self,
        model: Model,
        context: Context,
        options: Optional[StreamOptions] = None,
    ) -> AssistantMessageEventStream:
        transcript = normalize_context(context)

        async def setup():
            provider = self._require_chat_provider(model)
            if provider.stream is None:
                raise ModelsError("stream", f"Provider {model.provider} does not support streaming")
            request_model, request_options = await self._apply_auth(model, options, StreamOptions)
            return provider.stream(request_model, transcript, request_options)

        return lazy_stream(model, setup)

    async def complete(
        self,
        model: Model,
        context: Context,
        options: Optional[StreamOptions] = None,
    ) -> AssistantMessage:
        return await self.stream(model, context, options).result()

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: Optional[SimpleStreamOptions] = None,
    ) -> AssistantMessageEventStream:
        transcript = normalize_context(context)

        async def setup():
            provider = self._require_chat_provider(model)
            if provider.stream_simple is None:
                raise ModelsError("stream", f"Provider {model.provider} does not support streaming")
            request_model, request_options = await self._apply_auth(model, options, SimpleStreamOptions)
            return provider.stream_simple(request_model, transcript, request_options)

        return lazy_stream(model, setup)

    async def complete_simple(
        self,
        model: Model,
        context: Context,
        options: Optional[SimpleStreamOptions] = None,
    ) -> AssistantMessage:
        return await self.stream_simple(model, context, options).result()

    def stream_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: Optional[DeferredFetchOptions] = None,
    ) -> AssistantMessageEventStream:
        async def setup():
            provider = self._require_chat_provider(model)
            if not provider.fetch_deferred:
                raise ModelsError("provider", f"Provider {model.provider} does not support deferred responses")
            request_model, request_options = await self._apply_auth(model, options, DeferredFetchOptions)
            return provider.fetch_deferred(request_model, handle, request_options)

        return lazy_stream(model, setup)

    async def fetch_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: Optional[DeferredFetchOptions] = None,
    ) -> AssistantMessage:
        return await self.stream_deferred(model, handle, options).result()

    async def cancel_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: Optional[DeferredCancelOptions] = None,
    ) -> None:
        provider = self._require_chat_provider(model)
        if not provider.cancel_deferred:
            raise ModelsError("provider", f"Provider {model.provider} does not support deferred responses")
        request_model, request_options = await self._apply_auth(model, options, DeferredCancelOptions)
        await provider.cancel_deferred(request_model, handle, request_options)

    async def generate_images(
        self,
        model: ImageModel,
        context: ImagesContext,
        options: Optional[ProviderRequestOptions] = None,
    ) -> AssistantImages:
        """Generate images through the owning provider. Never raises: unknown providers,
        unconfigured auth, and providers without image support return an error result."""
        try:
            if not is_model_type(model, "image"):
                raise ModelsError("provider", f"Model {model.id} is not an image model")
            provider = self._require_provider(model)
            if not provider.generate_images:
                raise ModelsError("provider", f"Provider {model.provider} does not support image generation")
            request_model, request_options = await self._apply_auth(model, options)
            return await provider.generate_images(request_model, context, request_options)
        except Exception as error:
            return image_error_result(model, error, bool(options and options.signal and options.signal.aborted))

    async def classify(
        self,
        model: ClassifierModel,
        context: ClassifierContext,
        options: Optional[ProviderRequestOptions] = None,
    ) -> ClassifierResult:
        """Classify structured state through the owning provider. Never raises."""
        try:
            if not is_model_type(model, "classifier"):
                raise ModelsError("provider", f"Model {model.id} is not a classifier model")
            provider = self._require_provider(model)
            if not provider.classify:
                raise ModelsError("provider", f"Provider {model.provider} does not support classification")
            request_model, request_options = await self._apply_auth(model, options)
            return await provider.classify(request_model, context, request_options)
        except Exception as error:
            return classifier_error_result(model, error, bool(options and options.signal and options.signal.aborted))


async def _return_credential(credential: Credential) -> Credential:
    return credential


def create_models(options: Optional[CreateModelsOptions] = None) -> Models:
    return Models(options)


# ---------------------------------------------------------------------------
# create_provider
# ---------------------------------------------------------------------------


class CreateProviderOptions(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    name: Optional[str] = None
    base_url: Optional[str] = None
    headers: Optional[ProviderHeaders] = None
    auth: ProviderAuth
    models: List[AnyModel]
    fetch_models: Optional[Callable[[RefreshModelsContext], Awaitable[List[AnyModel]]]] = None
    filter_models: Optional[Callable[[Sequence[Model], Optional[Credential]], Sequence[Model]]] = None
    filter_all_models: Optional[Callable[[Sequence[AnyModel], Optional[Credential]], Sequence[AnyModel]]] = None
    api: Optional[Union[ProviderStreams, Dict[str, ProviderStreams]]] = None
    images: Optional[Dict[str, Any]] = None
    classifiers: Optional[Dict[str, Any]] = None


def create_provider(input: CreateProviderOptions) -> Provider:
    """Build a provider from parts.

    A single `api` streams all chat models; an `api` dict dispatches on
    `model.api`, and a model whose api has no entry produces a stream error.
    At least one concrete implementation across api/images/classifiers is required.
    """
    single: Optional[ProviderStreams] = input.api if isinstance(input.api, ProviderStreams) else None
    by_api: Optional[Dict[str, ProviderStreams]] = (
        None if single or not input.api else dict(input.api)  # type: ignore[arg-type]
    )
    images = input.images or {}
    classifiers = input.classifiers or {}

    streams = [single] if single else list((by_api or {}).values())
    if len(streams) == 0 and len(images) == 0 and len(classifiers) == 0:
        raise ValueError(f'Provider {input.id}: at least one of "api", "images", or "classifiers" is required.')

    baseline_models: List[AnyModel] = list(input.models)
    dynamic_models: List[AnyModel] = []

    def current_models() -> List[AnyModel]:
        merged: Dict[tuple[str, str], AnyModel] = {(get_model_type(m), m.id): m for m in baseline_models}
        for model in dynamic_models:
            merged[(get_model_type(model), model.id)] = model
        return list(merged.values())

    def api_for(model: Model) -> Optional[ProviderStreams]:
        return single if single else (by_api or {}).get(model.api)

    def dispatch(model: Model, run: Callable[[ProviderStreams], AssistantMessageEventStream]) -> AssistantMessageEventStream:
        streams_for_model = api_for(model)
        if streams_for_model is None:
            async def fail():
                raise ModelsError("stream", f'Provider {input.id} has no API implementation for "{model.api}"')

            return lazy_stream(model, fail)
        return run(streams_for_model)

    async def refresh_models(context: RefreshModelsContext) -> None:
        nonlocal dynamic_models
        if context.stored:
            restored = [m for m in context.stored.models if m.provider == input.id]

            def apply_restored() -> None:
                nonlocal dynamic_models
                dynamic_models = restored

            if not await context.publish(ModelsPublication(update=apply_restored)):
                return
        if not context.allow_network or context.signal.aborted or input_fetch_models is None:
            return
        fetched = await input_fetch_models(context)
        if context.signal.aborted:
            return
        refreshed = [m for m in fetched if _has_known_model_type(m)]

        def apply_refreshed() -> None:
            nonlocal dynamic_models
            dynamic_models = refreshed

        await context.publish(
            ModelsPublication(
                persist=ModelsStoreEntry(models=refreshed, checked_at=int(time.time() * 1000)),
                update=apply_refreshed,
            )
        )

    input_fetch_models = input.fetch_models

    provider = Provider(
        id=input.id,
        name=input.name or input.id,
        base_url=input.base_url,
        headers=input.headers,
        auth=input.auth,
        get_models=lambda: [m for m in current_models() if is_model_type(m, "chat")],
        get_all_models=current_models,
        refresh_models=refresh_models if input_fetch_models else None,
        filter_models=input.filter_models,
        filter_all_models=input.filter_all_models,
        stream=lambda model, context, options: dispatch(model, lambda s: s.stream(model, context, options)),
        stream_simple=lambda model, context, options: dispatch(model, lambda s: s.stream_simple(model, context, options)),
    )

    if any(s.fetch_deferred for s in streams):
        def fetch_deferred(model, handle, options):
            async def setup():
                implementation = api_for(model)
                if not implementation or not implementation.fetch_deferred:
                    raise ModelsError(
                        "provider", f'Provider {input.id} does not support deferred responses for "{model.api}"'
                    )
                return implementation.fetch_deferred(model, handle, options)

            return lazy_stream(model, setup)

        provider.fetch_deferred = fetch_deferred

    if any(s.cancel_deferred for s in streams):
        async def cancel_deferred(model, handle, options):
            implementation = api_for(model)
            if not implementation or not implementation.cancel_deferred:
                raise ModelsError("provider", f'Provider {input.id} cannot cancel deferred responses for "{model.api}"')
            await implementation.cancel_deferred(model, handle, options)

        provider.cancel_deferred = cancel_deferred

    if images:
        async def generate_images(model, context, options):
            implementation = images.get(model.api)
            if not implementation:
                return image_error_result(
                    model,
                    ModelsError("provider", f'Provider {input.id} has no image generation implementation for "{model.api}"'),
                )
            return await implementation(model, context, options)

        provider.generate_images = generate_images

    if classifiers:
        async def classify(model, context, options):
            implementation = classifiers.get(model.api)
            if not implementation:
                return classifier_error_result(
                    model,
                    ModelsError("provider", f'Provider {input.id} has no classifier implementation for "{model.api}"'),
                )
            return await implementation(model, context, options)

        provider.classify = classify

    return provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def has_api(model: AnyModel, api: Api) -> bool:
    """Runtime-checked narrowing for dynamically looked-up models."""
    return is_model_type(model, "chat") and model.api == api


def calculate_cost(model: AnyModel, usage: Usage) -> Any:
    """Compute usage.cost in place from model rates, honoring pricing tiers."""
    input_tokens = usage.input + usage.cache_read + usage.cache_write
    rates: ModelCostRates = model.cost
    matched_threshold = -1
    for tier in model.cost.tiers or []:
        if input_tokens > tier.input_tokens_above and tier.input_tokens_above > matched_threshold:
            rates = tier
            matched_threshold = tier.input_tokens_above

    # Anthropic charges 2x base input for 1h cache writes.
    long_write = usage.cache_write1h or 0
    short_write = usage.cache_write - long_write
    usage.cost.input = (rates.input / 1_000_000) * usage.input
    usage.cost.output = (rates.output / 1_000_000) * usage.output
    usage.cost.cache_read = (rates.cache_read / 1_000_000) * usage.cache_read
    usage.cost.cache_write = (rates.cache_write * short_write + rates.input * 2 * long_write) / 1_000_000
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    return usage.cost


_EXTENDED_THINKING_LEVELS: List[ModelThinkingLevel] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"]


def get_supported_thinking_levels(model: Model) -> List[ModelThinkingLevel]:
    if not model.reasoning:
        return ["off"]

    levels: List[ModelThinkingLevel] = []
    for level in _EXTENDED_THINKING_LEVELS:
        mapped = (model.thinking_level_map or {}).get(level, "__missing__")
        if mapped is None:
            continue
        if level in ("xhigh", "max") and mapped == "__missing__":
            continue
        levels.append(level)  # type: ignore[arg-type]
    return levels


def clamp_thinking_level(model: Model, level: ModelThinkingLevel) -> ModelThinkingLevel:
    available = get_supported_thinking_levels(model)
    if level in available:
        return level
    if level not in _EXTENDED_THINKING_LEVELS:
        return available[0] if available else "off"
    requested_index = _EXTENDED_THINKING_LEVELS.index(level)
    for candidate in _EXTENDED_THINKING_LEVELS[requested_index:]:
        if candidate in available:
            return candidate
    for candidate in reversed(_EXTENDED_THINKING_LEVELS[:requested_index]):
        if candidate in available:
            return candidate
    return available[0] if available else "off"


def models_are_equal(a: Optional[AnyModel], b: Optional[AnyModel]) -> bool:
    """Check if two models are equal by comparing their type, id, and provider."""
    if a is None or b is None:
        return False
    return get_model_type(a) == get_model_type(b) and a.id == b.id and a.provider == b.provider


def image_error_result(model: ImageModel, error: BaseException, aborted: bool = False) -> AssistantImages:
    from .types import AssistantImages

    return AssistantImages(
        api=model.api,
        provider=model.provider,
        model=model.id,
        output=[],
        stop_reason="aborted" if aborted else "error",
        error_message=str(error),
        timestamp=int(time.time() * 1000),
    )


def classifier_error_result(model: ClassifierModel, error: BaseException, aborted: bool = False) -> ClassifierResult:
    return ClassifierResult(
        api=model.api,
        provider=model.provider,
        model=model.id,
        answers={},
        stop_reason="aborted" if aborted else "error",
        error_message=str(error),
        timestamp=int(time.time() * 1000),
    )

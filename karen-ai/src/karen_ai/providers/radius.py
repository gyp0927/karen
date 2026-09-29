"""Radius gateway provider with a persisted, dynamically refreshed catalog.

Mirroring providers/radius.ts. Unlike the catalog-backed providers this builds
`Provider` directly: refresh restores the stored catalog, imports catalogs
cached on OAuth credentials (legacy interop), then fetches `/v1/config`.
"""

from __future__ import annotations

import time
from typing import List, Optional

from ..api import pi_messages_api
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import flatten_chat_model_catalog
from ..models import ModelsPublication, Provider, RefreshModelsContext
from ..models_store import ModelsStoreEntry
from ..types import Model
from .radius_config import (
    DEFAULT_RADIUS_GATEWAY,
    get_radius_models,
    get_radius_models_from_config,
    load_radius_gateway_config,
    normalize_radius_gateway_url,
)


def radius_provider(
    *,
    id: str = "radius",
    name: str = "Radius",
    gateway: Optional[str] = None,
) -> Provider:
    gateway = normalize_radius_gateway_url(gateway or DEFAULT_RADIUS_GATEWAY)
    baseline_models: List[Model] = (
        [model.model_copy(update={"provider": id}) for model in flatten_chat_model_catalog("radius").values()]
        if gateway == normalize_radius_gateway_url(DEFAULT_RADIUS_GATEWAY)
        else []
    )
    dynamic_models: List[Model] = get_radius_models(id, None)
    streams = pi_messages_api()

    from ..auth.oauth import create_radius_oauth

    def get_models() -> List[Model]:
        merged = list(baseline_models)
        for model in dynamic_models:
            index = next((i for i, entry in enumerate(merged) if entry.id == model.id), -1)
            if index >= 0:
                merged[index] = model
            else:
                merged.append(model)
        return merged

    async def refresh_models(context: RefreshModelsContext) -> None:
        nonlocal dynamic_models

        stored = context.stored
        if stored:
            restored = [m for m in stored.models if m.provider == id]

            def apply_restored() -> None:
                nonlocal dynamic_models
                dynamic_models = restored  # type: ignore[assignment]

            if not await context.publish(ModelsPublication(update=apply_restored)):
                return

        # Import catalogs cached by the pre-ModelsStore Radius implementation.
        if not stored and context.credential is not None and context.credential.type == "oauth":
            legacy = get_radius_models(id, context.credential)  # type: ignore[arg-type]
            if legacy:

                def apply_legacy() -> None:
                    nonlocal dynamic_models
                    dynamic_models = legacy

                published = await context.publish(
                    ModelsPublication(
                        persist=ModelsStoreEntry(models=legacy, checked_at=int(time.time() * 1000)),
                        update=apply_legacy,
                    )
                )
                if not published:
                    return

        if not context.allow_network or context.signal.aborted:
            return
        credential = context.credential
        api_key = credential.access if credential and credential.type == "oauth" else getattr(credential, "key", None)
        config = await load_radius_gateway_config(gateway, api_key, context.signal)
        if context.signal.aborted:
            return
        refreshed = get_radius_models_from_config(id, config)

        def apply_refreshed() -> None:
            nonlocal dynamic_models
            dynamic_models = refreshed

        await context.publish(
            ModelsPublication(
                persist=ModelsStoreEntry(models=refreshed, checked_at=int(time.time() * 1000)),
                update=apply_refreshed,
            )
        )

    return Provider(
        id=id,
        name=name,
        auth=ProviderAuth(
            api_key=env_api_key_auth("Radius API key", ["RADIUS_API_KEY"]),
            oauth=create_radius_oauth(name, gateway),
        ),
        get_models=get_models,
        get_all_models=get_models,
        refresh_models=refresh_models,
        stream=streams.stream,
        stream_simple=streams.stream_simple,
    )

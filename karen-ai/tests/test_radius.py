"""Radius provider dynamic catalog tests."""

import asyncio
import time

import httpx
import respx

from karen_ai.abort import AbortSignal
from karen_ai.auth.types import OAuthCredential
from karen_ai.models import ModelsPublication, RefreshModelsContext
from karen_ai.model_catalog import flatten_chat_model_catalog
from karen_ai.models_store import ModelsStoreEntry
from karen_ai.providers.radius import radius_provider
from karen_ai.providers.radius_config import (
    DEFAULT_RADIUS_GATEWAY,
    get_radius_models,
    get_radius_models_from_config,
    normalize_radius_gateway_url,
)

GATEWAY = "https://radius.test"


def _config(*model_ids: str) -> dict:
    return {
        "baseUrl": f"{GATEWAY}/v1",
        "models": [
            {
                "id": model_id,
                "name": model_id.title(),
                "reasoning": False,
                "input": ["text"],
                "cost": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 128000,
                "maxTokens": 64000,
            }
            for model_id in model_ids
        ],
    }


def _context(credential=None, stored=None, allow_network=True, publications=None):
    async def publish(publication: ModelsPublication) -> bool:
        if publications is not None:
            publications.append(publication)
        if publication.update:
            publication.update()
        return True

    return RefreshModelsContext(
        credential=credential,
        stored=stored,
        publish=publish,
        allow_network=allow_network,
        signal=AbortSignal(),
    )


def test_normalize_gateway_url():
    assert normalize_radius_gateway_url("radius.pi.dev") == "https://radius.pi.dev"
    assert normalize_radius_gateway_url("https://radius.pi.dev/") == "https://radius.pi.dev"
    assert normalize_radius_gateway_url("http://localhost:8787/") == "http://localhost:8787"


def test_models_from_config():
    models = get_radius_models_from_config("radius", _config("a", "b"))
    assert [m.id for m in models] == ["a", "b"]
    assert all(m.api == "pi-messages" and m.provider == "radius" for m in models)
    assert all(m.base_url == f"{GATEWAY}/v1" for m in models)


def test_models_from_oauth_credential_extra():
    credential = OAuthCredential(refresh="r", access="a", expires=0, gatewayConfig=_config("legacy-1"))
    models = get_radius_models("radius", credential)
    assert [m.id for m in models] == ["legacy-1"]
    assert get_radius_models("radius", None) == []
    malformed = OAuthCredential(refresh="r", access="a", expires=0, gatewayConfig={"baseUrl": 42})
    assert get_radius_models("radius", malformed) == []


def test_baseline_catalog_for_default_gateway():
    provider = radius_provider()
    # The default gateway serves the vendored radius.json catalog.
    assert len(provider.get_models()) == len(flatten_chat_model_catalog("radius"))
    # Custom gateways start empty (their catalog arrives via refresh).
    custom = radius_provider(gateway="radius.test")
    assert custom.get_models() == []


def test_refresh_fetches_and_merges():
    async def main():
        provider = radius_provider(gateway=GATEWAY)
        publications = []
        with respx.mock:
            route = respx.get(f"{GATEWAY}/v1/config").mock(
                return_value=httpx.Response(200, json=_config("dynamic-1", "dynamic-2"))
            )
            context = _context(publications=publications)
            await provider.refresh_models(context)

        assert route.calls[0].request.headers["accept"] == "application/json"
        assert "authorization" not in route.calls[0].request.headers
        assert [m.id for m in provider.get_models()] == ["dynamic-1", "dynamic-2"]
        persisted = [p.persist for p in publications if isinstance(p.persist, ModelsStoreEntry)]
        assert persisted and {m.id for m in persisted[-1].models} == {"dynamic-1", "dynamic-2"}

    asyncio.run(main())


def test_refresh_sends_oauth_access_token():
    async def main():
        provider = radius_provider(gateway=GATEWAY)
        credential = OAuthCredential(refresh="r", access="tok-1", expires=int(time.time() * 1000) + 60_000)
        with respx.mock:
            route = respx.get(f"{GATEWAY}/v1/config").mock(return_value=httpx.Response(200, json=_config("m")))
            await provider.refresh_models(_context(credential=credential, publications=[]))

        assert route.calls[0].request.headers["authorization"] == "Bearer tok-1"

    asyncio.run(main())


def test_refresh_restores_stored_without_network():
    async def main():
        provider = radius_provider(gateway=GATEWAY)
        stored_models = get_radius_models_from_config("radius", _config("stored-1"))
        stored = ModelsStoreEntry(models=stored_models, checked_at=1)
        with respx.mock:
            route = respx.get(f"{GATEWAY}/v1/config")
            await provider.refresh_models(_context(stored=stored, allow_network=False, publications=[]))

        assert route.call_count == 0
        assert [m.id for m in provider.get_models()] == ["stored-1"]

    asyncio.run(main())


def test_refresh_imports_legacy_credential_catalog():
    async def main():
        provider = radius_provider(gateway=GATEWAY)
        credential = OAuthCredential(
            refresh="r", access="a", expires=0, gatewayConfig=_config("legacy-9")
        )
        with respx.mock:
            respx.get(f"{GATEWAY}/v1/config").mock(return_value=httpx.Response(200, json=_config("fresh-1")))
            await provider.refresh_models(_context(credential=credential, publications=[]))

        # The legacy import is overlaid by the network refresh.
        assert [m.id for m in provider.get_models()] == ["fresh-1"]

    asyncio.run(main())

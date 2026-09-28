"""Tests for the Models registry: auth resolution, cost, thinking levels, dispatch."""

import asyncio

import pytest

from karen_ai import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthResult,
    Context,
    ModelAuth,
    ModelsError,
    ProviderAuth,
    Usage,
    UserMessage,
    calculate_cost,
    clamp_thinking_level,
    create_models,
    get_supported_thinking_levels,
)
from karen_ai.providers import faux_assistant_message, faux_model, register_faux_provider


def _models_with_faux(responses=None):
    models = create_models()
    registration = register_faux_provider(responses=responses or [])
    models.set_provider(registration.provider)
    return models, registration


def test_calculate_cost_basic():
    model = faux_model()
    model.cost.input = 1.0  # $/M
    model.cost.output = 2.0
    model.cost.cache_read = 0.1
    model.cost.cache_write = 1.25
    usage = Usage(input=1_000_000, output=500_000, cache_read=1_000_000, cache_write=0, total_tokens=2_500_000)
    calculate_cost(model, usage)
    assert usage.cost.input == pytest.approx(1.0)
    assert usage.cost.output == pytest.approx(1.0)
    assert usage.cost.cache_read == pytest.approx(0.1)
    assert usage.cost.total == pytest.approx(2.1)


def test_calculate_cost_1h_cache_write_double_rate():
    model = faux_model()
    model.cost.input = 2.0
    model.cost.cache_write = 2.5
    usage = Usage(cache_write=1000, cache_write1h=400)
    calculate_cost(model, usage)
    # 600 short writes at cacheWrite rate + 400 long writes at 2x input rate.
    expected = (2.5 * 600 + 2.0 * 2 * 400) / 1_000_000
    assert usage.cost.cache_write == pytest.approx(expected)


def test_thinking_levels():
    model = faux_model(reasoning=True)
    assert get_supported_thinking_levels(model) == ["off", "minimal", "low", "medium", "high"]
    assert clamp_thinking_level(model, "xhigh") == "high"
    assert clamp_thinking_level(model, "low") == "low"

    model.thinking_level_map = {"off": "none", "high": None, "xhigh": "max-effort"}
    levels = get_supported_thinking_levels(model)
    assert "high" not in levels
    assert "xhigh" in levels  # explicit mapping enables extended levels
    assert clamp_thinking_level(model, "high") == "xhigh"

    non_reasoning = faux_model(reasoning=False)
    assert get_supported_thinking_levels(non_reasoning) == ["off"]
    assert clamp_thinking_level(non_reasoning, "high") == "off"


def test_complete_simple_roundtrip():
    async def main():
        models, registration = _models_with_faux([faux_assistant_message("Hi there!")])
        model = registration.get_model()
        message = await models.complete_simple(model, Context(messages=[UserMessage(content="hello", timestamp=1)]))
        assert message.stop_reason == "stop"
        assert message.content[0].text == "Hi there!"

    asyncio.run(main())


def test_stream_events_flow_through_protocol():
    async def main():
        models, registration = _models_with_faux([faux_assistant_message("abcdef")])
        model = registration.get_model()
        stream = models.stream_simple(model, Context(messages=[UserMessage(content="hi", timestamp=1)]))
        events = [event async for event in stream]
        types = [e.type for e in events]
        assert types[0] == "start"
        assert "text_start" in types
        assert "text_delta" in types
        assert "text_end" in types
        assert types[-1] == "done"
        text = "".join(e.delta for e in events if e.type == "text_delta")
        assert text == "abcdef"

    asyncio.run(main())


def test_stream_requires_auth_configuration():
    async def main():
        models = create_models()
        registration = register_faux_provider(provider_id="needy")
        # Override auth to always report "not configured".
        registration.provider.auth = ProviderAuth(
            api_key=ApiKeyAuth(name="needy", resolve=lambda input: _never_configured(input))
        )
        models.set_provider(registration.provider)

        stream = models.stream_simple(
            registration.get_model(), Context(messages=[UserMessage(content="hi", timestamp=1)])
        )
        events = [event async for event in stream]
        assert events[-1].type == "error"
        failure = await stream.result()
        assert failure.stop_reason == "error"
        assert "not configured" in failure.error_message

    asyncio.run(main())


async def _never_configured(input):
    return None


def test_unknown_provider_stream_errors():
    async def main():
        models = create_models()
        model = faux_model(provider="ghost")
        stream = models.stream_simple(model, Context(messages=[UserMessage(content="hi", timestamp=1)]))
        failure = await stream.result()
        assert failure.stop_reason == "error"
        assert "Unknown provider" in failure.error_message

    asyncio.run(main())


def test_get_available_filters_unconfigured_providers():
    async def main():
        models, registration = _models_with_faux()
        registration.provider.auth = ProviderAuth(
            api_key=ApiKeyAuth(name="faux", resolve=lambda input: _env_key_auth(input))
        )
        available = await models.get_available()
        assert [m.id for m in available] == [registration.get_model().id]

    asyncio.run(main())


async def _env_key_auth(input):
    return AuthResult(auth=ModelAuth(api_key="x"), source="test")


def test_stored_credential_wins_over_ambient():
    async def main():
        from karen_ai import InMemoryCredentialStore

        store = InMemoryCredentialStore()
        await store.modify("faux", lambda current: _store_key(current))

        from karen_ai import CreateModelsOptions

        models = create_models(CreateModelsOptions(credentials=store))
        registration = register_faux_provider()
        seen_keys = []

        async def resolve(input):
            seen_keys.append(input.credential.key if input.credential else None)
            return AuthResult(auth=ModelAuth(api_key=input.credential.key if input.credential else None))

        registration.provider.auth = ProviderAuth(api_key=ApiKeyAuth(name="faux", resolve=resolve))
        models.set_provider(registration.provider)

        auth = await models.get_auth("faux")
        assert auth.auth.api_key == "stored-key"
        assert seen_keys[-1] == "stored-key"

    asyncio.run(main())


async def _store_key(current):
    return ApiKeyCredential(key="stored-key")

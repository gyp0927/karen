"""Deprecated ambient API surface (compat) tests."""

import asyncio
import json

import httpx
import pytest
import respx

from karen_ai import Context, UserMessage, normalize_context
from karen_ai.api import legacy_aliases
from karen_ai.compat import (
    ApiProvider,
    complete,
    complete_simple,
    get_api_provider,
    get_api_providers,
    get_model,
    get_models,
    get_providers,
    register_api_provider,
    register_faux_provider,
    reset_api_providers,
    stream,
    stream_simple,
    unregister_api_providers,
)
from karen_ai.env_api_keys import (
    AMBIENT_AUTH_MARKER,
    get_env_api_key,
    get_api_key_env_vars,
    find_env_keys,
)
from karen_ai.event_stream import AssistantMessageEventStream
from karen_ai.providers import faux_assistant_message, faux_model, faux_text
from karen_ai.types import DoneEvent, SimpleStreamOptions, StartEvent, StreamOptions, TextDeltaEvent

BUILTIN_APIS = {
    "anthropic-messages",
    "openai-completions",
    "openai-responses",
    "openai-codex-responses",
    "azure-openai-responses",
    "google-generative-ai",
    "google-vertex",
    "mistral-conversations",
    "bedrock-converse-stream",
    "pi-messages",
}


def _context():
    return Context(messages=[UserMessage(content="hi", timestamp=1)])


def _echo_stream(seen):
    """A trivial api implementation that records the options it was handed."""

    def do_stream(model, context, options=None):
        seen.append(options)
        event_stream = AssistantMessageEventStream()

        async def run():
            message = faux_assistant_message("ok", api=model.api, provider=model.provider, model=model.id)
            event_stream.push(StartEvent(partial=message))
            event_stream.push(DoneEvent(reason="stop", message=message))
            event_stream.end()

        asyncio.get_running_loop().create_task(run())
        return event_stream

    return do_stream


def test_builtin_api_providers_are_registered():
    assert {provider.api for provider in get_api_providers()} >= BUILTIN_APIS
    assert all(get_api_provider(api) is not None for api in BUILTIN_APIS)

    reset_api_providers()
    assert all(get_api_provider(api) is not None for api in BUILTIN_APIS)


def test_static_catalog_reads():
    model = get_model("anthropic", "claude-sonnet-5")
    assert model is not None and model.provider == "anthropic"
    assert get_model("anthropic", "does-not-exist") is None
    assert any(m.id == "claude-sonnet-5" for m in get_models("anthropic"))
    assert len(get_providers()) == 42


def test_registry_dispatch_and_unregister():
    async def main():
        seen = []
        register_api_provider(
            ApiProvider(api="custom-test", stream=_echo_stream(seen), stream_simple=_echo_stream(seen)),
            "test-source",
        )
        model = faux_model(provider="custom-provider", api="custom-test")

        message = await complete(model, _context(), StreamOptions())
        assert message.stop_reason == "stop"
        assert len(seen) == 1

        unregister_api_providers("test-source")
        assert get_api_provider("custom-test") is None

        # pi-ai resolves the api implementation synchronously and throws.
        with pytest.raises(ValueError, match="No API provider registered for api: custom-test"):
            stream(model, _context())

    asyncio.run(main())


def test_api_mismatch_is_rejected():
    async def main():
        seen = []
        register_api_provider(
            ApiProvider(api="custom-test", stream=_echo_stream(seen), stream_simple=_echo_stream(seen)),
            "test-source",
        )
        provider = get_api_provider("custom-test")
        mismatched = faux_model(provider="custom-provider", api="other-api")

        with pytest.raises(ValueError, match="Mismatched api: other-api expected custom-test"):
            provider.stream(mismatched, normalize_context(_context()))

        unregister_api_providers("test-source")

    asyncio.run(main())


def test_env_api_key_injection(monkeypatch):
    async def main():
        monkeypatch.setenv("GROQ_API_KEY", "env-key-1")
        seen = []
        register_api_provider(
            ApiProvider(api="custom-test", stream=_echo_stream(seen), stream_simple=_echo_stream(seen)),
            "test-source",
        )
        model = faux_model(provider="groq", api="custom-test")

        await complete(model, _context())
        assert seen[-1].api_key == "env-key-1"

        # An explicit key always wins over the environment.
        await complete(model, _context(), StreamOptions(api_key="explicit"))
        assert seen[-1].api_key == "explicit"

        # A scoped env override wins over the process environment.
        monkeypatch.delenv("GROQ_API_KEY")
        await complete(model, _context(), StreamOptions(env={"GROQ_API_KEY": "scoped"}))
        assert seen[-1].api_key == "scoped"

        unregister_api_providers("test-source")

    asyncio.run(main())


def test_env_api_key_resolution(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_OAUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    assert get_api_key_env_vars("github-copilot") == ["COPILOT_GITHUB_TOKEN"]
    assert get_api_key_env_vars("unknown-provider") is None
    assert get_env_api_key("openai") is None

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_env_api_key("openai") == "sk-test"
    assert find_env_keys("openai") == ["OPENAI_API_KEY"]

    # ANTHROPIC_AUTH_TOKEN is reported but never used as an api key.
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-token")
    assert find_env_keys("anthropic") == ["ANTHROPIC_AUTH_TOKEN"]
    assert get_env_api_key("anthropic") is None

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert get_env_api_key("anthropic") == "sk-ant"


def test_ambient_cloud_credentials_report_authenticated(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "dev")
    assert get_env_api_key("amazon-bedrock") == AMBIENT_AUTH_MARKER

    monkeypatch.delenv("AWS_PROFILE")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "id")
    assert get_env_api_key("amazon-bedrock") is None
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    assert get_env_api_key("amazon-bedrock") == AMBIENT_AUTH_MARKER

    # Ambient credentials never replace the bearer-token auth of a model.
    async def main():
        seen = []
        register_api_provider(
            ApiProvider(api="custom-test", stream=_echo_stream(seen), stream_simple=_echo_stream(seen)),
            "test-source",
        )
        model = faux_model(provider="amazon-bedrock", api="custom-test")
        await complete(model, _context())
        # Ambient credentials are not injected as an api key.
        assert seen[-1] is None
        unregister_api_providers("test-source")

    asyncio.run(main())


def test_faux_provider_registration():
    async def main():
        registration = register_faux_provider(api="faux-compat", provider_id="faux-compat")
        try:
            model = registration.get_model()
            assert model is not None and model.api == "faux-compat"
            assert registration.get_pending_response_count() == 0

            registration.set_responses([faux_assistant_message([faux_text("scripted")])])
            assert registration.get_pending_response_count() == 1

            event_stream = stream_simple(model, _context(), SimpleStreamOptions())
            events = [event async for event in event_stream]
            message = await event_stream.result()

            assert message.stop_reason == "stop"
            assert "".join(event.delta for event in events if event.type == "text_delta") == "scripted"
            assert registration.state.call_count == 1
        finally:
            registration.unregister()

        assert get_api_provider("faux-compat") is None

    asyncio.run(main())


def test_legacy_aliases_match_api_modules():
    from karen_ai.api import anthropic_messages, openai_completions, openai_responses

    assert legacy_aliases.stream_anthropic is anthropic_messages.stream
    assert legacy_aliases.stream_simple_anthropic is anthropic_messages.stream_simple
    assert legacy_aliases.stream_openai_completions is openai_completions.stream
    assert legacy_aliases.stream_simple_openai_responses is openai_responses.stream_simple


def test_builtin_provider_dispatch_injects_env_key(monkeypatch):
    """A builtin catalog model streams through its provider with the env key."""

    async def main():
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
        model = get_model("deepseek", "deepseek-flash")
        assert model is not None

        with respx.mock:
            route = respx.post("https://api.deepseek.com/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    text="".join(
                        f"data: {json.dumps(chunk)}\n\n"
                        for chunk in [
                            {
                                "choices": [{"delta": {"content": "hi"}, "index": 0, "finish_reason": None}],
                                "model": "deepseek-flash",
                            },
                            {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]},
                        ]
                    )
                    + "data: [DONE]\n\n",
                )
            )
            message = await complete(model, _context())

        assert message.stop_reason == "stop"
        assert route.calls[0].request.headers["authorization"] == "Bearer sk-deepseek"

    asyncio.run(main())


def test_complete_simple_uses_simple_defaults():
    async def main():
        seen = []
        register_api_provider(
            ApiProvider(api="custom-test", stream=_echo_stream(seen), stream_simple=_echo_stream(seen)),
            "test-source",
        )
        model = faux_model(provider="custom-provider", api="custom-test")
        message = await complete_simple(model, _context())
        assert message.stop_reason == "stop"
        unregister_api_providers("test-source")

    asyncio.run(main())

"""Built-in provider registry and thin-provider behavior tests."""

import asyncio

from karen_ai.abort import AbortSignal
from karen_ai.auth.types import ApiKeyResolveInput, OAuthCredential
from karen_ai.event_stream import AssistantMessageEventStream
from karen_ai.lazy import ProviderStreams
from karen_ai.models import create_models
from karen_ai.providers import builtin_models, builtin_providers
from karen_ai.providers.cloudflare_stream import resolve_cloudflare_model
from karen_ai.providers.opencode_headers import OPENCODE_SESSION_HEADER, with_opencode_session_header
from karen_ai.types import Model, ModelCost, StreamOptions, UserMessage
from karen_ai.transcript import TranscriptContext


class _EnvCtx:
    def __init__(self, env):
        self._env = env

    async def env(self, name):
        return self._env.get(name)

    async def file_exists(self, path):
        return False


def _resolve(provider, env):
    async def main():
        return await provider.auth.api_key.resolve(ApiKeyResolveInput(ctx=_EnvCtx(env), signal=AbortSignal()))

    return asyncio.run(main())


def test_builtin_providers_cover_pi_ai_registry():
    providers = builtin_providers()
    ids = [p.id for p in providers]
    assert len(ids) == 42
    assert len(set(ids)) == 42
    for expected in [
        "amazon-bedrock", "anthropic", "azure-openai-responses", "deepseek", "github-copilot",
        "google", "google-vertex", "groq", "kimi-coding", "meta", "mistral", "openai",
        "openai-codex", "opencode", "opencode-go", "openrouter", "radius", "typesafe", "xai",
    ]:
        assert expected in ids


def test_every_chat_model_dispatches():
    """One model per (provider, api): dispatch must reach a real streams
    implementation (auth fails fast) instead of the no-api error."""

    async def main():
        problems = []
        for provider in builtin_providers():
            seen_apis = set()
            for model in provider.get_models():
                if model.api in seen_apis:
                    continue
                seen_apis.add(model.api)
                stream = provider.stream(
                    model, TranscriptContext(messages=[UserMessage(content="hi", timestamp=1)]), None
                )
                message = await stream.result()
                assert message.stop_reason == "error", (provider.id, model.id)
                if "no API implementation" in (message.error_message or ""):
                    problems.append((provider.id, model.id, model.api))
        assert problems == []

    asyncio.run(main())


def test_env_api_key_auth_resolution(monkeypatch):
    from karen_ai.providers.groq import groq_provider

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert _resolve(groq_provider(), {}) is None

    result = _resolve(groq_provider(), {"GROQ_API_KEY": "gsk-test"})
    assert result.auth.api_key == "gsk-test"
    assert result.source == "GROQ_API_KEY"


def test_stored_credential_wins(monkeypatch):
    from karen_ai.auth.types import ApiKeyCredential
    from karen_ai.providers.together import together_provider

    async def main():
        provider = together_provider()
        return await provider.auth.api_key.resolve(
            ApiKeyResolveInput(
                ctx=_EnvCtx({"TOGETHER_API_KEY": "ambient"}),
                credential=ApiKeyCredential(key="stored", env={"REGION": "x"}),
                signal=AbortSignal(),
            )
        )

    result = asyncio.run(main())
    assert result.auth.api_key == "stored"
    assert result.env == {"REGION": "x"}
    assert result.source == "stored credential"


def test_oauth_wiring_present():
    from karen_ai.providers.kimi_coding import kimi_coding_provider
    from karen_ai.providers.meta import meta_provider
    from karen_ai.providers.openai_codex import openai_codex_provider
    from karen_ai.providers.xai import xai_provider

    assert kimi_coding_provider().auth.oauth is not None
    assert meta_provider().auth.oauth is not None
    assert xai_provider().auth.oauth is not None
    codex = openai_codex_provider()
    assert codex.auth.oauth is not None
    assert codex.auth.api_key is None  # subscription-only


def test_opencode_session_header_wrapper():
    seen = []

    def fake_stream(model, context, options):
        seen.append(options)
        return AssistantMessageEventStream()

    streams = ProviderStreams(stream=fake_stream, stream_simple=fake_stream)
    wrapped = with_opencode_session_header(streams)

    model = Model(
        id="m", name="m", api="openai-completions", provider="opencode", base_url="https://x",
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )
    context = TranscriptContext(messages=[UserMessage(content="hi", timestamp=1)])

    wrapped.stream(model, context, StreamOptions(session_id="sess-1"))
    assert seen[-1].headers == {OPENCODE_SESSION_HEADER: "sess-1"}

    # An explicit caller header wins; missing session id adds nothing.
    wrapped.stream(model, context, StreamOptions(session_id="sess-1", headers={"X-OpenCode-Session": "caller"}))
    assert seen[-1].headers == {"X-OpenCode-Session": "caller"}
    wrapped.stream(model, context, StreamOptions())
    assert seen[-1].headers is None


def test_cloudflare_model_placeholder_resolution():
    model = Model(
        id="m", name="m", api="openai-completions", provider="cloudflare-workers-ai",
        base_url="https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )
    resolved = resolve_cloudflare_model(model, {"CLOUDFLARE_ACCOUNT_ID": "acc-9"})
    assert resolved.base_url.endswith("/accounts/acc-9/ai/v1")
    # Unknown placeholders stay literal; missing env leaves the model untouched.
    unresolved = resolve_cloudflare_model(model, {})
    assert unresolved is model
    assert "{CLOUDFLARE_ACCOUNT_ID}" in resolve_cloudflare_model(model, None).base_url


def test_cloudflare_auth_requires_key_and_account(monkeypatch):
    from karen_ai.providers.cloudflare_auth import cloudflare_workers_ai_auth

    monkeypatch.delenv("CLOUDFLARE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    auth = cloudflare_workers_ai_auth()

    assert _resolve_auth(auth, {"CLOUDFLARE_API_KEY": "key-only"}) is None
    result = _resolve_auth(auth, {"CLOUDFLARE_API_KEY": "k", "CLOUDFLARE_ACCOUNT_ID": "a"})
    assert result.auth.api_key == "k"
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "a"}


def test_cloudflare_ai_gateway_auth_headers(monkeypatch):
    from karen_ai.providers.cloudflare_auth import cloudflare_ai_gateway_auth

    monkeypatch.delenv("CLOUDFLARE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_GATEWAY_ID", raising=False)
    auth = cloudflare_ai_gateway_auth()

    assert _resolve_auth(auth, {"CLOUDFLARE_API_KEY": "k", "CLOUDFLARE_ACCOUNT_ID": "a"}) is None
    result = _resolve_auth(
        auth, {"CLOUDFLARE_API_KEY": "k", "CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_GATEWAY_ID": "g"}
    )
    assert result.auth.headers == {
        "cf-aig-authorization": "Bearer k",
        "Authorization": None,
        "x-api-key": None,
    }
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_GATEWAY_ID": "g"}


def _resolve_auth(auth, env):
    async def main():
        return await auth.resolve(ApiKeyResolveInput(ctx=_EnvCtx(env), signal=AbortSignal()))

    return asyncio.run(main())


def test_github_copilot_oauth_model_filter():
    from karen_ai.providers.github_copilot import _filter_models

    provider_models = list(__import__("karen_ai.model_catalog", fromlist=["x"]).flatten_chat_model_catalog("github-copilot").values())
    api_key = None
    assert len(_filter_models(provider_models, api_key)) == len(provider_models)

    some = provider_models[:2]
    credential = OAuthCredential(
        refresh="r", access="a", expires=0, availableModelIds=[m.id for m in some]
    )
    filtered = _filter_models(provider_models, credential)
    assert [m.id for m in filtered] == [m.id for m in some]

    malformed = OAuthCredential(refresh="r", access="a", expires=0, availableModelIds="nope")
    assert len(_filter_models(provider_models, malformed)) == len(provider_models)


def test_builtin_models_collection():
    models = builtin_models()
    assert len(models.get_providers()) == 42
    assert models.get_model("groq", "llama-3.3-70b-versatile") is not None
    assert models.get_model("anthropic", "claude-opus-5") is not None

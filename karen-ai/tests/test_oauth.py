"""Tests for the OAuth login flows (auth/oauth/)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import http.client
import json
import time
from typing import Any, Optional
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx

from karen_ai.abort import AbortController
from karen_ai.auth.oauth import (
    anthropic,
    device_code,
    github_copilot,
    kimi_coding,
    meta,
    oauth_page,
    openai_codex,
    openrouter,
    radius,
    xai,
)
from karen_ai.auth.oauth.pkce import generate_pkce
from karen_ai.auth.types import OAuthCredential
from karen_ai.errors import AbortError


class FakeInteraction:
    """AuthInteraction test double: queued answers, recorded events/prompts."""

    def __init__(self, answers=None, signal=None):
        self.signal = signal
        self.events = []
        self.prompts = []
        self._answers = list(answers or [])

    async def prompt(self, prompt):
        self.prompts.append(prompt)
        if self._answers:
            answer = self._answers.pop(0)
            if callable(answer):
                answer = answer(prompt)
            if asyncio.iscoroutine(answer):
                return await answer
            return answer
        await asyncio.Event().wait()  # block until cancelled

    def notify(self, event):
        self.events.append(event)

    def events_of_type(self, type_name: str):
        return [e for e in self.events if e.type == type_name]


def _never() -> Any:
    return lambda _prompt: asyncio.Event().wait()


def _fire_callback(port: int, path: str) -> "tuple[int, str]":
    """Fire a real loopback GET. Uses stdlib http.client because respx's
    pass-through (which only patches httpx) mangles the callback response body."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    response = conn.getresponse()
    body = response.read().decode("utf-8")
    conn.close()
    return response.status, body


async def _wait_for(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


# ---------------------------------------------------------------------------
# PKCE / pages / device-code poller
# ---------------------------------------------------------------------------


def test_generate_pkce_shape():
    verifier, challenge = generate_pkce()
    assert len(verifier) == 43
    assert "=" not in verifier and "=" not in challenge
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected
    v2, c2 = generate_pkce()
    assert (v2, c2) != (verifier, challenge)


def test_oauth_pages_escape_html():
    page = oauth_page.oauth_success_html("Signed in. <script>alert(1)</script>")
    assert "Authentication successful" in page
    assert "&lt;script&gt;" in page
    assert "<script>alert" not in page
    error = oauth_page.oauth_error_html("Failed", 'details "quoted"')
    assert "Authentication failed" in error
    assert "&quot;quoted&quot;" in error


async def test_device_code_poll_complete_after_pending():
    calls = 0

    async def poll():
        nonlocal calls
        calls += 1
        if calls < 2:
            return device_code.poll_pending()
        return device_code.poll_complete("token")

    result = await device_code.poll_oauth_device_code_flow(
        poll=poll, signal=AbortController().signal, interval_seconds=1, expires_in_seconds=30
    )
    assert result == "token"
    assert calls == 2


async def test_device_code_poll_failed_raises_message():
    async def poll():
        return device_code.poll_failed("access denied by user")

    with pytest.raises(ValueError, match="access denied by user"):
        await device_code.poll_oauth_device_code_flow(
            poll=poll, signal=AbortController().signal, interval_seconds=1, expires_in_seconds=30
        )


async def test_device_code_poll_timeout_message():
    async def poll():
        return device_code.poll_pending()

    with pytest.raises(ValueError, match="^Device flow timed out$"):
        await device_code.poll_oauth_device_code_flow(
            poll=poll, signal=AbortController().signal, interval_seconds=5, expires_in_seconds=0.05
        )


async def test_device_code_poll_slow_down_timeout_message():
    async def poll():
        return device_code.poll_slow_down(interval_seconds=1)

    with pytest.raises(ValueError, match="slow_down responses"):
        await device_code.poll_oauth_device_code_flow(
            poll=poll, signal=AbortController().signal, interval_seconds=1, expires_in_seconds=0.05
        )


async def test_device_code_poll_abort_raises_login_cancelled():
    controller = AbortController()
    controller.abort()

    async def poll():
        raise AssertionError("poll should not run")

    with pytest.raises(AbortError, match="Login cancelled"):
        await device_code.poll_oauth_device_code_flow(
            poll=poll, signal=controller.signal, interval_seconds=1, expires_in_seconds=30
        )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_parse_authorization_input():
    parse = anthropic.parse_authorization_input
    assert parse("http://localhost:53692/callback?code=abc&state=xyz") == {"code": "abc", "state": "xyz"}
    assert parse("abc#xyz") == {"code": "abc", "state": "xyz"}
    assert parse("code=abc&state=xyz") == {"code": "abc", "state": "xyz"}
    assert parse("rawcode") == {"code": "rawcode"}
    assert parse("   ") == {}


async def test_anthropic_login_via_browser_callback():
    router = respx.mock(assert_all_mocked=False)
    with router:
        route = router.post(anthropic.TOKEN_URL).mock(
            return_value=httpx.Response(
                200, json={"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 3600}
            )
        )
        interaction = FakeInteraction(answers=[_never()])
        login_task = asyncio.create_task(anthropic.anthropic_oauth.login(interaction))
        await _wait_for(lambda: interaction.events_of_type("auth_url"))
        auth_url = interaction.events_of_type("auth_url")[0].url
        params = parse_qs(urlsplit(auth_url).query)
        state = params["state"][0]
        assert params["client_id"][0] == anthropic.CLIENT_ID
        assert params["code_challenge_method"][0] == "S256"

        status, body = await asyncio.to_thread(_fire_callback, 53692, f"/callback?code=code-123&state={state}")
        assert status == 200
        assert "Anthropic authentication completed" in body

        credential = await asyncio.wait_for(login_task, 5)
        assert credential.access == "at-1"
        assert credential.refresh == "rt-1"
        body = json.loads(route.calls[0].request.content)
        assert body["grant_type"] == "authorization_code"
        assert body["code"] == "code-123"
        assert body["state"] == state
        assert body["redirect_uri"] == anthropic.REDIRECT_URI
        assert body["code_verifier"] == state  # verifier doubles as state
        # 5-minute expiry skew
        now_ms = int(time.time() * 1000)
        assert now_ms + 3600 * 1000 - 300_000 - 5000 < credential.expires <= now_ms + 3600 * 1000 - 300_000 + 5000


async def test_anthropic_login_via_manual_code():
    with respx.mock:
        route = respx.post(anthropic.TOKEN_URL).mock(
            return_value=httpx.Response(
                200, json={"access_token": "at-2", "refresh_token": "rt-2", "expires_in": 60}
            )
        )

        def answer(_prompt):
            auth_url = interaction.events_of_type("auth_url")[0].url
            state = parse_qs(urlsplit(auth_url).query)["state"][0]
            return f"manual-code#{state}"

        interaction = FakeInteraction(answers=[answer])
        credential = await anthropic.anthropic_oauth.login(interaction)
        assert credential.access == "at-2"
        body = json.loads(route.calls[0].request.content)
        assert body["code"] == "manual-code"


async def test_anthropic_manual_code_state_mismatch():
    with respx.mock:
        respx.post(anthropic.TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "a", "refresh_token": "r", "expires_in": 1})
        )
        interaction = FakeInteraction(answers=["somecode#wrong-state"])
        with pytest.raises(ValueError, match="OAuth state mismatch"):
            await anthropic.anthropic_oauth.login(interaction)


async def test_anthropic_refresh():
    with respx.mock:
        route = respx.post(anthropic.TOKEN_URL).mock(
            return_value=httpx.Response(
                200, json={"access_token": "at-new", "refresh_token": "rt-new", "expires_in": 120}
            )
        )
        credential = await anthropic.refresh_anthropic_token("rt-old", None)
        assert credential.access == "at-new"
        body = json.loads(route.calls[0].request.content)
        assert body == {"grant_type": "refresh_token", "client_id": anthropic.CLIENT_ID, "refresh_token": "rt-old"}


async def test_anthropic_to_auth():
    credential = OAuthCredential(access="access-x", refresh="r", expires=1)
    auth = await anthropic.anthropic_oauth.to_auth(credential)
    assert auth.api_key == "access-x"


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------


def test_openrouter_parse_authorization_input():
    parse = openrouter.parse_authorization_input
    assert parse("http://127.0.0.1:1234/oauth/callback/x?code=abc") == "abc"
    assert parse("code=abc&foo=bar") == "abc"
    assert parse("rawcode") == "rawcode"
    assert parse("  ") is None


async def test_openrouter_login_via_callback():
    router = respx.mock(assert_all_mocked=False)
    with router:
        route = router.post(openrouter.TOKEN_URL).mock(return_value=httpx.Response(200, json={"key": "sk-or-1"}))
        interaction = FakeInteraction(answers=[_never()])
        login_task = asyncio.create_task(openrouter.openrouter_oauth.login(interaction))
        await _wait_for(lambda: interaction.events_of_type("auth_url"))
        auth_url = interaction.events_of_type("auth_url")[0].url
        callback_url = parse_qs(urlsplit(auth_url).query)["callback_url"][0]

        split = urlsplit(callback_url)
        status, body = await asyncio.to_thread(
            _fire_callback, split.port, f"{split.path}?code=code-9"
        )
        assert status == 200
        assert "Signed in to OpenRouter" in body

        credential = await asyncio.wait_for(login_task, 5)
        assert credential.access == "sk-or-1"
        assert credential.refresh == ""
        assert credential.expires == 2**53 - 1
        body = json.loads(route.calls[0].request.content)
        assert body["code"] == "code-9"
        assert body["code_challenge_method"] == "S256"
        assert body["code_verifier"]


async def test_openrouter_login_via_manual_paste():
    with respx.mock:
        route = respx.post(openrouter.TOKEN_URL).mock(return_value=httpx.Response(200, json={"key": "sk-or-2"}))

        def answer(_prompt):
            auth_url = interaction.events_of_type("auth_url")[0].url
            callback_url = parse_qs(urlsplit(auth_url).query)["callback_url"][0]
            return f"{callback_url}?code=pasted-code"

        interaction = FakeInteraction(answers=[answer])
        credential = await openrouter.openrouter_oauth.login(interaction)
        assert credential.access == "sk-or-2"
        assert json.loads(route.calls[0].request.content)["code"] == "pasted-code"


async def test_openrouter_exchange_error_detail():
    with respx.mock:
        respx.post(openrouter.TOKEN_URL).mock(
            return_value=httpx.Response(400, json={"error": {"message": "code expired"}})
        )
        with pytest.raises(ValueError, match=r"HTTP 400\): code expired"):
            await openrouter.exchange_authorization_code("bad", "verifier", None)


async def test_openrouter_refresh_is_identity():
    credential = OAuthCredential(access="k", refresh="", expires=1)
    assert await openrouter.openrouter_oauth.refresh(credential, AbortController().signal) is credential


# ---------------------------------------------------------------------------
# OpenAI Codex
# ---------------------------------------------------------------------------


def _make_jwt(payload: dict) -> str:
    def enc(obj: Any) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{enc({'alg': 'none'})}.{enc(payload)}.sig"


def test_codex_decode_jwt_account_id():
    token = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}})
    assert openai_codex._get_account_id(token) == "acct-123"
    assert openai_codex._get_account_id("not-a-jwt") is None
    assert openai_codex._get_account_id(_make_jwt({"sub": "x"})) is None


def test_codex_parse_authorization_input():
    parse = openai_codex.parse_authorization_input
    assert parse("http://localhost:1455/auth/callback?code=c&state=s") == {"code": "c", "state": "s"}
    assert parse("c#s") == {"code": "c", "state": "s"}
    assert parse("code=c") == {"code": "c", "state": None}
    assert parse("raw") == {"code": "raw"}


async def test_codex_login_via_browser_callback():
    access_jwt = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-browser"}})
    with respx.mock:
        route = respx.post(openai_codex.TOKEN_URL).mock(
            return_value=httpx.Response(
                200, json={"access_token": access_jwt, "refresh_token": "rt-codex", "expires_in": 3600}
            )
        )
        interaction2 = FakeInteraction(answers=[openai_codex.BROWSER_LOGIN_METHOD, _never()])
        login_task2 = asyncio.create_task(openai_codex.openai_codex_oauth.login(interaction2))
        await _wait_for(lambda: interaction2.events_of_type("auth_url"))
        auth_url = interaction2.events_of_type("auth_url")[0].url
        params = parse_qs(urlsplit(auth_url).query)
        state = params["state"][0]
        assert params["redirect_uri"][0] == openai_codex.REDIRECT_URI
        assert params["codex_cli_simplified_flow"][0] == "true"

        status, _body = await asyncio.to_thread(
            _fire_callback, 1455, f"/auth/callback?code=codex-code&state={state}"
        )
        assert status == 200

        credential = await asyncio.wait_for(login_task2, 5)
        assert credential.access == access_jwt
        assert (credential.model_extra or {}).get("accountId") == "acct-browser"
        body = route.calls[0].request.content.decode()
        assert "grant_type=authorization_code" in body
        assert "code=codex-code" in body


async def test_codex_login_via_device_code():
    access_jwt = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-device"}})
    with respx.mock:
        respx.post(openai_codex.DEVICE_USER_CODE_URL).mock(
            return_value=httpx.Response(
                200, json={"device_auth_id": "da-1", "user_code": "UC-42", "interval": "1"}
            )
        )
        poll_calls = 0

        def device_token_side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal poll_calls
            poll_calls += 1
            if poll_calls == 1:
                return httpx.Response(403, json={"error": {"code": "deviceauth_authorization_pending"}})
            return httpx.Response(200, json={"authorization_code": "dc-code", "code_verifier": "dc-verifier"})

        respx.post(openai_codex.DEVICE_TOKEN_URL).mock(side_effect=device_token_side_effect)
        exchange_route = respx.post(openai_codex.TOKEN_URL).mock(
            return_value=httpx.Response(
                200, json={"access_token": access_jwt, "refresh_token": "rt-dc", "expires_in": 3600}
            )
        )

        interaction = FakeInteraction(answers=[openai_codex.DEVICE_CODE_LOGIN_METHOD])
        credential = await openai_codex.openai_codex_oauth.login(interaction)
        assert credential.access == access_jwt
        assert (credential.model_extra or {}).get("accountId") == "acct-device"

        device_events = interaction.events_of_type("device_code")
        assert device_events and device_events[0].user_code == "UC-42"
        assert device_events[0].verification_uri == openai_codex.DEVICE_VERIFICATION_URI
        # Device exchange uses the deviceauth redirect URI.
        assert "redirect_uri=" in exchange_route.calls[0].request.content.decode()


async def test_codex_refresh():
    access_jwt = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-r"}})
    with respx.mock:
        route = respx.post(openai_codex.TOKEN_URL).mock(
            return_value=httpx.Response(
                200, json={"access_token": access_jwt, "refresh_token": "rt-new", "expires_in": 10}
            )
        )
        credential = await openai_codex.refresh_openai_codex_token("rt-old", None)
        assert credential.refresh == "rt-new"
        body = route.calls[0].request.content.decode()
        assert "grant_type=refresh_token" in body and "refresh_token=rt-old" in body


async def test_codex_credentials_require_account_id():
    with pytest.raises(ValueError, match="accountId"):
        openai_codex._credentials_from_token(_make_jwt({"sub": "no-claim"}), "r", 1)


# ---------------------------------------------------------------------------
# GitHub Copilot
# ---------------------------------------------------------------------------


def test_copilot_normalize_domain():
    assert github_copilot.normalize_domain("") is None
    assert github_copilot.normalize_domain("  ") is None
    assert github_copilot.normalize_domain("company.ghe.com") == "company.ghe.com"
    assert github_copilot.normalize_domain("https://company.ghe.com/path") == "company.ghe.com"


def test_copilot_base_url_from_token():
    token = "tid=1;exp=2;proxy-ep=proxy.individual.githubcopilot.com;other=3"
    assert github_copilot.get_base_url_from_token(token) == "https://api.individual.githubcopilot.com"
    assert github_copilot.get_base_url_from_token("tid=1") is None
    assert github_copilot.get_github_copilot_base_url(None, "ghe.example.com") == "https://copilot-api.ghe.example.com"
    assert github_copilot.get_github_copilot_base_url(None, None) == github_copilot.INDIVIDUAL_BASE_URL


def test_copilot_parse_model_catalog():
    raw = {
        "data": [
            {"id": "gpt-5", "model_picker_enabled": True, "capabilities": {"supports": {"tool_calls": True}}},
            {"id": "text-only", "capabilities": {"supports": {"tool_calls": False}}, "model_picker_enabled": True},
            {"id": "disabled-model", "model_picker_enabled": True, "policy": {"state": "disabled"}},
            {"id": "gpt-5.1", "model_picker_enabled": True, "policy": {"state": "unconfigured"}},
        ]
    }
    catalog = github_copilot.parse_github_copilot_model_catalog(raw, allow_policy_fallback=False)
    assert catalog["available_model_ids"] == ["gpt-5", "gpt-5.1"]
    assert catalog["policy_model_ids"] == ["gpt-5.1"]

    with pytest.raises(ValueError, match="Invalid Copilot models response"):
        github_copilot.parse_github_copilot_model_catalog({"nope": []}, allow_policy_fallback=False)


def test_copilot_parse_model_catalog_policy_fallback():
    raw = {
        "data": [
            {"id": "gpt-5", "model_picker_enabled": False, "policy": {"state": "enabled"}},
            {"id": "other", "model_picker_enabled": False, "policy": {"state": "unconfigured"}},
        ]
    }
    # No picker-enabled models + fallback allowed → policy "enabled" ids.
    catalog = github_copilot.parse_github_copilot_model_catalog(raw, allow_policy_fallback=True)
    assert catalog["available_model_ids"] == ["gpt-5"]
    # Without fallback, available is empty.
    catalog = github_copilot.parse_github_copilot_model_catalog(raw, allow_policy_fallback=False)
    assert catalog["available_model_ids"] == []


async def test_copilot_login_end_to_end():
    copilot_token = "tid=x;proxy-ep=proxy.individual.githubcopilot.com;exp=999"
    with respx.mock:
        respx.post("https://github.com/login/device/code").mock(
            return_value=httpx.Response(
                200,
                json={
                    "device_code": "dc",
                    "user_code": "UC",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 1,
                    "expires_in": 900,
                },
            )
        )
        respx.post("https://github.com/login/oauth/access_token").mock(
            return_value=httpx.Response(200, json={"access_token": "gh-token"})
        )
        respx.get("https://api.github.com/copilot_internal/v2/token").mock(
            return_value=httpx.Response(200, json={"token": copilot_token, "expires_at": int(time.time()) + 3600})
        )
        respx.get("https://api.individual.githubcopilot.com/models").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "gpt-5", "model_picker_enabled": True, "capabilities": {"supports": {"tool_calls": True}}},
                        {
                            "id": "gpt-5.1",
                            "model_picker_enabled": True,
                            "policy": {"state": "unconfigured"},
                            "capabilities": {"supports": {"tool_calls": True}},
                        },
                    ]
                },
            )
        )
        policy_route = respx.post("https://api.individual.githubcopilot.com/models/gpt-5.1/policy").mock(
            return_value=httpx.Response(200, json={})
        )

        interaction = FakeInteraction(answers=[""])  # blank → github.com
        credential = await github_copilot.github_copilot_oauth.login(interaction)

        assert credential.access == copilot_token
        extra = credential.model_extra or {}
        assert set(extra.get("availableModelIds") or []) == {"gpt-5", "gpt-5.1"}
        # Policy enable was called for the unconfigured known model.
        assert policy_route.called
        assert json.loads(policy_route.calls[0].request.content) == {"state": "enabled"}
        # Device code event surfaced.
        assert interaction.events_of_type("device_code")[0].user_code == "UC"

        auth = await github_copilot.github_copilot_oauth.to_auth(credential)
        assert auth.api_key == copilot_token
        assert auth.base_url == "https://api.individual.githubcopilot.com"


async def test_copilot_login_rejects_invalid_enterprise_domain():
    interaction = FakeInteraction(answers=["http://[::1"])
    with pytest.raises(ValueError, match="Invalid GitHub Enterprise URL/domain"):
        await github_copilot.github_copilot_oauth.login(interaction)


# ---------------------------------------------------------------------------
# Kimi Code
# ---------------------------------------------------------------------------


async def test_kimi_login_and_refresh(monkeypatch):
    host = "https://auth.kimi.test"
    monkeypatch.setenv("KIMI_CODE_OAUTH_HOST", host)
    with respx.mock:
        respx.post(f"{host}/api/oauth/device_authorization").mock(
            return_value=httpx.Response(
                200,
                json={
                    "device_code": "dc",
                    "user_code": "UC-K",
                    "verification_uri": "https://auth.kimi.test/device",
                    "verification_uri_complete": "https://auth.kimi.test/device?code=UC-K",
                    "interval": 1,
                    "expires_in": 900,
                },
            )
        )
        token_calls = 0

        def token_side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal token_calls
            token_calls += 1
            body = request.content.decode()
            if "grant_type=refresh_token" in body:
                return httpx.Response(200, json={"access_token": "at-r", "refresh_token": "rt-r", "expires_in": 100})
            return httpx.Response(200, json={"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 100})

        respx.post(f"{host}/api/oauth/token").mock(side_effect=token_side_effect)

        interaction = FakeInteraction()
        credential = await kimi_coding.kimi_coding_oauth.login(interaction)
        assert credential.access == "at-1"
        event = interaction.events_of_type("device_code")[0]
        assert event.verification_uri == "https://auth.kimi.test/device?code=UC-K"

        refreshed = await kimi_coding.kimi_coding_oauth.refresh(credential, AbortController().signal)
        assert refreshed.access == "at-r"

        auth = await kimi_coding.kimi_coding_oauth.to_auth(refreshed)
        assert auth.headers == {"Authorization": "Bearer at-r"}


async def test_kimi_refresh_unauthorized(monkeypatch):
    host = "https://auth.kimi.test"
    monkeypatch.setenv("KIMI_CODE_OAUTH_HOST", host)
    with respx.mock:
        respx.post(f"{host}/api/oauth/token").mock(
            return_value=httpx.Response(401, json={"error": "invalid_grant", "error_description": "expired"})
        )
        with pytest.raises(ValueError, match="refresh unauthorized"):
            await kimi_coding.refresh_token(host, "dead-refresh", None)


async def test_kimi_device_authorization_rejects_untrusted_uri(monkeypatch):
    host = "https://auth.kimi.test"
    monkeypatch.setenv("KIMI_CODE_OAUTH_HOST", host)
    with respx.mock:
        respx.post(f"{host}/api/oauth/device_authorization").mock(
            return_value=httpx.Response(
                200,
                json={
                    "device_code": "dc",
                    "user_code": "UC",
                    "verification_uri": "javascript:alert(1)",
                    "verification_uri_complete": "https://ok.test/",
                },
            )
        )
        with pytest.raises(ValueError, match="Invalid Kimi Code device authorization response"):
            await kimi_coding._start_device_authorization(host, None)


# ---------------------------------------------------------------------------
# xAI
# ---------------------------------------------------------------------------


async def test_xai_login_and_refresh():
    with respx.mock:
        respx.post(xai.XAI_DEVICE_CODE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "device_code": "dc",
                    "user_code": "UC-X",
                    "verification_uri": "https://auth.x.ai/device",
                    "interval": 1,
                    "expires_in": 900,
                },
            )
        )
        token_calls = 0

        def token_side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal token_calls
            token_calls += 1
            body = request.content.decode()
            if "grant_type=refresh_token" in body:
                # xAI omits refresh_token when not rotated.
                return httpx.Response(200, json={"access_token": "at-new", "expires_in": 3600})
            return httpx.Response(200, json={"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 3600})

        respx.post(xai.XAI_TOKEN_URL).mock(side_effect=token_side_effect)

        interaction = FakeInteraction()
        credential = await xai.xai_oauth.login(interaction)
        assert credential.access == "at-1"
        assert credential.refresh == "rt-1"
        now_ms = int(time.time() * 1000)
        assert credential.expires <= now_ms + 3600 * 1000 - xai.REFRESH_SKEW_MS + 5000

        refreshed = await xai.xai_oauth.refresh(credential, AbortController().signal)
        assert refreshed.access == "at-new"
        assert refreshed.refresh == "rt-1"  # previous refresh token retained


def test_xai_rejects_non_https_verification_uri():
    with pytest.raises(ValueError, match="Untrusted verification URI"):
        xai._validate_verification_uri("http://insecure.example.com/device")
    assert xai._validate_verification_uri("https://auth.x.ai/device") == "https://auth.x.ai/device"


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


async def test_meta_login_mints_api_key():
    with respx.mock:
        respx.post(meta.DEVICE_AUTHORIZATION_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "device_code": "dc",
                    "user_code": "UC-M",
                    "verification_uri": "https://auth.meta.com/device",
                    "interval": 1,
                    "expires_in": 900,
                },
            )
        )
        respx.post(meta.DEVICE_TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "identity-token"})
        )
        mint_route = respx.post(meta.API_KEY_MINT_URL).mock(
            return_value=httpx.Response(200, json={"api_key": "meta-api-key"})
        )

        interaction = FakeInteraction()
        credential = await meta.meta_oauth.login(interaction)
        assert credential.access == "meta-api-key"
        assert credential.refresh == "identity-token"
        now_ms = int(time.time() * 1000)
        assert now_ms + meta.API_KEY_LIFETIME_MS - 5000 < credential.expires <= now_ms + meta.API_KEY_LIFETIME_MS + 5000
        assert mint_route.calls[0].request.headers["Authorization"] == "Bearer identity-token"
        assert mint_route.calls[0].request.headers["x-api-version"] == "1.0.0"

        # Refresh re-mints with the identity token.
        refreshed = await meta.meta_oauth.refresh(credential, AbortController().signal)
        assert refreshed.access == "meta-api-key"
        assert refreshed.refresh == "identity-token"


async def test_meta_mint_session_expired():
    with respx.mock:
        respx.post(meta.API_KEY_MINT_URL).mock(return_value=httpx.Response(401, json={"detail": "bad token"}))
        with pytest.raises(ValueError, match="Meta session expired"):
            await meta.mint_api_key("dead-identity", None)


# ---------------------------------------------------------------------------
# Radius
# ---------------------------------------------------------------------------


def test_radius_normalize_gateway_url():
    assert radius.normalize_radius_gateway_url("radius.pi.dev") == "https://radius.pi.dev"
    assert radius.normalize_radius_gateway_url("https://radius.pi.dev/") == "https://radius.pi.dev"
    assert radius.normalize_radius_gateway_url("http://localhost:8787//") == "http://localhost:8787"


async def test_radius_device_code_login():
    gateway = "https://gateway.test"
    with respx.mock:
        respx.post(f"{gateway}/v1/oauth/device").mock(
            return_value=httpx.Response(
                200,
                json={
                    "device_code": "dc",
                    "user_code": "UC-R",
                    "verification_uri": "https://gateway.test/device",
                    "expires_in": 900,
                    "interval": 1,
                },
            )
        )
        token_calls = 0

        def token_side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal token_calls
            token_calls += 1
            if token_calls == 1:
                return httpx.Response(400, json={"error": "authorization_pending"})
            return httpx.Response(
                200, json={"access_token": "at-r", "refresh_token": "rt-r", "expires_in": 3600, "scope": "gateway"}
            )

        respx.post(f"{gateway}/v1/oauth/token").mock(side_effect=token_side_effect)

        oauth = radius.create_radius_oauth("Radius", gateway)
        interaction = FakeInteraction(answers=[radius.LOGIN_METHOD_DEVICE_CODE])
        credential = await oauth.login(interaction)
        assert credential.access == "at-r"
        assert (credential.model_extra or {}).get("scope") == "gateway"
        assert interaction.events_of_type("device_code")[0].user_code == "UC-R"
        # 60s expiry skew.
        now_ms = int(time.time() * 1000)
        assert credential.expires <= now_ms + 3600 * 1000 - 60_000 + 5000


async def test_radius_browser_login():
    gateway = "https://gateway.test"
    with respx.mock:
        respx.get(f"{gateway}/v1/oauth").mock(
            return_value=httpx.Response(200, json={"authorizationEndpoint": "https://gateway.test/authorize"})
        )
        token_route = respx.post(f"{gateway}/v1/oauth/token").mock(
            return_value=httpx.Response(200, json={"access_token": "at-b", "refresh_token": "rt-b", "expires_in": 60})
        )

        oauth = radius.create_radius_oauth("Radius", gateway)
        interaction = FakeInteraction(answers=[radius.LOGIN_METHOD_BROWSER])
        login_task = asyncio.create_task(oauth.login(interaction))
        await _wait_for(lambda: interaction.events_of_type("auth_url"))
        auth_url = interaction.events_of_type("auth_url")[0].url
        params = parse_qs(urlsplit(auth_url).query)
        assert params["handoff"][0] == "url"
        state = params["state"][0]

        status, body = await asyncio.to_thread(
            _fire_callback, 1456, f"/oauth/callback?code=radius-code&state={state}"
        )
        assert status == 200
        assert "Signed in to Radius" in body

        credential = await asyncio.wait_for(login_task, 5)
        assert credential.access == "at-b"
        body = token_route.calls[0].request.content.decode()
        assert "grant_type=authorization_code" in body and "code=radius-code" in body


def test_radius_oauth_response_error_formatting():
    error = radius.OAuthResponseError(400, "invalid_grant", "token expired", "Radius OAuth token request failed")
    assert str(error) == "Radius OAuth token request failed: invalid_grant: token expired"
    assert error.status == 400
    assert error.oauth_error == "invalid_grant"
    error2 = radius.OAuthResponseError(500, None, None, "Radius OAuth token request failed")
    assert str(error2) == "Radius OAuth token request failed: 500"

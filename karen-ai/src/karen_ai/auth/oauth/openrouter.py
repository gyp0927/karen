"""OpenRouter OAuth PKCE flow, mirroring auth/oauth/openrouter.ts.

OpenRouter exchanges an authorization code for a permanent, user-controlled
API key rather than an expiring access/refresh token pair. The callback is
handled by a one-shot loopback server on an ephemeral port, raced against a
manual prompt so remote/headless sessions can paste the redirect URL when the
browser cannot reach the loopback server.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit

from ...abort import AbortController, AbortSignal
from ...errors import AbortError
from ...utils.provider_env import get_provider_env_value
from ..types import AuthEventAuthUrl, AuthEventProgress, AuthInteraction, AuthPromptManualCode, ModelAuth, OAuthAuth, OAuthCredential
from ._http import post_json, race_with_abort
from .callback_server import CallbackHttpResponse, CallbackWaiter, OAuthCallbackServer, start_oauth_callback_server
from .oauth_page import oauth_error_html, oauth_success_html
from .pkce import generate_pkce

AUTHORIZE_URL = "https://openrouter.ai/auth"
TOKEN_URL = "https://openrouter.ai/api/v1/auth/keys"
LOGIN_TIMEOUT_SECONDS = 5 * 60
TOKEN_EXCHANGE_TIMEOUT_SECONDS = 30.0

# Number.MAX_SAFE_INTEGER
_MAX_SAFE_INTEGER = 2**53 - 1


def _callback_host() -> str:
    return get_provider_env_value("PI_OAUTH_CALLBACK_HOST") or "127.0.0.1"


def parse_authorization_input(input: str) -> Optional[str]:
    value = input.strip()
    if not value:
        return None

    try:
        url = urlsplit(value)
        if url.scheme and url.netloc:
            return dict(parse_qsl(url.query)).get("code")
    except ValueError:
        pass

    if "code=" in value:
        return dict(parse_qsl(value)).get("code")

    return value


def _error_detail(body: Any) -> Optional[str]:
    if not isinstance(body, dict):
        return None
    for key in ("error_description", "message", "error"):
        value = body.get(key)
        if isinstance(value, str):
            return value
    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return None


async def exchange_authorization_code(
    code: str,
    verifier: str,
    signal: Optional[AbortSignal],
) -> OAuthCredential:
    try:
        response = await post_json(
            TOKEN_URL,
            {"code": code, "code_verifier": verifier, "code_challenge_method": "S256"},
            signal,
            timeout=TOKEN_EXCHANGE_TIMEOUT_SECONDS,
        )
    except Exception as error:
        if signal is not None and signal.aborted:
            raise AbortError("Login cancelled") from error
        raise

    try:
        body: Any = response.json()
    except ValueError:
        if response.status_code < 400:
            raise ValueError("OpenRouter OAuth returned invalid JSON")
        body = {}

    if response.status_code >= 400:
        detail = _error_detail(body)
        raise ValueError(
            f"OpenRouter OAuth key exchange failed (HTTP {response.status_code}){f': {detail}' if detail else ''}"
        )

    if not isinstance(body, dict) or not isinstance(body.get("key"), str) or not body["key"]:
        raise ValueError('OpenRouter OAuth response carries no "key"')

    return OAuthCredential(access=body["key"], refresh="", expires=_MAX_SAFE_INTEGER)


class _OpenRouterCallback:
    """One-shot loopback server; the first callback carrying a code claims the exchange."""

    def __init__(self, callback_path: str) -> None:
        self._callback_path = callback_path
        self.waiter: CallbackWaiter[OAuthCredential] = CallbackWaiter()
        self.claimed = False
        self.server: Optional[OAuthCallbackServer] = None

    async def start(self, verifier: str, signal: Optional[AbortSignal]) -> "_OpenRouterCallback":
        callback_path = self._callback_path

        async def handler(path: str, query: Dict[str, str]) -> CallbackHttpResponse:
            if path != callback_path:
                return CallbackHttpResponse(404, oauth_error_html("OAuth callback route not found."))
            if self.claimed or self.waiter.settled:
                return CallbackHttpResponse(409, oauth_error_html("This OAuth callback has already been used."))

            oauth_error = query.get("error")
            if oauth_error:
                description = query.get("error_description") or oauth_error
                self.waiter.fail(ValueError(f"OpenRouter authorization failed: {description}"))
                return CallbackHttpResponse(
                    400, oauth_error_html("OpenRouter authorization was denied.", description)
                )

            code = query.get("code")
            if not code:
                return CallbackHttpResponse(400, oauth_error_html("OpenRouter returned no authorization code."))
            self.claimed = True

            try:
                credential = await exchange_authorization_code(code, verifier, signal)
                self.waiter.settle(credential)
                return CallbackHttpResponse(
                    200, oauth_success_html("Signed in to OpenRouter. You may now close this page.")
                )
            except Exception as error:
                message = str(error) or "Unknown token exchange error"
                self.waiter.fail(error)
                return CallbackHttpResponse(502, oauth_error_html("OpenRouter key exchange failed.", message))

        self.server = await start_oauth_callback_server(_callback_host(), 0, handler)
        return self

    @property
    def callback_url(self) -> str:
        assert self.server is not None
        return f"http://{_callback_host()}:{self.server.port}{self._callback_path}"

    def cancel_wait(self) -> None:
        # A claimed callback is already exchanging its code; let that exchange settle the login.
        if not self.claimed:
            self.waiter.settle(None)

    def close(self) -> None:
        if self.server is not None:
            self.server.close()


async def _login_openrouter(interaction: AuthInteraction) -> OAuthCredential:
    signal = interaction.signal
    verifier, challenge = generate_pkce()
    callback = await _OpenRouterCallback(f"/oauth/callback/{uuid.uuid4()}").start(verifier, signal)
    manual_controller = AbortController()
    manual: Dict[str, Any] = {}

    async def _manual_prompt() -> None:
        try:
            manual["input"] = await interaction.prompt(
                AuthPromptManualCode(
                    message="Complete sign-in in your browser, or paste the authorization code / redirect URL here:",
                    placeholder=callback.callback_url,
                    signal=manual_controller.signal,
                )
            )
        except Exception as error:
            manual["error"] = error
        callback.cancel_wait()

    manual_task = asyncio.create_task(_manual_prompt())
    try:
        params = urlencode(
            {
                "callback_url": callback.callback_url,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        authorize_url = f"{AUTHORIZE_URL}?{params}"
        interaction.notify(
            AuthEventProgress(message=f"Listening for OpenRouter OAuth callback on {callback.callback_url}")
        )
        interaction.notify(
            AuthEventAuthUrl(
                url=authorize_url,
                instructions=(
                    "Complete sign-in in your browser. If the browser is on another machine, "
                    "paste the final redirect URL here."
                ),
            )
        )

        credential = await race_with_abort(
            _wait_with_timeout(callback.waiter, LOGIN_TIMEOUT_SECONDS, "OpenRouter OAuth login timed out"),
            signal,
        )
        if manual.get("error") is not None:
            raise manual["error"]
        if credential is not None:
            return credential

        code = parse_authorization_input(manual["input"]) if manual.get("input") else None
        if not code:
            raise ValueError("Missing authorization code")
        interaction.notify(AuthEventProgress(message="Exchanging authorization code for an API key..."))
        return await exchange_authorization_code(code, verifier, signal)
    finally:
        manual_controller.abort()
        manual_task.cancel()
        try:
            await manual_task
        except BaseException:
            pass
        callback.close()


async def _wait_with_timeout(waiter: CallbackWaiter, timeout_seconds: float, message: str) -> Any:
    try:
        return await asyncio.wait_for(asyncio.shield(waiter.wait()), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise ValueError(message) from None


async def _refresh(credential: OAuthCredential, _signal: AbortSignal) -> OAuthCredential:
    return credential


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


openrouter_oauth = OAuthAuth(
    name="OpenRouter OAuth",
    login_label="Sign in with OpenRouter",
    login=_login_openrouter,
    refresh=_refresh,
    to_auth=_to_auth,
)

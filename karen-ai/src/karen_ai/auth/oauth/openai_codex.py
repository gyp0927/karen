"""OpenAI Codex (ChatGPT OAuth) flow, mirroring auth/oauth/openai-codex.ts.

Browser login: PKCE + localhost callback server on port 1455, raced against a
manual code prompt. Device-code login: OpenAI's bespoke deviceauth endpoints
headless-friendly. The resulting access token is a JWT carrying the ChatGPT
account id under the `https://api.openai.com/auth` claim.
"""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
import time
from typing import Any, Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit

from ...abort import AbortController, AbortSignal
from ...errors import AbortError
from ...utils.provider_env import get_provider_env_value
from ..types import (
    AuthEventAuthUrl,
    AuthEventDeviceCode,
    AuthInteraction,
    AuthPromptManualCode,
    AuthPromptSelect,
    AuthSelectOption,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
)
from ._http import post_form, post_json, race_with_abort
from .callback_server import CallbackHttpResponse, CallbackWaiter, OAuthCallbackServer, start_oauth_callback_server
from .device_code import poll_complete, poll_failed, poll_oauth_device_code_flow, poll_pending, poll_slow_down
from .oauth_page import oauth_error_html, oauth_success_html
from .pkce import generate_pkce

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_BASE_URL = "https://auth.openai.com"
AUTHORIZE_URL = f"{AUTH_BASE_URL}/oauth/authorize"
TOKEN_URL = f"{AUTH_BASE_URL}/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
DEVICE_USER_CODE_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/token"
DEVICE_VERIFICATION_URI = f"{AUTH_BASE_URL}/codex/device"
DEVICE_REDIRECT_URI = f"{AUTH_BASE_URL}/deviceauth/callback"
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
BROWSER_LOGIN_METHOD = "browser"
DEVICE_CODE_LOGIN_METHOD = "device_code"
SCOPE = "openid profile email offline_access"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"


def _callback_host() -> str:
    return get_provider_env_value("PI_OAUTH_CALLBACK_HOST") or "127.0.0.1"


def parse_authorization_input(input: str) -> Dict[str, Optional[str]]:
    """Accept a full redirect URL, `code#state`, `code=...&state=...`, or a raw code."""
    value = input.strip()
    if not value:
        return {}

    try:
        url = urlsplit(value)
        if url.scheme and url.netloc:
            params = dict(parse_qsl(url.query))
            return {"code": params.get("code"), "state": params.get("state")}
    except ValueError:
        pass

    if "#" in value:
        code, _, state = value.partition("#")
        return {"code": code, "state": state}

    if "code=" in value:
        params = dict(parse_qsl(value))
        return {"code": params.get("code"), "state": params.get("state")}

    return {"code": value}


def _decode_jwt(token: str) -> Optional[Dict[str, Any]]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        parsed = json.loads(decoded)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _get_account_id(access_token: str) -> Optional[str]:
    payload = _decode_jwt(access_token)
    auth = payload.get(JWT_CLAIM_PATH) if payload else None
    account_id = auth.get("chatgpt_account_id") if isinstance(auth, dict) else None
    return account_id if isinstance(account_id, str) and account_id else None


def _read_token_response(data: Any, operation: str, status: int, text: str, reason: str) -> "tuple[str, str, int]":
    if status >= 400:
        raise ValueError(f"OpenAI Codex token {operation} failed ({status}): {text or reason}")
    if (
        not isinstance(data, dict)
        or not data.get("access_token")
        or not data.get("refresh_token")
        or not isinstance(data.get("expires_in"), (int, float))
    ):
        raise ValueError(f"OpenAI Codex token {operation} response missing fields: {json.dumps(data)}")
    return data["access_token"], data["refresh_token"], int(time.time() * 1000) + int(data["expires_in"]) * 1000


async def _exchange_authorization_code(
    code: str,
    verifier: str,
    redirect_uri: str,
    signal: Optional[AbortSignal],
) -> "tuple[str, str, int]":
    response = await post_form(
        TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        },
        signal,
    )
    return _read_token_response(_safe_json(response), "exchange", response.status_code, response.text, response.reason_phrase)


def _safe_json(response: Any) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _credentials_from_token(access: str, refresh: str, expires: int) -> OAuthCredential:
    account_id = _get_account_id(access)
    if not account_id:
        raise ValueError("Failed to extract accountId from token")
    return OAuthCredential(access=access, refresh=refresh, expires=expires, accountId=account_id)


async def _exchange_for_credentials(
    code: str,
    verifier: str,
    redirect_uri: str,
    signal: Optional[AbortSignal],
) -> OAuthCredential:
    return _credentials_from_token(*await _exchange_authorization_code(code, verifier, redirect_uri, signal))


# -- Browser login ------------------------------------------------------------


class _LocalOAuthServer:
    def __init__(self, server: Optional[OAuthCallbackServer], waiter: CallbackWaiter) -> None:
        self._server = server
        self.waiter = waiter

    def close(self) -> None:
        if self._server is not None:
            self._server.close()

    def cancel_wait(self) -> None:
        self.waiter.settle(None)


async def _start_local_oauth_server(state: str) -> _LocalOAuthServer:
    waiter: CallbackWaiter[Dict[str, str]] = CallbackWaiter()

    def handler(path: str, query: Dict[str, str]) -> CallbackHttpResponse:
        if path != CALLBACK_PATH:
            return CallbackHttpResponse(404, oauth_error_html("Callback route not found."))
        if query.get("state") != state:
            return CallbackHttpResponse(400, oauth_error_html("State mismatch."))
        code = query.get("code")
        if not code:
            return CallbackHttpResponse(400, oauth_error_html("Missing authorization code."))
        waiter.settle({"code": code})
        return CallbackHttpResponse(
            200, oauth_success_html("OpenAI authentication completed. You can close this window.")
        )

    try:
        server = await start_oauth_callback_server(_callback_host(), CALLBACK_PORT, handler)
    except OSError:
        # Port busy: keep going with manual entry only (pi-ai resolves a dummy server).
        server = None
    return _LocalOAuthServer(server, waiter)


def _create_authorization_flow(originator: str = "pi") -> "tuple[str, str, str]":
    verifier, challenge = generate_pkce()
    state = secrets.token_hex(16)
    params = urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": originator,
        }
    )
    return verifier, state, f"{AUTHORIZE_URL}?{params}"


async def _login_browser(interaction: AuthInteraction) -> OAuthCredential:
    signal = interaction.signal
    verifier, state, url = _create_authorization_flow()
    server = await _start_local_oauth_server(state)
    manual_controller = AbortController()
    manual: Dict[str, Any] = {}

    async def _manual_prompt() -> None:
        try:
            manual["input"] = await interaction.prompt(
                AuthPromptManualCode(
                    message="Complete login in your browser, or paste the authorization code / redirect URL here:",
                    placeholder=REDIRECT_URI,
                    signal=manual_controller.signal,
                )
            )
        except Exception as error:
            manual["error"] = error
        server.cancel_wait()

    manual_task = asyncio.create_task(_manual_prompt())
    try:
        interaction.notify(
            AuthEventAuthUrl(url=url, instructions="A browser window should open. Complete login to finish.")
        )

        result = await race_with_abort(server.waiter.wait(), signal)
        if manual.get("error") is not None:
            raise manual["error"]

        code: Optional[str] = None
        if result and result.get("code"):
            code = result["code"]
        elif manual.get("input"):
            parsed = parse_authorization_input(manual["input"])
            if parsed.get("state") and parsed["state"] != state:
                raise ValueError("State mismatch")
            code = parsed.get("code")

        if not code:
            raise ValueError("Missing authorization code")
        return await _exchange_for_credentials(code, verifier, REDIRECT_URI, signal)
    finally:
        manual_controller.abort()
        manual_task.cancel()
        try:
            await manual_task
        except BaseException:
            pass
        server.close()


# -- Device-code login ----------------------------------------------------------


async def _start_device_auth(signal: Optional[AbortSignal]) -> Dict[str, Any]:
    response = await post_json(DEVICE_USER_CODE_URL, {"client_id": CLIENT_ID}, signal)
    if response.status_code >= 400:
        if response.status_code == 404:
            raise ValueError(
                "OpenAI Codex device code login is not enabled for this server. "
                "Use browser login or verify the server URL."
            )
        body = response.text
        raise ValueError(
            f"OpenAI Codex device code request failed with status {response.status_code}{f': {body}' if body else ''}"
        )

    data = _safe_json(response)
    interval = data.get("interval") if isinstance(data, dict) else None
    if isinstance(interval, str):
        try:
            interval = float(interval.strip())
        except ValueError:
            interval = None
    if (
        not isinstance(data, dict)
        or not data.get("device_auth_id")
        or not data.get("user_code")
        or not isinstance(interval, (int, float))
        or interval < 0
    ):
        raise ValueError(f"Invalid OpenAI Codex device code response: {json.dumps(data)}")
    return {"device_auth_id": data["device_auth_id"], "user_code": data["user_code"], "interval_seconds": interval}


async def _poll_device_auth(device: Dict[str, Any], signal: AbortSignal) -> Dict[str, str]:
    async def poll():
        response = await post_json(
            DEVICE_TOKEN_URL,
            {"device_auth_id": device["device_auth_id"], "user_code": device["user_code"]},
            signal,
        )
        if response.status_code < 400:
            data = _safe_json(response)
            if not isinstance(data, dict) or not data.get("authorization_code") or not data.get("code_verifier"):
                return poll_failed(f"Invalid OpenAI Codex device auth token response: {json.dumps(data)}")
            return poll_complete(
                {"authorization_code": data["authorization_code"], "code_verifier": data["code_verifier"]}
            )

        if response.status_code in (403, 404):
            return poll_pending()

        body = response.text
        error_code: Any = None
        try:
            parsed = json.loads(body)
            error = parsed.get("error") if isinstance(parsed, dict) else None
            error_code = error.get("code") if isinstance(error, dict) else error
        except ValueError:
            pass

        if error_code == "deviceauth_authorization_pending":
            return poll_pending()
        if error_code == "slow_down":
            return poll_slow_down()
        return poll_failed(
            f"OpenAI Codex device auth failed with status {response.status_code}{f': {body}' if body else ''}"
        )

    return await poll_oauth_device_code_flow(
        poll=poll,
        signal=signal,
        interval_seconds=device["interval_seconds"],
        expires_in_seconds=DEVICE_CODE_TIMEOUT_SECONDS,
    )


async def _login_device_code(interaction: AuthInteraction) -> OAuthCredential:
    device = await _start_device_auth(interaction.signal)
    interaction.notify(
        AuthEventDeviceCode(
            user_code=device["user_code"],
            verification_uri=DEVICE_VERIFICATION_URI,
            interval_seconds=int(device["interval_seconds"]),
            expires_in_seconds=DEVICE_CODE_TIMEOUT_SECONDS,
        )
    )
    code = await _poll_device_auth(device, interaction.signal or AbortSignal())
    return await _exchange_for_credentials(
        code["authorization_code"], code["code_verifier"], DEVICE_REDIRECT_URI, interaction.signal
    )


# -- Login / refresh --------------------------------------------------------------


async def _login_openai_codex(interaction: AuthInteraction) -> OAuthCredential:
    method = await interaction.prompt(
        AuthPromptSelect(
            message="Select OpenAI Codex login method:",
            options=[
                AuthSelectOption(id=BROWSER_LOGIN_METHOD, label="Browser login (default)"),
                AuthSelectOption(id=DEVICE_CODE_LOGIN_METHOD, label="Device code login (headless)"),
            ],
        )
    )
    if method == DEVICE_CODE_LOGIN_METHOD:
        return await _login_device_code(interaction)
    if method != BROWSER_LOGIN_METHOD:
        raise ValueError(f"Unknown OpenAI Codex login method: {method}")
    return await _login_browser(interaction)


async def refresh_openai_codex_token(refresh_token: str, signal: Optional[AbortSignal]) -> OAuthCredential:
    try:
        response = await post_form(
            TOKEN_URL,
            {"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": CLIENT_ID},
            signal,
        )
    except Exception as error:
        raise ValueError(f"OpenAI Codex token refresh error: {error}") from error
    return _credentials_from_token(
        *_read_token_response(_safe_json(response), "refresh", response.status_code, response.text, response.reason_phrase)
    )


async def _refresh(credential: OAuthCredential, signal: AbortSignal) -> OAuthCredential:
    return await refresh_openai_codex_token(credential.refresh, signal)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


openai_codex_oauth = OAuthAuth(
    name="OpenAI (ChatGPT Plus/Pro)",
    is_subscription=True,
    login=_login_openai_codex,
    refresh=_refresh,
    to_auth=_to_auth,
)

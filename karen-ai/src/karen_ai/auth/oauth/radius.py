"""Radius gateway OAuth flow, mirroring auth/oauth/radius.ts.

Radius is a pi-messages gateway. OAuth client APIs live on the configured
gateway; only the interactive browser authorization endpoint is discovered.
Login offers browser sign-in (PKCE + loopback callback on port 1456) or an
RFC 8628 device-code flow for signing in from another device.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Dict, Optional
from urllib.parse import urlencode

from ...abort import AbortSignal
from ...errors import AbortError
from ..types import (
    AuthEventAuthUrl,
    AuthEventDeviceCode,
    AuthEventProgress,
    AuthInteraction,
    AuthPromptSelect,
    AuthSelectOption,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
)
from ._http import get, post_form, race_with_abort
from .callback_server import CallbackHttpResponse, CallbackWaiter, OAuthCallbackServer, start_oauth_callback_server
from .device_code import poll_complete, poll_failed, poll_oauth_device_code_flow, poll_pending, poll_slow_down
from .oauth_page import oauth_error_html, oauth_success_html
from .pkce import generate_pkce

CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 1456
CALLBACK_PATH = "/oauth/callback"
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
TOKEN_EXPIRY_SKEW_MS = 60_000
LOGIN_METHOD_BROWSER = "browser"
LOGIN_METHOD_DEVICE_CODE = "device-code"
OAUTH_CLIENT_ID = "pi-gateway"
OAUTH_SCOPE = "gateway offline_access"
OAUTH_DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"

DEFAULT_RADIUS_GATEWAY = "https://radius.pi.dev"


def normalize_radius_gateway_url(value: str) -> str:
    """Mirroring providers/radius-config.ts normalizeRadiusGatewayUrl."""
    with_scheme = value if re.match(r"^https?://", value, re.IGNORECASE) else f"https://{value}"
    return with_scheme.rstrip("/")


class OAuthResponseError(ValueError):
    def __init__(self, status: int, oauth_error: Optional[str], description: Optional[str], message: str) -> None:
        if oauth_error:
            detail = f"{oauth_error}: {description}" if description else oauth_error
        else:
            detail = description or str(status)
        super().__init__(f"{message}: {detail}")
        self.status = status
        self.oauth_error = oauth_error


async def _read_oauth_response_error(response: Any, message: str) -> OAuthResponseError:
    text = response.text
    oauth_error: Optional[str] = None
    description: Optional[str] = None
    if text:
        try:
            data = response.json()
            if isinstance(data.get("error"), str):
                oauth_error = data["error"]
            if isinstance(data.get("error_description"), str):
                description = data["error_description"]
        except ValueError:
            description = text
    return OAuthResponseError(response.status_code, oauth_error, description, message)


async def _load_radius_oauth_discovery(gateway: str, signal: Optional[AbortSignal]) -> str:
    response = await get(f"{gateway}/v1/oauth", signal, headers={"accept": "application/json"})
    if response.status_code >= 400:
        raise ValueError(f"Could not load Radius OAuth config from {gateway}: {response.status_code} {response.text}")
    try:
        discovery = response.json()
    except ValueError:
        discovery = {}
    endpoint = discovery.get("authorizationEndpoint")
    if not isinstance(endpoint, str):
        raise ValueError(f"Invalid Radius OAuth config from {gateway}")
    return endpoint


async def _request_oauth_token(gateway: str, fields: Dict[str, str], signal: Optional[AbortSignal]) -> OAuthCredential:
    response = await post_form(f"{gateway}/v1/oauth/token", fields, signal)
    if response.status_code >= 400:
        raise await _read_oauth_response_error(response, "Radius OAuth token request failed")

    data = response.json()
    extra = {"scope": data["scope"]} if isinstance(data.get("scope"), str) else {}
    return OAuthCredential(
        access=data["access_token"],
        refresh=data["refresh_token"],
        expires=int(time.time() * 1000) + int(data["expires_in"]) * 1000 - TOKEN_EXPIRY_SKEW_MS,
        **extra,
    )


# -- Browser login ------------------------------------------------------------


class _CallbackServerResult:
    def __init__(self, server: Optional[OAuthCallbackServer], waiter: CallbackWaiter) -> None:
        self._server = server
        self.waiter = waiter

    def close(self) -> None:
        self.waiter.settle(None)
        if self._server is not None:
            self._server.close()


async def _start_oauth_callback_server(expected_state: str) -> _CallbackServerResult:
    waiter: CallbackWaiter[str] = CallbackWaiter()

    def handler(path: str, query: Dict[str, str]) -> CallbackHttpResponse:
        if path != CALLBACK_PATH:
            return CallbackHttpResponse(404, oauth_error_html("Callback route not found."))
        if query.get("state") != expected_state:
            return CallbackHttpResponse(400, oauth_error_html("OAuth state mismatch."))
        error = query.get("error")
        if error:
            description = query.get("error_description") or error
            waiter.settle(None)
            return CallbackHttpResponse(400, oauth_error_html(description))
        code = query.get("code")
        if not code:
            return CallbackHttpResponse(400, oauth_error_html("Missing authorization code."))
        waiter.settle(code)
        return CallbackHttpResponse(200, oauth_success_html("Signed in to Radius. You may now close this page."))

    try:
        server = await start_oauth_callback_server(CALLBACK_HOST, CALLBACK_PORT, handler)
    except OSError:
        # Port busy: wait resolves to None and the login reports the callback did not complete.
        server = None
    return _CallbackServerResult(server, waiter)


async def _login_with_browser(
    gateway: str, authorization_endpoint: str, interaction: AuthInteraction
) -> OAuthCredential:
    signal = interaction.signal
    verifier, challenge = generate_pkce()
    state = str(uuid.uuid4())
    params = urlencode(
        {
            "response_type": "code",
            "client_id": OAUTH_CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": OAUTH_SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "handoff": "url",
            "state": state,
        }
    )
    authorize_url = f"{authorization_endpoint}?{params}"

    callback_server = await _start_oauth_callback_server(state)
    interaction.notify(AuthEventProgress(message=f"Listening for OAuth callback on {REDIRECT_URI}"))
    interaction.notify(AuthEventAuthUrl(url=authorize_url, instructions="Continue in your browser."))

    try:
        code = await race_with_abort(callback_server.waiter.wait(), signal)
        if not code:
            if signal is not None and signal.aborted:
                raise AbortError("Login cancelled")
            raise ValueError("OAuth callback did not complete.")
        return await _request_oauth_token(
            gateway,
            {
                "grant_type": "authorization_code",
                "client_id": OAUTH_CLIENT_ID,
                "redirect_uri": REDIRECT_URI,
                "code": code,
                "code_verifier": verifier,
            },
            signal,
        )
    finally:
        callback_server.close()


# -- Device-code login ----------------------------------------------------------


async def _request_device_authorization(gateway: str, signal: Optional[AbortSignal]) -> Dict[str, Any]:
    response = await post_form(
        f"{gateway}/v1/oauth/device",
        {"client_id": OAUTH_CLIENT_ID, "scope": OAUTH_SCOPE},
        signal,
    )
    if response.status_code >= 400:
        raise await _read_oauth_response_error(response, "Radius OAuth device authorization failed")

    data = response.json()
    if not all(data.get(key) for key in ("device_code", "user_code", "verification_uri", "expires_in")):
        raise ValueError("Radius OAuth device authorization response is missing required fields")
    return {
        "device_code": data["device_code"],
        "user_code": data["user_code"],
        "verification_uri": data["verification_uri"],
        "expires_in": data["expires_in"],
        "interval": data.get("interval"),
    }


async def _login_with_device_code(gateway: str, interaction: AuthInteraction) -> OAuthCredential:
    signal = interaction.signal
    device = await _request_device_authorization(gateway, signal)
    interaction.notify(
        AuthEventDeviceCode(
            user_code=device["user_code"],
            verification_uri=device["verification_uri"],
            interval_seconds=int(device["interval"]) if device.get("interval") else None,
            expires_in_seconds=int(device["expires_in"]),
        )
    )

    async def poll():
        try:
            credentials = await _request_oauth_token(
                gateway,
                {
                    "grant_type": OAUTH_DEVICE_CODE_GRANT_TYPE,
                    "client_id": OAUTH_CLIENT_ID,
                    "device_code": device["device_code"],
                },
                signal,
            )
            return poll_complete(credentials)
        except OAuthResponseError as error:
            if error.oauth_error == "authorization_pending":
                return poll_pending()
            if error.oauth_error == "slow_down":
                return poll_slow_down()
            if error.oauth_error == "expired_token":
                return poll_failed("Device authorization expired.")
            if error.oauth_error == "access_denied":
                return poll_failed("Device authorization was denied.")
            raise

    return await poll_oauth_device_code_flow(
        poll=poll,
        signal=signal or AbortSignal(),
        interval_seconds=device["interval"],
        expires_in_seconds=device["expires_in"],
    )


def create_radius_oauth(name: str, gateway: str) -> OAuthAuth:
    gateway_url = normalize_radius_gateway_url(gateway)

    async def login(interaction: AuthInteraction) -> OAuthCredential:
        login_method = await interaction.prompt(
            AuthPromptSelect(
                message=f"Sign in to {name}:",
                options=[
                    AuthSelectOption(id=LOGIN_METHOD_BROWSER, label="Sign in with browser (recommended)"),
                    AuthSelectOption(
                        id=LOGIN_METHOD_DEVICE_CODE,
                        label="Sign in with device code (when signing in from another device)",
                    ),
                ],
            )
        )
        if login_method == LOGIN_METHOD_DEVICE_CODE:
            return await _login_with_device_code(gateway_url, interaction)
        if login_method == LOGIN_METHOD_BROWSER:
            authorization_endpoint = await _load_radius_oauth_discovery(gateway_url, interaction.signal)
            return await _login_with_browser(gateway_url, authorization_endpoint, interaction)
        raise ValueError(f"Unknown {name} sign-in method: {login_method}")

    async def refresh(credential: OAuthCredential, signal: AbortSignal) -> OAuthCredential:
        return await _request_oauth_token(
            gateway_url,
            {
                "grant_type": "refresh_token",
                "client_id": OAUTH_CLIENT_ID,
                "refresh_token": credential.refresh,
            },
            signal,
        )

    async def to_auth(credential: OAuthCredential) -> ModelAuth:
        return ModelAuth(api_key=credential.access)

    return OAuthAuth(name=name, login=login, refresh=refresh, to_auth=to_auth)

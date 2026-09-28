"""Meta Model API OAuth flow, mirroring auth/oauth/meta.ts.

RFC 8628 device authorization grant against https://auth.meta.com (JSON
responses). Meta splits identity from API access: the resulting identity
token is not accepted for inference, so it is exchanged for a Model API key
via the Muse Code key-mint endpoint (minted keys live about a day). The
identity token is stored as `refresh` and the minted key as `access`, so the
standard OAuth scheduler re-mints the key when it expires with no bespoke
renewal machinery. The identity token itself is not renewable, so a 401/403
from mint means the session is dead and the user must sign in again.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from ...abort import AbortSignal
from ...errors import AbortError
from ..types import AuthEventDeviceCode, AuthEventProgress, AuthInteraction, ModelAuth, OAuthAuth, OAuthCredential
from ._http import post_form, post_json
from .device_code import poll_complete, poll_failed, poll_oauth_device_code_flow, poll_pending, poll_slow_down

# Muse Code CLI client id.
CLIENT_ID = "1031625952748946"
AUTH_HOST = "https://auth.meta.com"
DEVICE_AUTHORIZATION_URL = f"{AUTH_HOST}/oidc/device/authorization/"
DEVICE_TOKEN_URL = f"{AUTH_HOST}/oidc/device/token/"
API_KEY_MINT_URL = "https://api.meta.ai/muse-code/key"
API_KEY_LIFETIME_MS = 24 * 60 * 60 * 1000
REQUEST_TIMEOUT_SECONDS = 30.0


def _read_json(response: Any) -> Optional[Dict[str, Any]]:
    try:
        data = response.json()
        return data if isinstance(data, dict) else None
    except ValueError:
        return None


def _error_detail(data: Optional[Dict[str, Any]]) -> str:
    for key in ("error_description", "detail", "message", "error"):
        value = data.get(key) if data else None
        if isinstance(value, str) and value.strip():
            return f": {value.strip()}"
    return ""


def _trusted_http_url(value: Any) -> Optional[str]:
    """The verification URI is opened in the user's browser; only http(s) URLs are trusted."""
    if not isinstance(value, str) or not value:
        return None
    try:
        url = urlsplit(value)
    except ValueError:
        return None
    if url.scheme not in ("https", "http") or not url.netloc:
        return None
    return url.geturl()


def _positive_number(value: Any) -> Optional[float]:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


async def _start_device_authorization(signal: Optional[AbortSignal]) -> Dict[str, Any]:
    response = await post_form(
        DEVICE_AUTHORIZATION_URL, {"client_id": CLIENT_ID}, signal, timeout=REQUEST_TIMEOUT_SECONDS
    )
    data = _read_json(response)
    if response.status_code >= 400:
        raise ValueError(f"Meta device authorization failed with status {response.status_code}{_error_detail(data)}")
    device_code = data.get("device_code") if data else None
    user_code = data.get("user_code") if data else None
    verification_uri = _trusted_http_url(data.get("verification_uri_complete") if data else None) or _trusted_http_url(
        data.get("verification_uri") if data else None
    )
    if not isinstance(device_code, str) or not device_code or not isinstance(user_code, str) or not user_code or not verification_uri:
        raise ValueError(f"Invalid Meta device authorization response: {json.dumps(data)}")
    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": verification_uri,
        "interval_seconds": _positive_number(data.get("interval") if data else None),
        "expires_in_seconds": _positive_number(data.get("expires_in") if data else None),
    }


async def _poll_for_identity_token(device: Dict[str, Any], signal: AbortSignal) -> str:
    async def poll():
        response = await post_form(
            DEVICE_TOKEN_URL,
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device["device_code"],
                "client_id": CLIENT_ID,
            },
            signal,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        data = _read_json(response)
        access_token = data.get("access_token") if data else None
        if response.status_code < 400 and isinstance(access_token, str) and access_token:
            return poll_complete(access_token)
        error = data.get("error") if data else None
        if error == "authorization_pending":
            return poll_pending()
        if error == "slow_down":
            return poll_slow_down(_positive_number(data.get("interval")) if data else None)
        if error == "access_denied":
            return poll_failed("Meta login was denied.")
        if error == "expired_token":
            return poll_failed("Meta device authorization expired. Please restart login.")
        return poll_failed(
            f"Meta device token request failed with status {response.status_code}{_error_detail(data)}"
        )

    return await poll_oauth_device_code_flow(
        poll=poll,
        signal=signal,
        interval_seconds=device["interval_seconds"],
        expires_in_seconds=device["expires_in_seconds"],
        wait_before_first_poll=True,
    )


async def mint_api_key(identity_token: str, signal: Optional[AbortSignal]) -> OAuthCredential:
    """Exchange an identity token for a Model API key. Keys are valid for about a day."""
    response = await post_json(
        API_KEY_MINT_URL,
        {},
        signal,
        headers={
            "Authorization": f"Bearer {identity_token}",
            "x-api-version": "1.0.0",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    data = _read_json(response)
    if response.status_code in (401, 403):
        # Identity token is not renewable (see module docstring); only a fresh device flow helps.
        raise ValueError(
            f"Meta session expired (status {response.status_code}). "
            f"Run `/login meta` to sign in again.{_error_detail(data)}"
        )
    if response.status_code >= 400:
        raise ValueError(f"Meta API key mint failed with status {response.status_code}{_error_detail(data)}")
    api_key = data.get("api_key") if data else None
    if not isinstance(api_key, str) or not api_key:
        action_url = _trusted_http_url(data.get("action_url") if data else None)
        raise ValueError(f"Meta did not issue an API key.{f' Complete setup at {action_url}' if action_url else ''}")
    return OAuthCredential(
        refresh=identity_token,
        access=api_key,
        expires=int(time.time() * 1000) + API_KEY_LIFETIME_MS,
    )


async def _login_meta(interaction: AuthInteraction) -> OAuthCredential:
    signal = interaction.signal
    try:
        device = await _start_device_authorization(signal)
        interaction.notify(
            AuthEventDeviceCode(
                user_code=device["user_code"],
                verification_uri=device["verification_uri"],
                interval_seconds=int(device["interval_seconds"]) if device["interval_seconds"] is not None else None,
                expires_in_seconds=int(device["expires_in_seconds"]) if device["expires_in_seconds"] is not None else None,
            )
        )
        identity_token = await _poll_for_identity_token(device, signal or AbortSignal())
        interaction.notify(AuthEventProgress(message="Enabling Meta Model API access..."))
        return await mint_api_key(identity_token, signal)
    except Exception as error:
        # An in-flight request raises on abort; the login UI matches on this message.
        if signal is not None and signal.aborted:
            raise AbortError("Login cancelled") from error
        raise


async def _refresh(credential: OAuthCredential, signal: AbortSignal) -> OAuthCredential:
    return await mint_api_key(credential.refresh, signal)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


meta_oauth = OAuthAuth(
    name="Meta (Muse subscription)",
    is_subscription=True,
    login_label="Sign in with Meta",
    login=_login_meta,
    refresh=_refresh,
    to_auth=_to_auth,
)

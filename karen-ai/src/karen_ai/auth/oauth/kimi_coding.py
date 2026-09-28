"""Kimi Code (subscription) OAuth flow, mirroring auth/oauth/kimi-coding.ts.

RFC 8628 device authorization grant against https://auth.kimi.com with JSON
responses. The access token authenticates requests to
https://api.kimi.com/coding as an `Authorization: Bearer` header.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from ...abort import AbortSignal, abortable_sleep
from ...errors import AbortError
from ...utils.provider_env import get_provider_env_value
from ..types import AuthEventDeviceCode, AuthInteraction, ModelAuth, OAuthAuth, OAuthCredential
from ._http import post_form
from .device_code import poll_complete, poll_failed, poll_oauth_device_code_flow, poll_pending, poll_slow_down

CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
DEFAULT_OAUTH_HOST = "https://auth.kimi.com"
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
DEFAULT_POLL_INTERVAL_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 30.0
REFRESH_MAX_RETRIES = 3


def _get_oauth_host() -> str:
    override = get_provider_env_value("KIMI_CODE_OAUTH_HOST") or get_provider_env_value("KIMI_OAUTH_HOST")
    return (override or DEFAULT_OAUTH_HOST).rstrip("/")


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


async def _start_device_authorization(oauth_host: str, signal: Optional[AbortSignal]) -> Dict[str, Any]:
    response = await post_form(
        f"{oauth_host}/api/oauth/device_authorization",
        {"client_id": CLIENT_ID},
        signal,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code >= 400:
        text = response.text
        raise ValueError(
            f"Kimi Code device authorization failed with status {response.status_code}{f': {text}' if text else ''}"
        )

    try:
        data: Any = response.json()
    except ValueError:
        data = None
    device_code = data.get("device_code") if isinstance(data, dict) else None
    user_code = data.get("user_code") if isinstance(data, dict) else None
    verification_uri = data.get("verification_uri") if isinstance(data, dict) else None
    verification_uri_complete = data.get("verification_uri_complete") if isinstance(data, dict) else None
    if (
        not isinstance(device_code, str)
        or not isinstance(user_code, str)
        or not _trusted_http_url(verification_uri)
        or not _trusted_http_url(verification_uri_complete)
    ):
        raise ValueError(f"Invalid Kimi Code device authorization response: {json.dumps(data)}")

    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": verification_uri,
        "verification_uri_complete": verification_uri_complete,
        "interval_seconds": _positive_number(data.get("interval")) or DEFAULT_POLL_INTERVAL_SECONDS,
        "expires_in_seconds": _positive_number(data.get("expires_in")) or DEVICE_CODE_TIMEOUT_SECONDS,
    }


def _parse_token_response(data: Any, operation: str) -> Dict[str, Any]:
    access_token = data.get("access_token") if isinstance(data, dict) else None
    refresh_token = data.get("refresh_token") if isinstance(data, dict) else None
    expires_in = data.get("expires_in") if isinstance(data, dict) else None
    if (
        not isinstance(access_token, str)
        or not access_token
        or not isinstance(refresh_token, str)
        or not refresh_token
        or not isinstance(expires_in, (int, float))
        or expires_in <= 0
    ):
        raise ValueError(f"Kimi Code token {operation} response missing fields: {json.dumps(data)}")
    return {"access": access_token, "refresh": refresh_token, "expires": int(time.time() * 1000) + int(expires_in) * 1000}


async def _poll_for_token(oauth_host: str, device: Dict[str, Any], signal: AbortSignal) -> Dict[str, Any]:
    async def poll():
        response = await post_form(
            f"{oauth_host}/api/oauth/token",
            {
                "client_id": CLIENT_ID,
                "device_code": device["device_code"],
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            signal,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code >= 500:
            text = response.text
            return poll_failed(
                f"Kimi Code device token request failed with status {response.status_code}{f': {text}' if text else ''}"
            )

        try:
            data: Any = response.json()
        except ValueError:
            data = None
        if response.status_code < 400 and isinstance(data, dict) and isinstance(data.get("access_token"), str):
            try:
                return poll_complete(_parse_token_response(data, "poll"))
            except ValueError as error:
                return poll_failed(str(error))

        error = data.get("error") if isinstance(data, dict) else None
        description = data.get("error_description") if isinstance(data, dict) else None
        suffix = f": {description}" if isinstance(description, str) else ""
        if error == "authorization_pending":
            return poll_pending()
        if error == "slow_down":
            interval = data.get("interval") if isinstance(data, dict) else None
            return poll_slow_down(interval if isinstance(interval, (int, float)) and interval > 0 else None)
        if error == "expired_token":
            return poll_failed("Kimi Code device authorization expired. Please restart login.")
        if error == "access_denied":
            return poll_failed("Kimi Code login was denied.")
        error_suffix = f": {error}{suffix}" if isinstance(error, str) else ""
        return poll_failed(f"Kimi Code device token request failed (status {response.status_code}){error_suffix}")

    return await poll_oauth_device_code_flow(
        poll=poll,
        signal=signal,
        interval_seconds=device["interval_seconds"],
        expires_in_seconds=device["expires_in_seconds"],
        wait_before_first_poll=True,
    )


def _is_retryable_refresh_failure(status: int) -> bool:
    return status == 429 or status >= 500


async def refresh_token(oauth_host: str, refresh_token_value: str, signal: Optional[AbortSignal]) -> Dict[str, Any]:
    last_error: Optional[BaseException] = None
    for attempt in range(REFRESH_MAX_RETRIES + 1):
        if attempt > 0:
            await abortable_sleep(1.0 * 2 ** (attempt - 1), signal)
        if signal is not None and signal.aborted:
            raise AbortError("Kimi Code token refresh aborted")

        try:
            response = await post_form(
                f"{oauth_host}/api/oauth/token",
                {
                    "client_id": CLIENT_ID,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token_value,
                },
                signal,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as error:
            last_error = error
            continue

        try:
            data: Any = response.json()
        except ValueError:
            data = None
        if response.status_code < 400:
            return _parse_token_response(data, "refresh")

        # Unauthorized: the stored credential is dead; Models clears it and prompts re-login.
        error = data.get("error") if isinstance(data, dict) else None
        if response.status_code in (401, 403) or error == "invalid_grant":
            description = data.get("error_description") if isinstance(data, dict) else None
            suffix = f": {description}" if isinstance(description, str) else ""
            raise ValueError(f"Kimi Code token refresh unauthorized (status {response.status_code}){suffix}")

        if _is_retryable_refresh_failure(response.status_code) and attempt < REFRESH_MAX_RETRIES:
            last_error = ValueError(f"Kimi Code token refresh failed with status {response.status_code}")
            continue

        text = json.dumps(data)
        raise ValueError(
            f"Kimi Code token refresh failed with status {response.status_code}{f': {text}' if text else ''}"
        )

    if last_error is not None:
        raise last_error
    raise ValueError("Kimi Code token refresh failed")


async def _login_kimi_coding(interaction: AuthInteraction) -> OAuthCredential:
    oauth_host = _get_oauth_host()
    device = await _start_device_authorization(oauth_host, interaction.signal)
    interaction.notify(
        AuthEventDeviceCode(
            user_code=device["user_code"],
            verification_uri=device["verification_uri_complete"],
            interval_seconds=int(device["interval_seconds"]),
            expires_in_seconds=int(device["expires_in_seconds"]),
        )
    )
    token = await _poll_for_token(oauth_host, device, interaction.signal or AbortSignal())
    return OAuthCredential(access=token["access"], refresh=token["refresh"], expires=token["expires"])


async def _refresh(credential: OAuthCredential, signal: AbortSignal) -> OAuthCredential:
    token = await refresh_token(_get_oauth_host(), credential.refresh, signal)
    return OAuthCredential(access=token["access"], refresh=token["refresh"], expires=token["expires"])


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(headers={"Authorization": f"Bearer {credential.access}"})


kimi_coding_oauth = OAuthAuth(
    name="Kimi Code (subscription)",
    is_subscription=True,
    login_label="Sign in with Kimi Code",
    login=_login_kimi_coding,
    refresh=_refresh,
    to_auth=_to_auth,
)

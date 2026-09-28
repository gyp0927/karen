"""xAI OAuth device-code flow, mirroring auth/oauth/xai.ts."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from ...abort import AbortSignal
from ..types import AuthEventDeviceCode, AuthInteraction, ModelAuth, OAuthAuth, OAuthCredential
from ._http import post_form
from .device_code import poll_complete, poll_failed, poll_oauth_device_code_flow, poll_pending, poll_slow_down

XAI_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"
# Refresh slightly before the reported expiry to avoid using a token that dies mid-request.
REFRESH_SKEW_MS = 5 * 60 * 1000
DEFAULT_TOKEN_LIFETIME_SECONDS = 3600


def _required_string(body: Dict[str, Any], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid xAI OAuth response field: {field}")
    return value


def _positive_number(body: Dict[str, Any], field: str) -> float:
    value = body.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Invalid xAI OAuth response field: {field}")
    return value


def _validate_verification_uri(raw: str) -> str:
    """The verification URI is opened in the user's browser; force https so a
    malicious response cannot make `open` launch something else."""
    try:
        url = urlsplit(raw)
    except ValueError:
        raise ValueError("Untrusted verification URI in xAI OAuth response") from None
    if url.scheme != "https" or not url.netloc:
        raise ValueError("Untrusted verification URI in xAI OAuth response")
    return url.geturl()


def _request_failure(action: str, status: int, body: Any) -> ValueError:
    error = body.get("error") if isinstance(body, dict) else None
    description = body.get("error_description") if isinstance(body, dict) else None
    detail = ": ".join(part for part in (error, description) if isinstance(part, str) and part)
    return ValueError(f"xAI OAuth {action} failed (HTTP {status}){f': {detail}' if detail else ''}")


async def _post_form(url: str, fields: Dict[str, str], signal: Optional[AbortSignal]) -> "tuple[int, Any]":
    response = await post_form(url, fields, signal)
    try:
        body: Any = response.json()
        if not isinstance(body, dict):
            body = {}
    except ValueError:
        raise ValueError(f"xAI OAuth returned invalid JSON (HTTP {response.status_code})") from None
    return response.status_code, body


def _parse_device_code(body: Dict[str, Any]) -> Dict[str, Any]:
    # RFC 8628 allows interval 0 (no minimum wait); fall back to the poller's
    # default instead of failing on non-positive or malformed values.
    interval = body.get("interval")
    interval_seconds = interval if isinstance(interval, (int, float)) and not isinstance(interval, bool) and interval > 0 else None
    complete = body.get("verification_uri_complete")
    return {
        "device_code": _required_string(body, "device_code"),
        "user_code": _required_string(body, "user_code"),
        "verification_uri": _validate_verification_uri(_required_string(body, "verification_uri")),
        "verification_uri_complete": _validate_verification_uri(complete)
        if isinstance(complete, str) and complete
        else None,
        "interval_seconds": interval_seconds,
        "expires_in_seconds": _positive_number(body, "expires_in"),
    }


def _credentials_from_token_response(body: Dict[str, Any], previous_refresh_token: Optional[str] = None) -> OAuthCredential:
    access = _required_string(body, "access_token")
    # xAI may omit refresh_token on refresh when the token is not rotated.
    if body.get("refresh_token") is None and previous_refresh_token:
        refresh = previous_refresh_token
    else:
        refresh = _required_string(body, "refresh_token")
    expires_in = DEFAULT_TOKEN_LIFETIME_SECONDS if body.get("expires_in") is None else _positive_number(body, "expires_in")
    return OAuthCredential(
        access=access,
        refresh=refresh,
        expires=int(time.time() * 1000) + int(expires_in) * 1000 - REFRESH_SKEW_MS,
    )


async def _request_device_code(signal: Optional[AbortSignal]) -> Dict[str, Any]:
    status, body = await _post_form(
        XAI_DEVICE_CODE_URL,
        {"client_id": XAI_CLIENT_ID, "scope": XAI_SCOPE, "referrer": "pi"},
        signal,
    )
    if status >= 400:
        raise _request_failure("device authorization", status, body)
    return _parse_device_code(body)


async def _poll_for_tokens(device: Dict[str, Any], signal: AbortSignal) -> OAuthCredential:
    async def poll():
        status, body = await _post_form(
            XAI_TOKEN_URL,
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": XAI_CLIENT_ID,
                "device_code": device["device_code"],
            },
            signal,
        )

        if status < 400:
            return poll_complete(_credentials_from_token_response(body))

        error = body.get("error")
        if error == "authorization_pending":
            return poll_pending()
        if error == "slow_down":
            interval = body.get("interval")
            return poll_slow_down(interval if isinstance(interval, (int, float)) else None)
        if error in ("access_denied", "authorization_denied"):
            return poll_failed("xAI device authorization was denied")
        if error == "expired_token":
            return poll_failed("xAI device code expired")
        return poll_failed(str(_request_failure("device token polling", status, body)))

    return await poll_oauth_device_code_flow(
        poll=poll,
        signal=signal,
        interval_seconds=device["interval_seconds"],
        expires_in_seconds=device["expires_in_seconds"],
        wait_before_first_poll=True,
    )


async def _login_xai(interaction: AuthInteraction) -> OAuthCredential:
    device = await _request_device_code(interaction.signal)
    interaction.notify(
        AuthEventDeviceCode(
            user_code=device["user_code"],
            verification_uri=device["verification_uri_complete"] or device["verification_uri"],
            interval_seconds=int(device["interval_seconds"]) if device["interval_seconds"] is not None else None,
            expires_in_seconds=int(device["expires_in_seconds"]),
        )
    )
    return await _poll_for_tokens(device, interaction.signal or AbortSignal())


async def refresh_xai_token(refresh_token: str, signal: Optional[AbortSignal]) -> OAuthCredential:
    status, body = await _post_form(
        XAI_TOKEN_URL,
        {"grant_type": "refresh_token", "client_id": XAI_CLIENT_ID, "refresh_token": refresh_token},
        signal,
    )
    if status >= 400:
        raise _request_failure("token refresh", status, body)
    return _credentials_from_token_response(body, refresh_token)


async def _refresh(credential: OAuthCredential, signal: AbortSignal) -> OAuthCredential:
    return await refresh_xai_token(credential.refresh, signal)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


xai_oauth = OAuthAuth(
    name="xAI (Grok/X subscription)",
    is_subscription=True,
    login_label="Sign in with SuperGrok or X Premium",
    login=_login_xai,
    refresh=_refresh,
    to_auth=_to_auth,
)

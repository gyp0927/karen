"""GitHub Copilot OAuth flow, mirroring auth/oauth/github-copilot.ts.

Device-code login against GitHub (optionally a GitHub Enterprise domain),
then exchange of the GitHub access token for a short-lived Copilot token.
The Copilot token's `proxy-ep` field determines the per-account API base URL.
Model catalog fetching and policy enabling are part of login: accounts can
have models that need a one-time policy opt-in.
"""

from __future__ import annotations

import base64
import re
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlsplit

import httpx

from ...abort import AbortSignal, abortable_sleep
from ...errors import AbortError
from ..types import (
    AuthEventDeviceCode,
    AuthEventProgress,
    AuthInteraction,
    AuthPromptText,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
)
from ._http import get, post_form, race_with_abort
from .device_code import poll_complete, poll_failed, poll_oauth_device_code_flow, poll_pending, poll_slow_down

CLIENT_ID = base64.b64decode("SXYxLmI1MDdhMDhjODdlY2ZlOTg=").decode("ascii")

COPILOT_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}
COPILOT_API_VERSION = "2026-06-01"
INDIVIDUAL_BASE_URL = "https://api.individual.githubcopilot.com"

_TOKEN_EXPIRY_SKEW_MS = 5 * 60 * 1000

#: Models pi-ai's generated catalog knows; used for the "unconfigured" policy
#: fallback that auto-enables known models the account has never configured.
#: Curated list (pi-ai's catalog is build-generated and not in the repo).
GITHUB_COPILOT_KNOWN_MODEL_IDS = frozenset(
    {
        "gpt-4o",
        "gpt-4.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.1-codex",
        "gpt-5.1-codex-mini",
        "o3",
        "o4-mini",
        "claude-sonnet-4",
        "claude-sonnet-4.5",
        "claude-opus-4.1",
        "claude-opus-4.5",
        "claude-haiku-4.5",
        "gemini-2.5-pro",
        "gemini-3-pro-preview",
        "grok-code-fast-1",
    }
)


def normalize_domain(input: str) -> Optional[str]:
    trimmed = input.strip()
    if not trimmed:
        return None
    try:
        url = urlsplit(trimmed if "://" in trimmed else f"https://{trimmed}")
        return url.hostname or None
    except ValueError:
        return None


def _get_urls(domain: str) -> Dict[str, str]:
    return {
        "device_code_url": f"https://{domain}/login/device/code",
        "access_token_url": f"https://{domain}/login/oauth/access_token",
        "copilot_token_url": f"https://api.{domain}/copilot_internal/v2/token",
    }


def get_base_url_from_token(token: str) -> Optional[str]:
    """Parse proxy-ep from a Copilot token and convert to the API base URL.

    Token format: tid=...;exp=...;proxy-ep=proxy.individual.githubcopilot.com;...
    """
    match = re.search(r"proxy-ep=([^;]+)", token)
    if not match:
        return None
    api_host = re.sub(r"^proxy\.", "api.", match.group(1))
    return f"https://{api_host}"


def get_github_copilot_base_url(token: Optional[str] = None, enterprise_domain: Optional[str] = None) -> str:
    if token:
        url_from_token = get_base_url_from_token(token)
        if url_from_token:
            return url_from_token
    if enterprise_domain:
        return f"https://copilot-api.{enterprise_domain}"
    return INDIVIDUAL_BASE_URL


def parse_github_copilot_model_catalog(
    raw: Any,
    allow_policy_fallback: bool,
    known_model_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    known = set(known_model_ids) if known_model_ids is not None else GITHUB_COPILOT_KNOWN_MODEL_IDS
    data = raw.get("data") if isinstance(raw, dict) else None
    if not isinstance(data, list):
        raise ValueError("Invalid Copilot models response")

    account_models = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        capabilities = item.get("capabilities")
        supports = capabilities.get("supports") if isinstance(capabilities, dict) else None
        if isinstance(supports, dict) and supports.get("tool_calls") is False:
            continue
        policy = item.get("policy")
        account_models.append(
            {
                "id": item["id"],
                "picker_enabled": item.get("model_picker_enabled") is True,
                "policy_state": policy.get("state") if isinstance(policy, dict) else None,
            }
        )

    picker_model_ids = [m["id"] for m in account_models if m["picker_enabled"] and m["policy_state"] != "disabled"]
    use_policy_fallback = allow_policy_fallback and not picker_model_ids
    if picker_model_ids or not allow_policy_fallback:
        available_model_ids = picker_model_ids
    else:
        available_model_ids = [m["id"] for m in account_models if m["policy_state"] == "enabled"]
    policy_model_ids = [
        m["id"]
        for m in account_models
        if m["policy_state"] == "unconfigured" and m["id"] in known and (m["picker_enabled"] or use_policy_fallback)
    ]
    return {"available_model_ids": available_model_ids, "policy_model_ids": policy_model_ids}


async def _fetch_with_rate_limit_retry(
    url: str,
    *,
    headers: Dict[str, str],
    signal: Optional[AbortSignal],
    max_retries: int,
    max_elapsed_ms: int,
    method: str = "GET",
    json_body: Any = None,
) -> Any:
    deadline = time.monotonic() + max_elapsed_ms / 1000 if max_retries > 0 and max_elapsed_ms > 0 else None
    retry = 0
    while True:
        # pi-ai caps each attempt at 5s.
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            request = client.request(method, url, headers=headers, json=json_body)
            response = await race_with_abort(request, signal)
        if response.status_code != 429 or retry == max_retries:
            return response

        retry_after = response.headers.get("retry-after")
        delay_ms = 500 * 2**retry
        if retry_after:
            try:
                delay_ms = float(retry_after) * 1000
            except ValueError:
                try:
                    delay_ms = (parsedate_to_datetime(retry_after).timestamp() - time.time()) * 1000
                except (TypeError, ValueError):
                    return response
        delay_ms = max(0.0, delay_ms)
        if deadline is not None and delay_ms >= (deadline - time.monotonic()) * 1000:
            return response
        await abortable_sleep(delay_ms / 1000, signal)
        retry += 1


async def fetch_github_copilot_models(
    copilot_token: str,
    enterprise_domain: Optional[str],
    signal: Optional[AbortSignal],
    *,
    max_retries: int = 0,
    max_elapsed_ms: int = 0,
) -> Dict[str, Any]:
    base_url = get_github_copilot_base_url(copilot_token, enterprise_domain)
    # Some Individual accounts return false for every picker flag despite explicit
    # enabled policies. Limit the fallback to that endpoint so other account types
    # keep strict picker semantics.
    allow_policy_fallback = base_url == INDIVIDUAL_BASE_URL
    response = await _fetch_with_rate_limit_retry(
        f"{base_url}/models",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {copilot_token}",
            **COPILOT_HEADERS,
            "X-GitHub-Api-Version": COPILOT_API_VERSION,
        },
        signal=signal,
        max_retries=max_retries,
        max_elapsed_ms=max_elapsed_ms,
    )
    if response.status_code >= 400:
        raise ValueError(f"{response.status_code} {response.reason_phrase}: {response.text}")
    return parse_github_copilot_model_catalog(response.json(), allow_policy_fallback)


async def _fetch_json(response: Any) -> Any:
    if response.status_code >= 400:
        raise ValueError(f"{response.status_code} {response.reason_phrase}: {response.text}")
    return response.json()


async def _start_device_flow(domain: str, signal: Optional[AbortSignal]) -> Dict[str, Any]:
    urls = _get_urls(domain)
    data = await _fetch_json(
        await post_form(
            urls["device_code_url"],
            {"client_id": CLIENT_ID, "scope": "read:user"},
            signal,
            headers={"User-Agent": "GitHubCopilotChat/0.35.0"},
        )
    )

    if not isinstance(data, dict):
        raise ValueError("Invalid device code response")
    device_code = data.get("device_code")
    user_code = data.get("user_code")
    verification_uri = data.get("verification_uri")
    interval = data.get("interval")
    expires_in = data.get("expires_in")
    if (
        not isinstance(device_code, str)
        or not isinstance(user_code, str)
        or not isinstance(verification_uri, str)
        or (interval is not None and not isinstance(interval, (int, float)))
        or not isinstance(expires_in, (int, float))
    ):
        raise ValueError("Invalid device code response fields")

    # The verification URI is opened in the user's browser; only http(s) URLs are trusted.
    parsed = urlsplit(verification_uri)
    if parsed.scheme not in ("https", "http") or not parsed.netloc:
        raise ValueError("Untrusted verification_uri in device code response")

    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": parsed.geturl(),
        "interval": interval,
        "expires_in": expires_in,
    }


async def _poll_for_github_access_token(
    domain: str, device: Dict[str, Any], signal: AbortSignal
) -> str:
    urls = _get_urls(domain)

    async def poll():
        raw = await _fetch_json(
            await post_form(
                urls["access_token_url"],
                {
                    "client_id": CLIENT_ID,
                    "device_code": device["device_code"],
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                signal,
                headers={"User-Agent": "GitHubCopilotChat/0.35.0"},
            )
        )

        if isinstance(raw, dict) and isinstance(raw.get("access_token"), str):
            return poll_complete(raw["access_token"])

        if isinstance(raw, dict) and isinstance(raw.get("error"), str):
            error = raw["error"]
            if error == "authorization_pending":
                return poll_pending()
            if error == "slow_down":
                interval = raw.get("interval")
                return poll_slow_down(interval if isinstance(interval, (int, float)) else None)
            description = raw.get("error_description")
            suffix = f": {description}" if isinstance(description, str) and description else ""
            return poll_failed(f"Device flow failed: {error}{suffix}")

        return poll_failed("Invalid device token response")

    return await poll_oauth_device_code_flow(
        poll=poll,
        signal=signal,
        interval_seconds=device["interval"],
        expires_in_seconds=device["expires_in"],
        wait_before_first_poll=True,
    )


async def refresh_github_copilot_access_token(
    refresh_token: str,
    enterprise_domain: Optional[str],
    signal: Optional[AbortSignal],
) -> OAuthCredential:
    domain = enterprise_domain or "github.com"
    urls = _get_urls(domain)
    raw = await _fetch_json(
        await get(
            urls["copilot_token_url"],
            signal,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {refresh_token}",
                **COPILOT_HEADERS,
            },
        )
    )

    if not isinstance(raw, dict):
        raise ValueError("Invalid Copilot token response")
    token = raw.get("token")
    expires_at = raw.get("expires_at")
    if not isinstance(token, str) or not isinstance(expires_at, (int, float)):
        raise ValueError("Invalid Copilot token response fields")

    extra = {"enterpriseUrl": enterprise_domain} if enterprise_domain else {}
    return OAuthCredential(
        refresh=refresh_token,
        access=token,
        expires=int(expires_at * 1000) - _TOKEN_EXPIRY_SKEW_MS,
        **extra,
    )


async def refresh_github_copilot_token(
    refresh_token: str,
    enterprise_domain: Optional[str],
    signal: Optional[AbortSignal],
) -> OAuthCredential:
    credentials = await refresh_github_copilot_access_token(refresh_token, enterprise_domain, signal)
    catalog = await fetch_github_copilot_models(
        credentials.access, enterprise_domain, signal, max_retries=0, max_elapsed_ms=0
    )
    return OAuthCredential(**{**credentials.model_dump(), "availableModelIds": catalog["available_model_ids"]})


async def _enable_github_copilot_model(
    token: str,
    model_id: str,
    enterprise_domain: Optional[str],
    signal: Optional[AbortSignal],
) -> bool:
    base_url = get_github_copilot_base_url(token, enterprise_domain)
    url = f"{base_url}/models/{model_id}/policy"
    try:
        response = await _fetch_with_rate_limit_retry(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                **COPILOT_HEADERS,
                "openai-intent": "chat-policy",
                "x-interaction-type": "chat-policy",
            },
            signal=signal,
            max_retries=2,
            max_elapsed_ms=5000,
            method="POST",
            json_body={"state": "enabled"},
        )
    except Exception:
        if signal is not None and signal.aborted:
            raise
        return False
    if response.status_code == 429:
        raise ValueError(f"{response.status_code} {response.reason_phrase}: {response.text}")
    return response.status_code < 400


async def _enable_github_copilot_models(
    token: str,
    model_ids: Iterable[str],
    enterprise_domain: Optional[str],
    signal: Optional[AbortSignal],
) -> "list[str]":
    enabled: "list[str]" = []
    for model_id in model_ids:
        try:
            if await _enable_github_copilot_model(token, model_id, enterprise_domain, signal):
                enabled.append(model_id)
        except Exception:
            if signal is not None and signal.aborted:
                raise
            # Policy updates are best effort; exhausted rate limiting stops the batch.
            break
    return enabled


async def _login_github_copilot(interaction: AuthInteraction) -> OAuthCredential:
    signal = interaction.signal
    input = await interaction.prompt(
        AuthPromptText(
            message="GitHub Enterprise URL/domain (blank for github.com)",
            placeholder="company.ghe.com",
        )
    )
    if signal is not None and signal.aborted:
        raise AbortError("Login cancelled")

    trimmed = input.strip()
    enterprise_domain = normalize_domain(input)
    if trimmed and not enterprise_domain:
        raise ValueError("Invalid GitHub Enterprise URL/domain")
    domain = enterprise_domain or "github.com"

    device = await _start_device_flow(domain, signal)
    interaction.notify(
        AuthEventDeviceCode(
            user_code=device["user_code"],
            verification_uri=device["verification_uri"],
            interval_seconds=int(device["interval"]) if device["interval"] is not None else None,
            expires_in_seconds=int(device["expires_in"]),
        )
    )

    github_access_token = await _poll_for_github_access_token(domain, device, signal or AbortSignal())
    credentials = await refresh_github_copilot_access_token(github_access_token, enterprise_domain, signal)
    catalog = await fetch_github_copilot_models(
        credentials.access, enterprise_domain, signal, max_retries=2, max_elapsed_ms=5000
    )
    enabled_model_ids: "list[str]" = []
    if catalog["policy_model_ids"]:
        interaction.notify(AuthEventProgress(message="Enabling models..."))
        enabled_model_ids = await _enable_github_copilot_models(
            credentials.access, catalog["policy_model_ids"], enterprise_domain, signal
        )
    available = list(dict.fromkeys([*catalog["available_model_ids"], *enabled_model_ids]))
    return OAuthCredential(**{**credentials.model_dump(), "availableModelIds": available})


def copilot_enterprise_domain(credential: OAuthCredential) -> Optional[str]:
    extra = credential.model_extra or {}
    enterprise_url = extra.get("enterpriseUrl")
    if not isinstance(enterprise_url, str) or not enterprise_url:
        return None
    return normalize_domain(enterprise_url)


async def _refresh(credential: OAuthCredential, signal: AbortSignal) -> OAuthCredential:
    return await refresh_github_copilot_token(credential.refresh, copilot_enterprise_domain(credential), signal)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    """Derive the credential-specific proxy endpoint for each request."""
    return ModelAuth(
        api_key=credential.access,
        base_url=get_github_copilot_base_url(credential.access, copilot_enterprise_domain(credential)),
    )


github_copilot_oauth = OAuthAuth(
    name="GitHub Copilot",
    is_subscription=True,
    login=_login_github_copilot,
    refresh=_refresh,
    to_auth=_to_auth,
)

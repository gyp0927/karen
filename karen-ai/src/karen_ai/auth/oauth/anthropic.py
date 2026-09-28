"""Anthropic OAuth flow (Claude Pro/Max), mirroring auth/oauth/anthropic.ts.

PKCE + localhost callback server on port 53692, raced against a manual code
prompt for remote/headless sessions. The PKCE verifier doubles as the OAuth
state parameter.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import traceback
from typing import Any, Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit

from ...abort import AbortController, AbortSignal
from ...utils.provider_env import get_provider_env_value
from ..types import AuthEventAuthUrl, AuthEventProgress, AuthInteraction, AuthPromptManualCode, ModelAuth, OAuthAuth, OAuthCredential
from ._http import post_json, race_with_abort
from .callback_server import CallbackHttpResponse, CallbackWaiter, OAuthCallbackServer, start_oauth_callback_server
from .oauth_page import oauth_error_html, oauth_success_html
from .pkce import generate_pkce

CLIENT_ID = base64.b64decode("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl").decode("ascii")
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CALLBACK_PORT = 53692
CALLBACK_PATH = "/callback"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"
SCOPES = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"

_EXPIRY_SKEW_MS = 5 * 60 * 1000


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


def _format_error_details(error: BaseException) -> str:
    details = [f"{type(error).__name__}: {error}"]
    cause = error.__cause__ if error.__cause__ is not None else error.__context__
    if cause is not None:
        details.append(f"cause={_format_error_details(cause)}")
    stack = "".join(traceback.format_exception(type(error), error, error.__traceback__)).strip()
    if stack:
        details.append(f"stack={stack}")
    return "; ".join(details)


async def _start_callback_server(expected_state: str) -> tuple[OAuthCallbackServer, CallbackWaiter[Dict[str, str]]]:
    waiter: CallbackWaiter[Dict[str, str]] = CallbackWaiter()

    def handler(path: str, query: Dict[str, str]) -> CallbackHttpResponse:
        if path != CALLBACK_PATH:
            return CallbackHttpResponse(404, oauth_error_html("Callback route not found."))
        code = query.get("code")
        state = query.get("state")
        error = query.get("error")
        if error:
            return CallbackHttpResponse(
                400, oauth_error_html("Anthropic authentication did not complete.", f"Error: {error}")
            )
        if not code or not state:
            return CallbackHttpResponse(400, oauth_error_html("Missing code or state parameter."))
        if state != expected_state:
            return CallbackHttpResponse(400, oauth_error_html("State mismatch."))
        waiter.settle({"code": code, "state": state})
        return CallbackHttpResponse(
            200, oauth_success_html("Anthropic authentication completed. You can close this window.")
        )

    server = await start_oauth_callback_server(_callback_host(), CALLBACK_PORT, handler)
    return server, waiter


async def _post_json(url: str, body: Dict[str, Any], signal: Optional[AbortSignal]) -> str:
    response = await post_json(url, body, signal, headers={"Content-Type": "application/json"})
    response_body = response.text
    if response.status_code >= 400:
        raise ValueError(f"HTTP request failed. status={response.status_code}; url={url}; body={response_body}")
    return response_body


async def exchange_authorization_code(
    code: str,
    state: str,
    verifier: str,
    redirect_uri: str,
    signal: Optional[AbortSignal],
) -> OAuthCredential:
    try:
        response_body = await _post_json(
            TOKEN_URL,
            {
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
            },
            signal,
        )
    except Exception as error:
        raise ValueError(
            f"Token exchange request failed. url={TOKEN_URL}; redirect_uri={redirect_uri}; "
            f"response_type=authorization_code; details={_format_error_details(error)}"
        ) from error

    try:
        token_data = json.loads(response_body)
    except ValueError as error:
        raise ValueError(
            f"Token exchange returned invalid JSON. url={TOKEN_URL}; body={response_body}; "
            f"details={_format_error_details(error)}"
        ) from error

    return OAuthCredential(
        refresh=token_data["refresh_token"],
        access=token_data["access_token"],
        expires=int(time.time() * 1000) + int(token_data["expires_in"]) * 1000 - _EXPIRY_SKEW_MS,
    )


async def _login_anthropic(interaction: AuthInteraction) -> OAuthCredential:
    signal = interaction.signal
    verifier, challenge = generate_pkce()
    server, waiter = await _start_callback_server(verifier)
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
        waiter.settle(None)

    manual_task = asyncio.create_task(_manual_prompt())
    try:
        auth_params = urlencode(
            {
                "code": "true",
                "client_id": CLIENT_ID,
                "response_type": "code",
                "redirect_uri": REDIRECT_URI,
                "scope": SCOPES,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": verifier,
            }
        )
        interaction.notify(
            AuthEventAuthUrl(
                url=f"{AUTHORIZE_URL}?{auth_params}",
                instructions=(
                    "Complete login in your browser. If the browser is on another machine, "
                    "paste the final redirect URL here."
                ),
            )
        )

        result = await race_with_abort(waiter.wait(), signal)
        if manual.get("error") is not None:
            raise manual["error"]

        code: Optional[str] = None
        state: Optional[str] = None
        if result and result.get("code"):
            code = result["code"]
            state = result["state"]
        elif manual.get("input"):
            parsed = parse_authorization_input(manual["input"])
            if parsed.get("state") and parsed["state"] != verifier:
                raise ValueError("OAuth state mismatch")
            code = parsed.get("code")
            state = parsed.get("state") or verifier

        if not code:
            raise ValueError("Missing authorization code")
        if not state:
            raise ValueError("Missing OAuth state")
        interaction.notify(AuthEventProgress(message="Exchanging authorization code for tokens..."))
        return await exchange_authorization_code(code, state, verifier, REDIRECT_URI, signal)
    finally:
        manual_controller.abort()
        manual_task.cancel()
        try:
            await manual_task
        except BaseException:
            pass
        server.close()


async def refresh_anthropic_token(refresh_token: str, signal: Optional[AbortSignal]) -> OAuthCredential:
    try:
        response_body = await _post_json(
            TOKEN_URL,
            {"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": refresh_token},
            signal,
        )
    except Exception as error:
        raise ValueError(
            f"Anthropic token refresh request failed. url={TOKEN_URL}; details={_format_error_details(error)}"
        ) from error

    try:
        data = json.loads(response_body)
    except ValueError as error:
        raise ValueError(
            f"Anthropic token refresh returned invalid JSON. url={TOKEN_URL}; body={response_body}; "
            f"details={_format_error_details(error)}"
        ) from error

    return OAuthCredential(
        refresh=data["refresh_token"],
        access=data["access_token"],
        expires=int(time.time() * 1000) + int(data["expires_in"]) * 1000 - _EXPIRY_SKEW_MS,
    )


async def _refresh(credential: OAuthCredential, signal: AbortSignal) -> OAuthCredential:
    return await refresh_anthropic_token(credential.refresh, signal)


async def _to_auth(credential: OAuthCredential) -> ModelAuth:
    return ModelAuth(api_key=credential.access)


anthropic_oauth = OAuthAuth(
    name="Anthropic (Claude Pro/Max)",
    is_subscription=True,
    login=_login_anthropic,
    refresh=_refresh,
    to_auth=_to_auth,
)

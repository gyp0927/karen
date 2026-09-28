"""Auth resolution shared by all operations in a Models collection.

Mirrors pi-ai's auth/resolve.ts: a stored credential owns the provider —
ambient/env is consulted only when nothing is stored. OAuth tokens with less
than five minutes remaining refresh once under the credential-store lock.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from pydantic import ConfigDict

from ..abort import AbortSignal, operation_signal
from ..errors import ModelsError
from ..types import KarenBase, ProviderEnv
from .context import AuthContext
from .types import (
    ApiKeyAuth,
    ApiKeyCredential,
    ApiKeyResolveInput,
    AuthResult,
    Credential,
    CredentialStore,
    OAuthAuth,
    OAuthCredential,
    ProviderAuth,
)

DEFAULT_OAUTH_MINIMUM_VALIDITY_MS = 5 * 60 * 1000
DEFAULT_OAUTH_REFRESH_TIMEOUT_MS = 15_000


class AuthResolutionOverrides(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: Optional[str] = None
    env: Optional[ProviderEnv] = None
    min_oauth_validity_ms: Optional[int] = None
    signal: Optional[AbortSignal] = None


class _OverlayEnvAuthContext:
    def __init__(self, base: AuthContext, env: ProviderEnv) -> None:
        self._base = base
        self._env = env

    async def env(self, name: str) -> Optional[str]:
        value = self._env.get(name)
        if value:
            return value
        return await self._base.env(name)

    async def file_exists(self, path: str) -> bool:
        return await self._base.file_exists(path)


async def resolve_provider_auth(
    provider_id: str,
    auth: ProviderAuth,
    credentials: CredentialStore,
    auth_context: AuthContext,
    overrides: Optional[AuthResolutionOverrides] = None,
) -> Optional[AuthResult]:
    signal = operation_signal(overrides.signal if overrides else None)
    signal.throw_if_aborted()
    request_ctx = (
        _OverlayEnvAuthContext(auth_context, overrides.env)
        if overrides and overrides.env
        else auth_context
    )

    if overrides and overrides.api_key is not None and auth.api_key:
        return await _resolve_api_key(
            request_ctx,
            auth.api_key,
            provider_id,
            ApiKeyCredential(key=overrides.api_key, env=overrides.env),
            signal,
        )

    stored = await _read_credential(credentials, provider_id, signal)
    if stored is not None:
        if isinstance(stored, OAuthCredential) and auth.oauth:
            return await _resolve_stored_oauth(
                credentials,
                provider_id,
                auth.oauth,
                stored,
                signal,
                overrides.min_oauth_validity_ms if overrides else None,
            )
        if isinstance(stored, ApiKeyCredential) and auth.api_key:
            credential = stored
            if overrides and overrides.env:
                credential = ApiKeyCredential(key=stored.key, env={**(stored.env or {}), **overrides.env})
            return await _resolve_api_key(request_ctx, auth.api_key, provider_id, credential, signal)
        return None

    # Ambient (env vars, credential files).
    if auth.api_key:
        return await _resolve_api_key(request_ctx, auth.api_key, provider_id, None, signal)
    return None


async def _resolve_stored_oauth(
    credentials: CredentialStore,
    provider_id: str,
    oauth: OAuthAuth,
    stored: OAuthCredential,
    signal: AbortSignal,
    min_oauth_validity_ms: Optional[int],
) -> Optional[AuthResult]:
    minimum_validity_ms = max(DEFAULT_OAUTH_MINIMUM_VALIDITY_MS, min_oauth_validity_ms or 0)

    def expires_soon(credential: OAuthCredential) -> bool:
        return time.time() * 1000 + minimum_validity_ms >= credential.expires

    credential = stored

    if expires_soon(credential):
        async def refresh_if_needed(current: Optional[Credential]) -> Optional[Credential]:
            if not isinstance(current, OAuthCredential):
                return None  # logged out meanwhile
            if not expires_soon(current):
                return None  # another process/request refreshed
            try:
                return await asyncio.wait_for(oauth.refresh(current, signal), timeout=DEFAULT_OAUTH_REFRESH_TIMEOUT_MS / 1000)
            except Exception as error:
                raise ModelsError("oauth", f"OAuth refresh failed for {provider_id}", cause=error)

        try:
            post = await credentials.modify(provider_id, refresh_if_needed)
        except ModelsError:
            raise
        except Exception as error:
            raise ModelsError("auth", f"Credential store modify failed for {provider_id}", cause=error)
        if not isinstance(post, OAuthCredential):
            return None  # logged out meanwhile
        credential = post
        if min_oauth_validity_ms is not None and expires_soon(credential):
            raise ModelsError("oauth", f"OAuth refresh returned a token that expires too soon for {provider_id}")

    try:
        return AuthResult(auth=await oauth.to_auth(credential), source="OAuth")
    except Exception as error:
        raise ModelsError("oauth", f"OAuth auth derivation failed for {provider_id}", cause=error)


async def _resolve_api_key(
    auth_context,
    api_key: ApiKeyAuth,
    provider_id: str,
    credential: Optional[ApiKeyCredential],
    signal: AbortSignal,
) -> Optional[AuthResult]:
    try:
        return await api_key.resolve(ApiKeyResolveInput(ctx=auth_context, credential=credential, signal=signal))
    except Exception as error:
        raise ModelsError("auth", f"API key auth failed for provider {provider_id}", cause=error)


async def _read_credential(
    credentials: CredentialStore,
    provider_id: str,
    signal: AbortSignal,
) -> Optional[Credential]:
    try:
        return await credentials.read(provider_id)
    except Exception as error:
        raise ModelsError("auth", f"Credential store read failed for {provider_id}", cause=error)

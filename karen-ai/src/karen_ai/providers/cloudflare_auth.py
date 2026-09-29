"""Cloudflare auth resolvers, mirroring providers/cloudflare-auth.ts.

Both Cloudflare providers need an API key plus account id (and a gateway id for
AI Gateway). Resolution merges per-field: a stored credential may carry only the
key while the ids come from ambient env.
"""

from __future__ import annotations

from typing import Optional

from ..auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    ApiKeyResolveInput,
    AuthContextLike,
    AuthPromptSecret,
    AuthPromptText,
    AuthResult,
    ModelAuth,
)
from ..types import ProviderEnv

CLOUDFLARE_API_KEY = "CLOUDFLARE_API_KEY"
CLOUDFLARE_ACCOUNT_ID = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_GATEWAY_ID = "CLOUDFLARE_GATEWAY_ID"


async def _resolve_value(
    name: str,
    ctx: AuthContextLike,
    credential: Optional[ApiKeyCredential],
    signal,
) -> Optional[str]:
    if credential is not None:
        from_credential = credential.key if name == CLOUDFLARE_API_KEY else (credential.env or {}).get(name)
        if from_credential is not None:
            return from_credential
    signal.throw_if_aborted()
    value = await ctx.env(name)
    signal.throw_if_aborted()
    return value


async def _resolve_cloudflare_env(
    kind: str,  # "workers-ai" | "ai-gateway"
    ctx: AuthContextLike,
    credential: Optional[ApiKeyCredential],
    signal,
) -> Optional[tuple[str, ProviderEnv, str]]:
    api_key = await _resolve_value(CLOUDFLARE_API_KEY, ctx, credential, signal)
    account_id = await _resolve_value(CLOUDFLARE_ACCOUNT_ID, ctx, credential, signal)
    gateway_id = (
        await _resolve_value(CLOUDFLARE_GATEWAY_ID, ctx, credential, signal) if kind == "ai-gateway" else None
    )
    if not api_key or not account_id or (kind == "ai-gateway" and not gateway_id):
        return None
    env: ProviderEnv = {CLOUDFLARE_ACCOUNT_ID: account_id}
    if gateway_id:
        env[CLOUDFLARE_GATEWAY_ID] = gateway_id
    return api_key, env, "stored credential" if credential else CLOUDFLARE_API_KEY


def cloudflare_workers_ai_auth() -> ApiKeyAuth:
    async def login(interaction) -> ApiKeyCredential:
        key = await interaction.prompt(AuthPromptSecret(message="Enter Cloudflare API key"))
        account_id = await interaction.prompt(AuthPromptText(message="Enter Cloudflare account ID"))
        return ApiKeyCredential(key=key, env={CLOUDFLARE_ACCOUNT_ID: account_id})

    async def resolve(input: ApiKeyResolveInput) -> Optional[AuthResult]:
        resolved = await _resolve_cloudflare_env("workers-ai", input.ctx, input.credential, input.signal)
        if not resolved:
            return None
        api_key, env, source = resolved
        return AuthResult(auth=ModelAuth(api_key=api_key), env=env, source=source)

    return ApiKeyAuth(name="Cloudflare API key", login=login, resolve=resolve)


def cloudflare_ai_gateway_auth() -> ApiKeyAuth:
    async def login(interaction) -> ApiKeyCredential:
        key = await interaction.prompt(AuthPromptSecret(message="Enter Cloudflare API key"))
        account_id = await interaction.prompt(AuthPromptText(message="Enter Cloudflare account ID"))
        gateway_id = await interaction.prompt(AuthPromptText(message="Enter Cloudflare AI Gateway ID"))
        return ApiKeyCredential(
            key=key, env={CLOUDFLARE_ACCOUNT_ID: account_id, CLOUDFLARE_GATEWAY_ID: gateway_id}
        )

    async def resolve(input: ApiKeyResolveInput) -> Optional[AuthResult]:
        resolved = await _resolve_cloudflare_env("ai-gateway", input.ctx, input.credential, input.signal)
        if not resolved:
            return None
        api_key, env, source = resolved
        return AuthResult(
            auth=ModelAuth(
                headers={
                    "cf-aig-authorization": f"Bearer {api_key}",
                    "Authorization": None,
                    "x-api-key": None,
                }
            ),
            env=env,
            source=source,
        )

    return ApiKeyAuth(name="Cloudflare API key", login=login, resolve=resolve)

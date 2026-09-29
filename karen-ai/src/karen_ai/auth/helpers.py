"""Shared auth helpers, mirroring pi-ai's auth/helpers.ts."""

from __future__ import annotations

from typing import Optional, Sequence

from ..errors import AbortError
from .types import ApiKeyAuth, ApiKeyCredential, ApiKeyResolveInput, AuthResult, ModelAuth


def env_api_key_auth(name: str, env_vars: Sequence[str]) -> ApiKeyAuth:
    """Api-key auth resolving from a stored credential or the first set env var."""

    async def login(interaction) -> ApiKeyCredential:
        from .types import AuthPromptSecret

        if getattr(interaction, "signal", None) and interaction.signal.aborted:
            raise AbortError()
        key = await interaction.prompt(AuthPromptSecret(message=f"Enter {name}"))
        return ApiKeyCredential(key=key)

    async def resolve(input: ApiKeyResolveInput) -> Optional[AuthResult]:
        input.signal.throw_if_aborted()
        if input.credential and input.credential.key:
            return AuthResult(
                auth=ModelAuth(api_key=input.credential.key), env=input.credential.env, source="stored credential"
            )
        for env_var in env_vars:
            value = await input.ctx.env(env_var)
            input.signal.throw_if_aborted()
            if value:
                return AuthResult(auth=ModelAuth(api_key=value), source=env_var)
        return None

    return ApiKeyAuth(name=name, login=login, resolve=resolve)

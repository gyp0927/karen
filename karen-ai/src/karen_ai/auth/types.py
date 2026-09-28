"""Auth types, mirroring pi-ai's auth/types.ts."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Protocol, Union, runtime_checkable

from pydantic import ConfigDict

from ..abort import AbortSignal
from ..types import KarenBase, ProviderEnv, ProviderHeaders


class ModelAuth(KarenBase):
    """Request auth for a single model request.

    If a value cannot be expressed as api_key, headers, or base_url, it is
    provider config, not auth.
    """

    api_key: Optional[str] = None
    headers: Optional[ProviderHeaders] = None
    base_url: Optional[str] = None


class ApiKeyCredential(KarenBase):
    """Stored api-key credential. `env` holds provider-scoped environment/config values."""

    type: Literal["api_key"] = "api_key"
    key: Optional[str] = None
    env: Optional[ProviderEnv] = None


class OAuthCredential(KarenBase):
    """Stored canonical OAuth credential."""

    model_config = ConfigDict(extra="allow")

    type: Literal["oauth"] = "oauth"
    refresh: str
    access: str
    expires: int  # Unix timestamp in milliseconds


Credential = Union[ApiKeyCredential, OAuthCredential]


class CredentialInfo(KarenBase):
    """Non-secret credential metadata for account/status enumeration."""

    provider_id: str
    type: Literal["api_key", "oauth"]


class AuthOperationOptions(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal: Optional[AbortSignal] = None


@runtime_checkable
class CredentialStore(Protocol):
    """App-owned credential storage, keyed by provider id, one credential per provider.

    `modify` is the only write path, so every mutation is a serialized
    read-modify-write.
    """

    async def read(self, provider_id: str, options: Optional[AuthOperationOptions] = None) -> Optional[Credential]: ...

    async def list(self, options: Optional[AuthOperationOptions] = None) -> List[CredentialInfo]: ...

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Optional[Credential]], Awaitable[Optional[Credential]]],
        options: Optional[AuthOperationOptions] = None,
    ) -> Optional[Credential]: ...

    async def delete(self, provider_id: str, options: Optional[AuthOperationOptions] = None) -> None: ...


class AuthResult(KarenBase):
    """Result of resolving auth for a model."""

    auth: ModelAuth
    env: Optional[ProviderEnv] = None
    source: Optional[str] = None  # Human-readable label for status UI


class AuthCheck(KarenBase):
    source: Optional[str] = None
    type: Literal["api_key", "oauth"]


AuthType = Literal["api_key", "oauth"]


# -- Login interaction --------------------------------------------------------


class _AuthPromptBase(KarenBase):
    """Prompts carry an optional per-prompt signal so a flow can cancel one
    prompt (e.g. a `manual_code` prompt raced against a callback server)
    without aborting the whole login."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal: Optional[AbortSignal] = None


class AuthPromptText(_AuthPromptBase):
    type: Literal["text"] = "text"
    message: str
    placeholder: Optional[str] = None


class AuthPromptSecret(_AuthPromptBase):
    type: Literal["secret"] = "secret"
    message: str
    placeholder: Optional[str] = None


class AuthSelectOption(KarenBase):
    id: str
    label: str
    description: Optional[str] = None


class AuthPromptSelect(_AuthPromptBase):
    type: Literal["select"] = "select"
    message: str
    options: List[AuthSelectOption]


class AuthPromptManualCode(_AuthPromptBase):
    type: Literal["manual_code"] = "manual_code"
    message: str
    placeholder: Optional[str] = None


AuthPrompt = Union[AuthPromptText, AuthPromptSecret, AuthPromptSelect, AuthPromptManualCode]


class AuthInfoLink(KarenBase):
    url: str
    label: Optional[str] = None


class AuthEventInfo(KarenBase):
    type: Literal["info"] = "info"
    message: str
    links: Optional[List[AuthInfoLink]] = None


class AuthEventAuthUrl(KarenBase):
    type: Literal["auth_url"] = "auth_url"
    url: str
    instructions: Optional[str] = None


class AuthEventDeviceCode(KarenBase):
    type: Literal["device_code"] = "device_code"
    user_code: str
    verification_uri: str
    interval_seconds: Optional[int] = None
    expires_in_seconds: Optional[int] = None


class AuthEventProgress(KarenBase):
    type: Literal["progress"] = "progress"
    message: str


AuthEvent = Union[AuthEventInfo, AuthEventAuthUrl, AuthEventDeviceCode, AuthEventProgress]


class AuthInteraction(Protocol):
    """Login interaction callbacks serving both api-key and OAuth flows."""

    signal: Optional[AbortSignal]

    async def prompt(self, prompt: AuthPrompt) -> str: ...

    def notify(self, event: AuthEvent) -> None: ...


# -- Provider auth methods ----------------------------------------------------


class ApiKeyResolveInput(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ctx: "AuthContextLike"
    credential: Optional[ApiKeyCredential] = None
    signal: AbortSignal


@runtime_checkable
class AuthContextLike(Protocol):
    """Environment access for auth resolution. Injectable for tests."""

    async def env(self, name: str) -> Optional[str]: ...

    async def file_exists(self, path: str) -> bool: ...


ApiKeyResolveInput.model_rebuild()


class ApiKeyAuth:
    """Api-key auth: stored key/provider env plus ambient sources (env vars, files).

    Ambient-only providers omit `login`.
    """

    def __init__(
        self,
        name: str,
        resolve: Callable[[ApiKeyResolveInput], Awaitable[Optional[AuthResult]]],
        login: Optional[Callable[[AuthInteraction], Awaitable[ApiKeyCredential]]] = None,
        check: Optional[Callable[[ApiKeyResolveInput], Awaitable[Optional[AuthCheck]]]] = None,
    ) -> None:
        #: Display name, e.g. "Anthropic API key".
        self.name = name
        self.resolve = resolve
        self.login = login
        self.check = check


class OAuthAuth:
    """OAuth auth. The refresh/to_auth split lets Models own the locked refresh pattern."""

    def __init__(
        self,
        name: str,
        login: Callable[[AuthInteraction], Awaitable[OAuthCredential]],
        refresh: Callable[[OAuthCredential, AbortSignal], Awaitable[OAuthCredential]],
        to_auth: Callable[[OAuthCredential], Awaitable[ModelAuth]],
        is_subscription: Optional[bool] = None,
        login_label: Optional[str] = None,
    ) -> None:
        #: Display name, e.g. "Anthropic (Claude Pro/Max)".
        self.name = name
        self.is_subscription = is_subscription
        self.login_label = login_label
        self.login = login
        self.refresh = refresh
        self.to_auth = to_auth


class ProviderAuth(KarenBase):
    """Provider auth. At least one of api_key/oauth must be present."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: Optional[ApiKeyAuth] = None
    oauth: Optional[OAuthAuth] = None

"""OAuth error taxonomy (pi's `oauth/errors.ts`)."""

from __future__ import annotations

from typing import Optional

__all__ = [
    "McpOAuthAuthorizationRequiredError",
    "OAuthError",
    "OAuthInsecureEndpointError",
    "OAuthIssuerMismatchError",
    "OAuthRegistrationError",
]


class OAuthError(Exception):
    """An OAuth error response (`error`/`error_description`/`error_uri`)."""

    def __init__(self, code: str, message: str, error_uri: Optional[str] = None) -> None:
        super().__init__(message or code)
        self.code = code
        self.error_uri = error_uri


class OAuthIssuerMismatchError(Exception):
    """RFC 9207: the issuer an authorization response or document names is not
    the server it came from."""

    def __init__(self, expected: str, received: Optional[str]) -> None:
        super().__init__(
            "OAuth issuer mismatch: expected "
            + repr(expected)
            + ", received "
            + ("none" if received is None else repr(received))
        )
        self.expected = expected
        #: `None` when the response lacked the `iss` parameter its server
        #: promised.
        self.received = received


class OAuthInsecureEndpointError(Exception):
    def __init__(self, endpoint: str) -> None:
        super().__init__(f"Refusing to send OAuth credentials to non-HTTPS endpoint {endpoint}")
        self.endpoint = endpoint


class OAuthRegistrationError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"OAuth dynamic client registration failed with status {status}: {body}")
        self.status = status
        self.body = body


class McpOAuthAuthorizationRequiredError(Exception):
    def __init__(self) -> None:
        super().__init__("MCP OAuth authorization requires user interaction")

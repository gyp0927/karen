"""The MCP OAuth client subset (pi's `oauth/`, itself adapted from the MCP
TypeScript SDK v1.29.0, MIT).

Documents stay dicts in their wire shape — camelCase server metadata, the
snake_case fields of a token response — with the parsers validating the known
members and passing the rest through. That keeps a persisted discovery cache
JSON-round-trippable and matches what pi's state store persists.
"""

from .callback import OAuthCallback, OAuthCallbackPage, OAuthCallbackServer, OAuthCallbackServerOptions
from .discovery import (
    build_authorization_server_discovery_urls,
    discover_authorization_server_metadata,
    discover_oauth_server_info,
    discover_protected_resource_metadata,
    parse_www_authenticate,
    resource_url_from_server_url,
    select_resource,
)
from .errors import (
    McpOAuthAuthorizationRequiredError,
    OAuthError,
    OAuthInsecureEndpointError,
    OAuthIssuerMismatchError,
    OAuthRegistrationError,
)
from .flow import (
    AddClientAuthentication,
    OAuthFlowOptions,
    adapt_oauth_provider,
    authorize_mcp,
    exchange_authorization_code,
    refresh_authorization,
    register_client,
    start_authorization,
    step_up_scope,
)
from .provider import MemoryOAuthStateStore, McpOAuthProvider, McpOAuthProviderOptions
from .types import (
    parse_authorization_server_metadata,
    parse_client_information,
    parse_oauth_tokens,
    parse_protected_resource_metadata,
)

__all__ = [
    "AddClientAuthentication",
    "MemoryOAuthStateStore",
    "McpOAuthAuthorizationRequiredError",
    "McpOAuthProvider",
    "McpOAuthProviderOptions",
    "OAuthCallback",
    "OAuthCallbackPage",
    "OAuthCallbackServer",
    "OAuthCallbackServerOptions",
    "OAuthError",
    "OAuthFlowOptions",
    "OAuthInsecureEndpointError",
    "OAuthIssuerMismatchError",
    "OAuthRegistrationError",
    "adapt_oauth_provider",
    "authorize_mcp",
    "build_authorization_server_discovery_urls",
    "discover_authorization_server_metadata",
    "discover_oauth_server_info",
    "discover_protected_resource_metadata",
    "exchange_authorization_code",
    "parse_authorization_server_metadata",
    "parse_client_information",
    "parse_oauth_tokens",
    "parse_protected_resource_metadata",
    "parse_www_authenticate",
    "refresh_authorization",
    "register_client",
    "resource_url_from_server_url",
    "select_resource",
    "start_authorization",
    "step_up_scope",
]

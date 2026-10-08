"""OAuth discovery (pi's `oauth/discovery.ts`).

Protected-resource metadata under `/.well-known/oauth-protected-resource`,
authorization-server metadata under the two well-known paths (plus the OIDC
suffix variant), and the `WWW-Authenticate` challenge parser that points the
flow at them.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from ..auth_provider import McpFetch
from ..protocol.types import LATEST_PROTOCOL_VERSION
from ..transports.http_client import HttpRequest, HttpResponse, http_fetch
from .errors import OAuthIssuerMismatchError
from .types import _is_usable_url, parse_authorization_server_metadata, parse_protected_resource_metadata

__all__ = [
    "build_authorization_server_discovery_urls",
    "discover_authorization_server_metadata",
    "discover_oauth_server_info",
    "discover_protected_resource_metadata",
    "parse_www_authenticate",
    "resource_url_from_server_url",
    "select_resource",
]


def _is_discovery_miss(status: int) -> bool:
    """4xx and 502 mean "not here", so discovery tries the next candidate URL."""
    return 400 <= status < 500 or status == 502


def _path_suffix(pathname: str) -> str:
    """Path suffix for `/.well-known/<kind><path>`; empty for the root path."""
    return pathname[:-1] if pathname.endswith("/") else pathname


def _default_port(scheme: str) -> Optional[int]:
    return {"http": 80, "https": 443, "ws": 80, "wss": 443}.get(scheme.lower())


def _origin(parts: Any) -> str:
    """WHATWG `URL.origin`: lowercase scheme and host, default port dropped."""
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    if port is not None and port != _default_port(scheme):
        host = f"{host}:{port}"
    return f"{scheme}://{host}"


def _field(header: str, name: str) -> Optional[str]:
    match = re.search(r'(?:^|[,\s])' + re.escape(name) + r'=(?:"([^"]*)"|([^\s,]+))', header, re.IGNORECASE)
    if match is None:
        return None
    # An empty value (`scope=""`) carries no information, so it counts as absent.
    return match.group(1) or match.group(2) or None


def parse_www_authenticate(header: Optional[str]) -> Dict[str, Any]:
    """A bearer challenge as `resource_metadata_url`/`scope`/`error`/
    `error_description`, only the members that carry a value."""
    challenge: Dict[str, Any] = {}
    if not header:
        return challenge
    parts = header.lstrip().split(None, 1)
    scheme = parts[0].lower() if parts else ""
    if scheme not in ("bearer", "dpop"):
        return challenge
    resource_metadata = _field(header, "resource_metadata")
    # Like `new URL(...)` in a try block: a malformed value is dropped, so
    # discovery falls back to the well-known path.
    if resource_metadata and _is_usable_url(resource_metadata):
        challenge["resource_metadata_url"] = resource_metadata
    for key in ("scope", "error", "error_description"):
        value = _field(header, key)
        if value:
            challenge[key] = value
    return challenge


async def _fetch_metadata(url: str, fetch: McpFetch, protocol_version: str) -> HttpResponse:
    return await fetch(
        url,
        HttpRequest(method="GET", headers={"Accept": "application/json", "MCP-Protocol-Version": protocol_version}),
    )


async def discover_protected_resource_metadata(
    server_url: str,
    *,
    resource_metadata_url: Optional[str] = None,
    protocol_version: Optional[str] = None,
    fetch: Optional[McpFetch] = None,
) -> Dict[str, Any]:
    server = urlsplit(server_url)
    use_fetch = fetch or http_fetch
    version = protocol_version or LATEST_PROTOCOL_VERSION
    origin = _origin(server)
    if resource_metadata_url is not None:
        url = str(resource_metadata_url)
    else:
        url = f"{origin}/.well-known/oauth-protected-resource{_path_suffix(server.path)}"
    response = await _fetch_metadata(url, use_fetch, version)
    if resource_metadata_url is None and server.path not in ("", "/") and _is_discovery_miss(response.status):
        await response.close()
        response = await _fetch_metadata(f"{origin}/.well-known/oauth-protected-resource", use_fetch, version)
    if not response.ok:
        await response.close()
        raise Exception(f"HTTP {response.status} loading OAuth protected resource metadata")
    return parse_protected_resource_metadata(await response.json())


def build_authorization_server_discovery_urls(authorization_server_url: str) -> List[Tuple[str, str]]:
    """`(url, "oauth" | "oidc")` candidates, in the order discovery tries them."""
    issuer = urlsplit(authorization_server_url)
    origin = _origin(issuer)
    path = _path_suffix(issuer.path)
    urls = [
        (f"{origin}/.well-known/oauth-authorization-server{path}", "oauth"),
        (f"{origin}/.well-known/openid-configuration{path}", "oidc"),
    ]
    if path:
        urls.append((f"{origin}{path}/.well-known/openid-configuration", "oidc"))
    return urls


async def discover_authorization_server_metadata(
    authorization_server_url: str,
    *,
    fetch: Optional[McpFetch] = None,
    protocol_version: Optional[str] = None,
    skip_issuer_validation: bool = False,
) -> Optional[Dict[str, Any]]:
    use_fetch = fetch or http_fetch
    version = protocol_version or LATEST_PROTOCOL_VERSION
    for url, _kind in build_authorization_server_discovery_urls(authorization_server_url):
        response = await _fetch_metadata(url, use_fetch, version)
        if not response.ok:
            await response.close()
            if _is_discovery_miss(response.status):
                continue
            raise Exception(f"HTTP {response.status} loading authorization server metadata from {url}")
        metadata = parse_authorization_server_metadata(await response.json())
        if not skip_issuer_validation:
            expected = str(authorization_server_url)
            # URL parsing adds a trailing slash to bare origins, so compare
            # without one on either side.
            trim = lambda value: value[:-1] if value.endswith("/") else value
            if trim(metadata["issuer"]) != trim(expected):
                raise OAuthIssuerMismatchError(expected, metadata["issuer"])
        return metadata
    return None


async def discover_oauth_server_info(
    server_url: str,
    *,
    resource_metadata_url: Optional[str] = None,
    #: Metadata document to use instead of discovery. It is trusted as
    #: configured, so its issuer is not checked.
    authorization_server_metadata_url: Optional[str] = None,
    fetch: Optional[McpFetch] = None,
    skip_issuer_validation: bool = False,
) -> Dict[str, Any]:
    """`authorization_server_url`, `authorization_server_metadata` (or None),
    and `resource_metadata` (or None)."""
    resource_metadata: Optional[Dict[str, Any]] = None
    try:
        resource_metadata = await discover_protected_resource_metadata(
            server_url, resource_metadata_url=resource_metadata_url, fetch=fetch
        )
    except Exception as error:
        # Like pi, only a network failure (fetch's TypeError) is fatal here;
        # a server without protected-resource metadata just has none.
        if isinstance(error, (OSError, EOFError)):
            raise
    if authorization_server_metadata_url is not None:
        url = str(authorization_server_metadata_url)
        response = await _fetch_metadata(url, fetch or http_fetch, LATEST_PROTOCOL_VERSION)
        if not response.ok:
            await response.close()
            raise Exception(f"HTTP {response.status} loading authorization server metadata from {url}")
        metadata = parse_authorization_server_metadata(await response.json())
        return {
            "authorization_server_url": metadata["issuer"],
            "authorization_server_metadata": metadata,
            "resource_metadata": resource_metadata,
        }
    servers = (resource_metadata or {}).get("authorization_servers")
    if servers:
        authorization_server_url = servers[0]
    else:
        parts = urlsplit(server_url)
        authorization_server_url = f"{_origin(parts)}/"
    return {
        "authorization_server_url": authorization_server_url,
        "authorization_server_metadata": await discover_authorization_server_metadata(
            authorization_server_url, fetch=fetch, skip_issuer_validation=skip_issuer_validation
        ),
        "resource_metadata": resource_metadata,
    }


def resource_url_from_server_url(value: str) -> str:
    """The server URL without its fragment, as OAuth `resource` parameters use
    it."""
    parts = urlsplit(str(value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def select_resource(server_url: str, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """The `resource` to request, checked against the server URL so a hostile
    metadata document cannot redirect tokens elsewhere."""
    if not metadata:
        return None
    requested_text = resource_url_from_server_url(server_url)
    requested = urlsplit(requested_text)
    configured = urlsplit(metadata["resource"])
    # Compared as origins, so a host's case and an explicit default port do
    # not turn a matching server into a rejection.
    if _origin(requested) != _origin(configured):
        raise Exception(f"Protected resource {metadata['resource']} does not match MCP server {requested_text}")
    requested_path = requested.path if requested.path.endswith("/") else requested.path + "/"
    configured_path = configured.path if configured.path.endswith("/") else configured.path + "/"
    if not requested_path.startswith(configured_path):
        raise Exception(f"Protected resource {metadata['resource']} does not match MCP server {requested_text}")
    return metadata["resource"]

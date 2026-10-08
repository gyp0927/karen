"""Parsers for OAuth documents (pi's `oauth/types.ts`).

Each parser takes the decoded JSON of one document and returns a dict with the
known members validated and normalized and the unknown members passed through
(token responses are the exception: unknown members are dropped, as pi does).
"Compact" means members a server sends as `null` or `""` are absent in the
result, since servers send them for fields they have no value for.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

__all__ = [
    "parse_authorization_server_metadata",
    "parse_client_information",
    "parse_oauth_tokens",
    "parse_protected_resource_metadata",
]

#: URL schemes `safeUrl` refuses (pi's list).
_UNSAFE_SCHEMES = ("javascript:", "data:", "vbscript:")


def _object(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {name}")
    return value


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) == 0:
        raise ValueError(f"Invalid {name}")
    return value


def _absent(value: Any) -> bool:
    """`null` and `""` count as absent: servers send them for fields they have
    no value for, like `scope: ""`."""
    return value is None or value == ""


def _optional_string(value: Any, name: str) -> Optional[str]:
    if _absent(value):
        return None
    return _required_string(value, name)


def _optional_strings(value: Any, name: str) -> Optional[List[str]]:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Invalid {name}")
    return list(value)


def _is_usable_url(text: str) -> bool:
    """What `URL.canParse` accepts, for the shapes this package handles: an
    absolute URL that is not a scripting scheme, with a usable authority when
    it has one."""
    parts = urlsplit(text)
    if not parts.scheme or parts.scheme + ":" in _UNSAFE_SCHEMES:
        return False
    if "://" in text:
        host = parts.hostname
        # `new URL` rejects a space in the host and an out-of-range port;
        # `urlsplit` accepts both until the port is read.
        if not host or any(char.isspace() for char in host):
            return False
        try:
            parts.port
        except ValueError:
            return False
    return True


def _safe_url(value: Any, name: str) -> str:
    text = _required_string(value, name)
    if not _is_usable_url(text):
        raise ValueError(f"Invalid {name}")
    return text


def _optional_url(value: Any, name: str) -> Optional[str]:
    if _absent(value):
        return None
    return _safe_url(value, name)


def _js_number(value: Any) -> float:
    """JavaScript's `Number()` for the values a JSON document can carry:
    booleans and non-numeric strings are not finite. (Numeric-looking strings
    JavaScript accepts, like `"0x10"`, are rejected rather than parsed.)"""
    if isinstance(value, bool):
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return math.nan
    return math.nan


def parse_protected_resource_metadata(value: Any) -> Dict[str, Any]:
    result = dict(_object(value, "OAuth protected resource metadata"))
    result["resource"] = _safe_url(result.get("resource"), "OAuth protected resource metadata resource")
    servers = _optional_strings(result.get("authorization_servers"), "authorization_servers")
    if servers is None:
        result.pop("authorization_servers", None)
    else:
        result["authorization_servers"] = [_safe_url(url, "authorization server URL") for url in servers]
    scopes = _optional_strings(result.get("scopes_supported"), "scopes_supported")
    if scopes is None:
        result.pop("scopes_supported", None)
    else:
        result["scopes_supported"] = scopes
    return result


def parse_authorization_server_metadata(value: Any) -> Dict[str, Any]:
    result = dict(_object(value, "authorization server metadata"))
    response_types = _optional_strings(result.get("response_types_supported"), "response_types_supported")
    if response_types is None:
        raise ValueError("Invalid response_types_supported")
    result["issuer"] = _safe_url(result.get("issuer"), "authorization server issuer")
    result["authorization_endpoint"] = _safe_url(result.get("authorization_endpoint"), "authorization endpoint")
    result["token_endpoint"] = _safe_url(result.get("token_endpoint"), "token endpoint")
    for key, name in (
        ("registration_endpoint", "registration endpoint"),
    ):
        url = _optional_url(result.get(key), name)
        if url is None:
            result.pop(key, None)
        else:
            result[key] = url
    for key in (
        "scopes_supported",
        "grant_types_supported",
        "token_endpoint_auth_methods_supported",
        "code_challenge_methods_supported",
    ):
        items = _optional_strings(result.get(key), key)
        if items is None:
            result.pop(key, None)
        else:
            result[key] = items
    result["response_types_supported"] = response_types
    for key in ("client_id_metadata_document_supported", "authorization_response_iss_parameter_supported"):
        flag = result.get(key)
        if not isinstance(flag, bool):
            result.pop(key, None)
    return result


def parse_oauth_tokens(value: Any) -> Dict[str, Any]:
    result = _object(value, "OAuth token response")
    # `Number(null)` is 0, which would mark the token as expired at once.
    expires = None if _absent(result.get("expires_in")) else _js_number(result.get("expires_in"))
    if expires is not None and not math.isfinite(expires):
        raise ValueError("Invalid expires_in")
    compact: Dict[str, Any] = {
        "access_token": _required_string(result.get("access_token"), "access_token"),
        "token_type": _required_string(result.get("token_type"), "token_type"),
    }
    if expires is not None:
        compact["expires_in"] = expires
    for key in ("scope", "refresh_token", "id_token"):
        text = _optional_string(result.get(key), key)
        if text is not None:
            compact[key] = text
    return compact


def parse_client_information(value: Any) -> Dict[str, Any]:
    result = dict(_object(value, "OAuth client registration response"))
    compact: Dict[str, Any] = {
        **result,
        "client_id": _required_string(result.get("client_id"), "client_id"),
    }
    secret = _optional_string(result.get("client_secret"), "client_secret")
    if secret is None:
        compact.pop("client_secret", None)
    else:
        compact["client_secret"] = secret
    for key in ("client_id_issued_at", "client_secret_expires_at"):
        if not isinstance(result.get(key), (int, float)) or isinstance(result.get(key), bool):
            compact.pop(key, None)
    uris = _optional_strings(result.get("redirect_uris"), "redirect_uris")
    compact["redirect_uris"] = uris if uris is not None else []
    return compact

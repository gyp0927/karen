"""AWS Signature Version 4 request signing, mirroring what the AWS SDK does for
pi-ai's Bedrock adapter (which relies on @aws-sdk's signer).

Only standard request signing is needed: ConverseStream is a plain JSON POST
whose *response* is an event stream, so there is no streaming-upload signing.
Credential resolution covers env vars and the shared credentials/config files;
SSO, IMDS, and container credential providers are intentionally not supported.
"""

from __future__ import annotations

import configparser
import hashlib
import hmac
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urlsplit

from .provider_env import get_provider_env_value
from ..types import ProviderEnv


@dataclass
class AwsCredentials:
    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = None


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _uri_encode(value: str, safe: str = "-_.~") -> str:
    return quote(value, safe=safe)


def _normalize_header_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def sign_request(
    *,
    method: str,
    url: str,
    headers: Dict[str, str],
    body: bytes,
    credentials: AwsCredentials,
    region: str,
    service: str,
    request_datetime: Optional[datetime] = None,
    include_content_sha256_header: bool = True,
) -> Dict[str, str]:
    """Sign a request and return the complete header set to send.

    `url` must already carry its final (encoded) path and query; `headers`
    are the caller-supplied headers (content-type, custom headers), which
    participate in the signature except `authorization`/`host`. The payload
    hash always closes the canonical request; `include_content_sha256_header`
    additionally signs it as an `x-amz-content-sha256` header (Bedrock default).
    """
    split = urlsplit(url)
    now = request_datetime or datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    payload_hash = _sha256_hex(body)

    # Collect the headers that participate in the signature.
    to_sign: Dict[str, str] = {}
    for name, value in headers.items():
        lower = name.lower()
        if lower in ("authorization", "host"):
            continue
        to_sign[lower] = _normalize_header_value(value)
    to_sign["host"] = split.netloc
    to_sign["x-amz-date"] = amz_date
    if include_content_sha256_header:
        to_sign["x-amz-content-sha256"] = payload_hash
    if credentials.session_token:
        to_sign["x-amz-security-token"] = credentials.session_token

    signed_names = sorted(to_sign)
    canonical_headers = "".join(f"{name}:{to_sign[name]}\n" for name in signed_names)
    signed_headers = ";".join(signed_names)

    # Canonical query string: URI-encode and sort by (name, value).
    query_pairs: List[Tuple[str, str]] = []
    if split.query:
        for pair in split.query.split("&"):
            name, _, value = pair.partition("=")
            query_pairs.append((_uri_encode(name), _uri_encode(value)))
    query_pairs.sort()
    canonical_query = "&".join(f"{name}={value}" for name, value in query_pairs)

    canonical_uri = split.path or "/"

    canonical_request = "\n".join(
        [
            method.upper(),
            canonical_uri,
            canonical_query,
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )

    scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope, _sha256_hex(canonical_request.encode("utf-8"))]
    )

    k_date = _hmac_sha256(f"AWS4{credentials.secret_access_key}".encode("utf-8"), date_stamp)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, service)
    k_signing = _hmac_sha256(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    authorization = (
        f"AWS4-HMAC-SHA256 Credential={credentials.access_key_id}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    # Final header set: caller headers (minus any stale authorization) + signed additions.
    final = {name: value for name, value in headers.items() if name.lower() != "authorization"}
    final["x-amz-date"] = amz_date
    if include_content_sha256_header:
        final["x-amz-content-sha256"] = payload_hash
    if credentials.session_token:
        final["x-amz-security-token"] = credentials.session_token
    final["Authorization"] = authorization
    return final


# -- Credential & region resolution ------------------------------------------


def _read_ini(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    return parser


def _profile_section(profile: str, config_file: bool) -> str:
    # The config file prefixes non-default profiles with "profile ".
    if config_file and profile != "default":
        return f"profile {profile}"
    return profile


def resolve_aws_credentials(
    env: Optional[ProviderEnv] = None,
    profile: Optional[str] = None,
) -> Optional[AwsCredentials]:
    """Env vars first, then the shared credentials/config files for `profile`
    (or AWS_PROFILE / AWS_DEFAULT_PROFILE / "default")."""
    access_key = get_provider_env_value("AWS_ACCESS_KEY_ID", env)
    secret_key = get_provider_env_value("AWS_SECRET_ACCESS_KEY", env)
    if access_key and secret_key:
        session_token = get_provider_env_value("AWS_SESSION_TOKEN", env)
        return AwsCredentials(access_key, secret_key, session_token)

    profile_name = (
        profile
        or get_provider_env_value("AWS_PROFILE", env)
        or get_provider_env_value("AWS_DEFAULT_PROFILE", env)
        or "default"
    )

    credentials_file = Path(
        get_provider_env_value("AWS_SHARED_CREDENTIALS_FILE", env) or Path.home() / ".aws" / "credentials"
    )
    config_file = Path(get_provider_env_value("AWS_CONFIG_FILE", env) or Path.home() / ".aws" / "config")

    for path, is_config in ((credentials_file, False), (config_file, True)):
        if not path.is_file():
            continue
        section = _profile_section(profile_name, is_config)
        parser = _read_ini(path)
        if not parser.has_section(section):
            continue
        access_key = parser.get(section, "aws_access_key_id", fallback=None)
        secret_key = parser.get(section, "aws_secret_access_key", fallback=None)
        if access_key and secret_key:
            session_token = parser.get(section, "aws_session_token", fallback=None)
            return AwsCredentials(access_key, secret_key, session_token)
    return None


def resolve_aws_profile_region(
    env: Optional[ProviderEnv] = None,
    profile: Optional[str] = None,
) -> Optional[str]:
    """Region from the shared credentials/config files for the active profile."""
    profile_name = (
        profile
        or get_provider_env_value("AWS_PROFILE", env)
        or get_provider_env_value("AWS_DEFAULT_PROFILE", env)
        or "default"
    )
    credentials_file = Path(
        get_provider_env_value("AWS_SHARED_CREDENTIALS_FILE", env) or Path.home() / ".aws" / "credentials"
    )
    config_file = Path(get_provider_env_value("AWS_CONFIG_FILE", env) or Path.home() / ".aws" / "config")
    for path, is_config in ((credentials_file, False), (config_file, True)):
        if not path.is_file():
            continue
        section = _profile_section(profile_name, is_config)
        parser = _read_ini(path)
        if parser.has_section(section):
            region = parser.get(section, "region", fallback=None)
            if region:
                return region
    return None

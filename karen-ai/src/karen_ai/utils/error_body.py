"""Shared normalization for provider HTTP error objects, mirroring utils/error-body.ts.

Endpoints behind a proxy / gateway may return a non-2xx response whose body
the exception object does not fold into `str(error)`. `normalize_provider_error`
probes the known error field shapes and returns a struct each provider composes
into its display string.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

MAX_PROVIDER_ERROR_BODY_CHARS = 4000


@dataclass
class NormalizedProviderError:
    """HTTP status and body extracted from a provider exception."""

    message: str
    message_carries_body: bool
    status: Optional[int] = None
    body: Optional[str] = None


def _extract_status(error: BaseException) -> Optional[int]:
    for attr in ("status_code", "status"):
        value = getattr(error, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    metadata = getattr(error, "metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("httpStatusCode") or metadata.get("http_status_code")
        if isinstance(value, int):
            return value
    return None


def _is_plain_non_empty_object(value: Any) -> bool:
    return isinstance(value, dict) and len(value) > 0


def safe_json_stringify(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _pick_body_text(error: BaseException) -> Optional[str]:
    body = getattr(error, "body", None)
    if isinstance(body, str):
        return body
    if _is_plain_non_empty_object(body):
        return safe_json_stringify(body)
    error_field = getattr(error, "error", None)
    if _is_plain_non_empty_object(error_field):
        return safe_json_stringify(error_field)
    return None


def truncate_error_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


def _extract_body(error: BaseException) -> Optional[str]:
    body_text = _pick_body_text(error)
    if body_text is None:
        return None
    trimmed = body_text.strip()
    if not trimmed:
        return None
    return truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS)


def normalize_provider_error(error: Any) -> NormalizedProviderError:
    if not isinstance(error, BaseException):
        return NormalizedProviderError(message=safe_json_stringify(error), message_carries_body=False)

    status = _extract_status(error)
    body = _extract_body(error)
    message = str(error) or type(error).__name__
    return NormalizedProviderError(
        status=status,
        body=body,
        message=message,
        message_carries_body=body is None or body in message,
    )


def format_provider_error(norm: NormalizedProviderError, prefix: Optional[str] = None) -> str:
    """Compose a display string from a normalized error.

    - no prefix: ``"<status>: <body>"``
    - prefix:    ``"<prefix> (<status>): <body>"``
    """
    if norm.message_carries_body or norm.status is None or norm.body is None:
        if prefix is not None and norm.status is not None:
            return f"{prefix} ({norm.status}): {norm.message}"
        return norm.message
    if prefix is not None:
        return f"{prefix} ({norm.status}): {norm.body}"
    return f"{norm.status}: {norm.body}"

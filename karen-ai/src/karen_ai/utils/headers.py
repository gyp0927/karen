"""Header helpers, mirroring utils/headers.ts."""

from __future__ import annotations

from typing import Dict, Optional

from ..types import ProviderHeaders


def headers_to_record(headers) -> Dict[str, str]:
    return {key: value for key, value in headers.items()}


def provider_headers_to_record(*header_sources: Optional[ProviderHeaders]) -> Optional[Dict[str, str]]:
    """Merge ProviderHeaders maps: later sources win, None values delete,
    name comparison is case-insensitive but the surviving key keeps its casing."""
    merged: Dict[str, tuple[str, str]] = {}
    for source in header_sources:
        for name, value in (source or {}).items():
            normalized = name.lower()
            merged.pop(normalized, None)
            if value is not None:
                merged[normalized] = (name, value)
    if not merged:
        return None
    return {name: value for name, value in merged.values()}

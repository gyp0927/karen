"""Option handling shared by the API adapters.

Every adapter reads its own extended fields straight off the options object
(`options.tool_choice`, `options.service_tier`, ...) instead of guarding each access,
mirroring pi-ai, where each adapter's `*Options` type extends `StreamOptions` with
optional properties and a missing one is simply `undefined`.

`Models.stream()` only promises a `StreamOptions` — or nothing at all — so each
adapter re-types what it is handed before reading those fields. Where pi-ai sees
`undefined`, the adapter here sees an unset field.
"""

from __future__ import annotations

from typing import Any, Optional, TypeVar

T = TypeVar("T")


def coerce_options(options: Optional[Any], options_type: Any) -> Any:
    """Return `options` as an instance of `options_type` (None stays None)."""
    if options is None or isinstance(options, options_type):
        return options
    return options_type(**options.model_dump(exclude_none=True))


__all__ = ["coerce_options"]

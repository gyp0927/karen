"""JSONL header detection and entry (de)serialization (pi's `jsonl/codec.ts`).

karen has no legacy v3 sessions to migrate, so v3 headers are *detected* (and
reported as such) but opening them raises `LegacyV3UnsupportedError` instead of
pi's transparent upgrade path.

Loaded message entries are coerced back into karen-ai message models via the
`Message` discriminated union on `role`; unknown/custom message shapes stay as
plain dicts (pi keeps them as plain objects — TypeScript types are erased).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple, Union

from karen_ai.types import Message
from pydantic import BaseModel, TypeAdapter, ValidationError

from ..types import CompactionEntry, Entry, MessageEntry

JSONL_FORMAT_VERSION = 4


class LegacyV3SessionHeader(BaseModel):
    """pi's pre-v4 session header; recognized but not supported."""

    type: str  # == "session"
    version: int  # == 3
    id: str
    timestamp: str
    cwd: str
    parent_session: Optional[str] = None


class LegacyV3UnsupportedError(ValueError):
    def __init__(self, path: str) -> None:
        super().__init__(
            f"Invalid JSONL storage {path}: legacy v3 sessions are not supported by karen-agent "
            "(no migration path is ported)"
        )


def _is_safe_integer_at_least(value: Any, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def is_legacy_v3_session_header(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("type") != "session" or value.get("version") != 3:
        return False
    if not isinstance(value.get("id"), str) or not isinstance(value.get("cwd"), str):
        return False
    timestamp = value.get("timestamp")
    if not isinstance(timestamp, str):
        return False
    from datetime import datetime

    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return False
    return "parentSession" not in value or isinstance(value["parentSession"], str)


def is_jsonl_storage_header(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("kind") == "header"
        and value.get("v") == JSONL_FORMAT_VERSION
        and isinstance(value.get("id"), str)
        and isinstance(value.get("cwd"), str)
        and _is_safe_integer_at_least(value.get("storageVersion"), 1)
        and _is_safe_integer_at_least(value.get("createdAt"), 0)
        and ("nextSeq" not in value or _is_safe_integer_at_least(value["nextSeq"], 1))
        and ("parentSessionId" not in value or isinstance(value["parentSessionId"], str))
        and ("legacyParentSessionPath" not in value or isinstance(value["legacyParentSessionPath"], str))
    )


def parse_jsonl_session_header(line: str) -> Tuple[str, Any]:
    """Parse one header line -> ("v4", JsonlStorageHeader) | ("v3-legacy", LegacyV3SessionHeader)."""
    from .types import JsonlStorageHeader

    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError("Invalid JSONL session header: not valid JSON") from error
    if is_jsonl_storage_header(value):
        return "v4", JsonlStorageHeader.model_validate(value)
    if is_legacy_v3_session_header(value):
        return "v3-legacy", LegacyV3SessionHeader.model_validate(value)
    raise ValueError("Unsupported JSONL session header")


# ---------------------------------------------------------------------------
# Entry (de)serialization
# ---------------------------------------------------------------------------

_MESSAGE_ADAPTER: TypeAdapter = TypeAdapter(Message)
_ENTRY_ADAPTER: TypeAdapter = TypeAdapter(Entry)


def to_jsonable(value: Any) -> Any:
    """Normalize pydantic models / containers to JSON-serializable data.

    Models dump in camelCase wire form with `None` fields omitted (matching
    pi's JSON.stringify, which drops absent/undefined keys).
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def coerce_agent_message(value: Any) -> Any:
    """Coerce a loaded message dict into the matching karen-ai model, if any."""
    if not isinstance(value, dict) or "role" not in value:
        return value
    try:
        return _MESSAGE_ADAPTER.validate_python(value)
    except ValidationError:
        return value


def entry_from_json(data: Dict[str, Any]) -> Entry:
    entry: Entry = _ENTRY_ADAPTER.validate_python(data)
    if isinstance(entry, MessageEntry):
        entry.message = coerce_agent_message(entry.message)
    elif isinstance(entry, CompactionEntry):
        entry.retained_tail = [coerce_agent_message(message) for message in entry.retained_tail]
    return entry


def entry_to_json(entry: Entry) -> Dict[str, Any]:
    data = entry.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(entry, MessageEntry):
        data["message"] = to_jsonable(entry.message)
    elif isinstance(entry, CompactionEntry):
        data["retainedTail"] = [to_jsonable(message) for message in entry.retained_tail]
    return data

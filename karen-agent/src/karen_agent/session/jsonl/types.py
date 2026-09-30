"""JSONL format constants and header/metadata types (pi's `harness/session/jsonl/types.ts`)."""

from __future__ import annotations

from typing import Literal, Optional

from karen_ai.types import KarenBase

from ..types import SessionCreateOptions, SessionMetadata

JSONL_FORMAT_VERSION = 4
JSONL_STORAGE_VERSION = 1


class JsonlStorageHeader(SessionMetadata):
    """First line of every JSONL session file (camelCase on disk)."""

    v: int = JSONL_FORMAT_VERSION
    kind: Literal["header"] = "header"
    cwd: str  # type: ignore[assignment]  # required on disk
    #: Sequence high-water mark written by snapshot rewrites (fork, migration).
    next_seq: Optional[int] = None


class JsonlSessionMetadata(SessionMetadata):
    cwd: str  # type: ignore[assignment]  # required for file-backed sessions
    path: str
    #: Filesystem modification time as milliseconds since Unix epoch.
    modified_at: int


class JsonlSessionCreateOptions(SessionCreateOptions):
    cwd: str


class JsonlSessionListOptions(KarenBase):
    cwd: Optional[str] = None

"""File-backed format-4 session repository (pi's `jsonl/repo.ts`).

Layout mirrors pi: `<sessionsRoot>/<--cwd-encoded-->/<isotime>_<id>.jsonl`.
The legacy v3 fork/migration paths are not ported — v3 files are skipped by
`list()` and rejected on open (`LegacyV3UnsupportedError`).
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import quote

from ..ids import uuid7
from ..session import StorageBackedSession
from ..types import ForkOptions, Session, SessionRepo
from .codec import parse_jsonl_session_header
from .fork import ClosedForkInput, JsonlForkInput, OpenForkInput, run_jsonl_fork
from .storage import JsonlStorage
from .types import (
    JSONL_FORMAT_VERSION,
    JSONL_STORAGE_VERSION,
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionMetadata,
    JsonlStorageHeader,
)

#: Characters JavaScript's encodeURIComponent leaves alone beyond Python's always-safe set.
_URI_SAFE = "!~*'()"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _metadata_from_header(header: JsonlStorageHeader, path: str, modified_at: int) -> JsonlSessionMetadata:
    return JsonlSessionMetadata(
        id=header.id,
        created_at=header.created_at,
        storage_version=header.storage_version,
        cwd=header.cwd,
        path=path,
        modified_at=modified_at,
        parent_session_id=header.parent_session_id,
        legacy_parent_session_path=header.legacy_parent_session_path,
    )


def session_directory_name(cwd: str) -> str:
    """Lossy by design: /a/b and /a-b both map to `--a-b--` (pi parity)."""
    stripped = re.sub(r"^[/\\]", "", cwd)
    return "--" + re.sub(r"[/\\:]", "-", stripped) + "--"


def session_file_name(created_at: int, id: str) -> str:
    iso = (
        datetime.fromtimestamp(created_at / 1000, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    timestamp = iso.replace(":", "-").replace(".", "-")
    return f"{timestamp}_{quote(id, safe=_URI_SAFE)}.jsonl"


def _mtime_ms(path: str) -> int:
    return os.stat(path).st_mtime_ns // 1_000_000


class JsonlSessionRepo(SessionRepo):
    def __init__(self, sessions_root: str, now: Optional[Callable[[], int]] = None) -> None:
        self._sessions_root_input = sessions_root
        self._now = now or _now_ms
        self._open_sessions: Dict[str, JsonlStorage] = {}
        self._pending_creates: set = set()
        self._closed = False
        self._close_task: Optional[asyncio.Future] = None

    # --- lifecycle ----------------------------------------------------------

    async def create(self, options: JsonlSessionCreateOptions) -> Session:
        self._assert_open()
        created_at = self._now()
        cwd = self._absolute_path(options.cwd, f"Failed to resolve session cwd {options.cwd}")
        id = options.id or uuid7(created_at)
        key = self._session_key(cwd, id)
        if key in self._open_sessions or key in self._pending_creates:
            raise ValueError(f"Session already exists: {id}")
        self._pending_creates.add(key)
        path: Optional[str] = None
        storage: Optional[JsonlStorage] = None
        try:
            path = self._resolve_new_session_path(cwd, created_at, id)
            header = JsonlStorageHeader(
                v=JSONL_FORMAT_VERSION,
                kind="header",
                id=id,
                storage_version=JSONL_STORAGE_VERSION,
                created_at=created_at,
                cwd=cwd,
                parent_session_id=options.parent_session_id,
            )
            storage = await JsonlStorage.create(path, header, [], now=self._now)
            metadata = _metadata_from_header(header, path, _mtime_ms(path))
            return self._publish_open_session(metadata, storage, key)
        except Exception:
            if storage is not None:
                try:
                    await storage.close()
                except Exception:
                    pass
            if path is not None:
                try:
                    os.remove(path)
                except OSError:
                    pass
            raise
        finally:
            self._pending_creates.discard(key)

    async def open(self, metadata: JsonlSessionMetadata) -> Session:
        self._assert_open()
        key = self._session_key(metadata.cwd, metadata.id)
        if key in self._open_sessions:
            raise ValueError(f"Session is already open: {metadata.id}")
        storage: Optional[JsonlStorage] = None
        try:
            storage = await self._load_storage(metadata)
            return self._publish_open_session(metadata, storage, key)
        except Exception:
            if storage is not None:
                try:
                    await storage.close()
                except Exception:
                    pass
            raise

    async def list(self, options: Optional[JsonlSessionListOptions] = None) -> List[JsonlSessionMetadata]:
        self._assert_open()
        cwd = None
        if options is not None and options.cwd is not None:
            cwd = self._absolute_path(options.cwd, f"Failed to resolve session cwd {options.cwd}")
        root = self._root()
        if not os.path.exists(root):
            return []
        if cwd is None:
            directories = [
                os.path.join(root, name)
                for name in os.listdir(root)
                if os.path.isdir(os.path.join(root, name))
            ]
        else:
            directories = [os.path.join(root, session_directory_name(cwd))]
        metadata: List[JsonlSessionMetadata] = []
        for directory in directories:
            metadata.extend(self._list_directory(directory, cwd))
        metadata.sort(key=lambda m: (-m.created_at, m.id, m.cwd))
        return metadata

    async def delete(self, metadata: JsonlSessionMetadata) -> None:
        self._assert_open()
        key = self._session_key(metadata.cwd, metadata.id)
        if key in self._open_sessions:
            raise ValueError(f"Session is open: {metadata.id}")
        if not os.path.exists(metadata.path):
            raise ValueError(f"Session file does not exist: {metadata.path}")
        try:
            os.remove(metadata.path)
        except OSError as error:
            raise ValueError(f"Failed to delete session {metadata.path}: {error}") from error

    async def fork(self, source: JsonlSessionMetadata, options: ForkOptions) -> Session:
        self._assert_open()
        created_at = self._now()
        cwd = source.cwd
        id = options.id or uuid7(created_at)
        destination_key = self._session_key(cwd, id)
        if destination_key in self._open_sessions or destination_key in self._pending_creates:
            raise ValueError(f"Session already exists: {id}")
        self._pending_creates.add(destination_key)

        source_storage = self._open_sessions.get(self._session_key(source.cwd, source.id))
        path: Optional[str] = None
        storage: Optional[JsonlStorage] = None
        try:
            if source_storage is not None:
                next_seq = await source_storage.capture_fork_next_seq()
                input: JsonlForkInput = OpenForkInput(metadata=source, next_seq=next_seq)
            else:
                input = ClosedForkInput(metadata=source)
            path = self._resolve_new_session_path(cwd, created_at, id)
            header = JsonlStorageHeader(
                v=JSONL_FORMAT_VERSION,
                kind="header",
                id=id,
                storage_version=JSONL_STORAGE_VERSION,
                created_at=created_at,
                cwd=cwd,
                parent_session_id=source.id,
            )
            await run_jsonl_fork(
                input=input,
                destination_path=path,
                destination_header=header,
                fork=options,
            )
            storage = await JsonlStorage.open(path, now=self._now)
            metadata = _metadata_from_header(header, path, _mtime_ms(path))
            return self._publish_open_session(metadata, storage, destination_key)
        except Exception:
            if storage is not None:
                try:
                    await storage.close()
                except Exception:
                    pass
            if path is not None:
                try:
                    os.remove(path)
                except OSError:
                    pass
            raise
        finally:
            self._pending_creates.discard(destination_key)

    def close(self) -> "asyncio.Future[None]":
        # TODO(pi parity): ownership semantics are undefined upstream; repository
        # close does not close session handles.
        if self._close_task is not None:
            return self._close_task
        self._closed = True

        async def finish() -> None:
            return None

        self._close_task = asyncio.ensure_future(finish())
        return self._close_task

    # --- internals -------------------------------------------------------------

    def _list_directory(self, directory: str, cwd: Optional[str]) -> List[JsonlSessionMetadata]:
        if not os.path.isdir(directory):
            return []
        metadata: List[JsonlSessionMetadata] = []
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if not os.path.isfile(path) or not name.endswith(".jsonl"):
                continue
            discovered = self._read_session_metadata(path)
            if discovered is None:
                continue
            # Directory encoding is lossy: /a/b and /a-b both map to --a-b--.
            if cwd is None or discovered.cwd == cwd:
                metadata.append(discovered)
        return metadata

    def _read_session_metadata(self, path: str) -> Optional[JsonlSessionMetadata]:
        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                first_line = handle.readline()
        except OSError as error:
            raise ValueError(f"Failed to read session header {path}: {error}") from error
        if not first_line:
            return None
        try:
            format, header = parse_jsonl_session_header(first_line.rstrip("\n"))
        except ValueError:
            return None
        if format == "v3-legacy":
            return None  # legacy migration is not ported
        return _metadata_from_header(header, path, _mtime_ms(path))

    def _resolve_new_session_path(self, cwd: str, created_at: int, id: str) -> str:
        directory = os.path.join(self._root(), session_directory_name(cwd))
        self._assert_session_id_available(directory, id)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, session_file_name(created_at, id))

    def _assert_session_id_available(self, directory: str, id: str) -> None:
        if not os.path.isdir(directory):
            return
        suffix = f"_{quote(id, safe=_URI_SAFE)}.jsonl"
        id_exists = any(
            os.path.isfile(os.path.join(directory, name)) and name.endswith(suffix)
            for name in os.listdir(directory)
        )
        if id_exists:
            raise ValueError(f"Session already exists: {id}")

    async def _load_storage(self, metadata: JsonlSessionMetadata) -> JsonlStorage:
        if not os.path.exists(metadata.path):
            raise ValueError(f"Session file does not exist: {metadata.path}")
        storage = await JsonlStorage.open(metadata.path, now=self._now)
        if storage.header.id != metadata.id or storage.header.cwd != metadata.cwd:
            raise ValueError(f"Session identity does not match header: {metadata.id}")
        if storage.header.storage_version != JSONL_STORAGE_VERSION:
            raise ValueError(
                f"Session {metadata.id} uses unsupported storage version {storage.header.storage_version}"
            )
        return storage

    def _publish_open_session(
        self,
        metadata: JsonlSessionMetadata,
        storage: JsonlStorage,
        key: str,
    ) -> StorageBackedSession:
        if key in self._open_sessions:
            raise ValueError(f"Session is already open: {metadata.id}")

        def on_close() -> None:
            if self._open_sessions.get(key) is storage:
                del self._open_sessions[key]

        session = StorageBackedSession(metadata, storage, on_close=on_close)
        self._open_sessions[key] = storage
        return session

    def _session_key(self, cwd: str, id: str) -> str:
        return f"{cwd}\x00{id}"

    def _root(self) -> str:
        return self._absolute_path(self._sessions_root_input, f"Failed to resolve sessions root {self._sessions_root_input}")

    @staticmethod
    def _absolute_path(path: str, action: str) -> str:
        try:
            return os.path.abspath(path)
        except (OSError, ValueError) as error:
            raise ValueError(f"{action}: {error}") from error

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("JsonlSessionRepo is closed")

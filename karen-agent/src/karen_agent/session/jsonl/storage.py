"""JSONL file-backed storage (pi's `jsonl/storage.ts`).

One file per session: header line, then one transaction line per commit. The
whole file is replayed into an `InMemoryStorageState` on open; commits append a
single line. A torn (unterminated) final line is discarded and the file is
atomically rewritten without it before new writes are admitted.

The legacy v3 transparent-upgrade path is not ported (karen has no v3 files);
opening a v3 header raises `LegacyV3UnsupportedError`.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..mutation_line import MutationLine
from ..storage_state import InMemoryStorageState
from ..types import (
    CommitResult,
    Entry,
    EntryScan,
    EntryStructure,
    SessionStats,
    Storage,
    StorageBranchScan,
    UsageRow,
    UsageScan,
    Write,
)
from ..values import StoredValue, Value, ValueList
from .codec import LegacyV3UnsupportedError
from .io import (
    parse_jsonl_transaction,
    publish_file_atomically,
    publish_jsonl,
    read_jsonl_header_line,
    serialize_jsonl_transaction,
    split_complete_lines,
)
from .types import JSONL_STORAGE_VERSION, JsonlStorageHeader


def _now_ms() -> int:
    return int(time.time() * 1000)


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Failed to read JSONL storage {path}: {error}") from error


class JsonlStorage(Storage):
    def __init__(self, path: str, header: JsonlStorageHeader, now: Optional[Callable[[], int]] = None) -> None:
        self._path = path
        self.header = header
        self._now = now or _now_ms
        self._storage_state = InMemoryStorageState()
        self._commit_line = MutationLine()
        self._state = "open"  # "open" | "closing" | "closed"
        self._close_task: Optional[asyncio.Future] = None

    @classmethod
    async def create(
        cls,
        path: str,
        header: JsonlStorageHeader,
        initial_writes: List[Write],
        now: Optional[Callable[[], int]] = None,
    ) -> "JsonlStorage":
        storage = cls(path, header, now)
        prepared = storage._storage_state.prepare_commit(initial_writes, storage._now())

        async def transactions(append) -> None:
            if prepared.writes:
                await append(prepared.writes)

        await publish_jsonl(path, header, transactions)
        storage._storage_state.apply_validated(prepared.writes)
        return storage

    @classmethod
    async def open(cls, path: str, now: Optional[Callable[[], int]] = None) -> "JsonlStorage":
        content = _read_text(path)
        lines, torn = split_complete_lines(content)
        if not lines or lines[0] == "":
            raise ValueError(f"Invalid JSONL storage {path}: missing header")
        format, header = read_jsonl_header_line((lines[0], True), path)
        if format == "v3-legacy":
            raise LegacyV3UnsupportedError(path)
        if header.storage_version != JSONL_STORAGE_VERSION:
            raise ValueError(f"Session {header.id} uses unsupported storage version {header.storage_version}")
        storage = cls(path, header, now)
        for index in range(1, len(lines)):
            try:
                storage._replay_committed(parse_jsonl_transaction(lines[index]))
            except Exception as error:
                raise ValueError(f"Invalid JSONL storage {path}: line {index + 1}") from error
        if header.next_seq is not None:
            storage._storage_state.advance_next_seq(header.next_seq)
        if torn:
            # Discard the torn tail before admitting writes.

            async def content_body(append) -> None:
                await append("\n".join(lines) + "\n")

            await publish_file_atomically(path, content_body)
        return storage

    def _replay_committed(self, writes: List[Write]) -> None:
        self._storage_state.validate_committed(writes)
        self._storage_state.apply_validated(writes)

    async def commit(self, writes: List[Write]) -> CommitResult:
        if self._state != "open":
            raise RuntimeError("JsonlStorage is closed")
        return await self._commit_line.run(lambda: self._apply_commit(writes))

    def _apply_commit(self, writes: List[Write]) -> CommitResult:
        prepared = self._storage_state.prepare_commit(writes, self._now())
        if prepared.writes:
            try:
                with open(self._path, "a", encoding="utf-8", newline="") as handle:
                    handle.write(serialize_jsonl_transaction(prepared.writes) + "\n")
            except OSError as error:
                raise ValueError(f"Failed to append JSONL storage {self._path}: {error}") from error
        stats = self._storage_state.apply_validated(prepared.writes)
        return CommitResult(
            first_seq=prepared.first_seq,
            seqs=prepared.seqs,
            timestamp=prepared.timestamp,
            stats=stats,
        )

    async def get_entries(self, ids: List[str]) -> Dict[str, Entry]:
        self._assert_open()
        return self._storage_state.get_entries(ids)

    async def get_value(self, address: Value) -> Optional[StoredValue]:
        self._assert_open()
        return self._storage_state.get_value(address)

    async def scan_values(self, prefix: Value) -> List[StoredValue]:
        self._assert_open()
        return self._storage_state.scan_values(prefix)

    async def read_list(self, address: ValueList, options=None) -> List[Any]:
        self._assert_open()
        return self._storage_state.read_list(address, options)

    async def scan_branch(self, query: StorageBranchScan) -> List[Entry]:
        self._assert_open()
        return self._storage_state.scan_branch(query)

    async def scan_branch_structure(self, query: StorageBranchScan) -> List[EntryStructure]:
        self._assert_open()
        return self._storage_state.scan_branch_structure(query)

    async def scan_entries(self, query: EntryScan) -> List[Entry]:
        self._assert_open()
        return self._storage_state.scan_entries(query)

    async def scan_usage(self, query: UsageScan) -> List[UsageRow]:
        self._assert_open()
        return self._storage_state.scan_usage(query)

    async def get_stats(self) -> SessionStats:
        self._assert_open()
        return self._storage_state.get_stats()

    async def capture_fork_next_seq(self) -> int:
        """Capture the first sequence a later source commit would use."""
        if self._state != "open":
            raise RuntimeError("JsonlStorage is closed")
        return await self._commit_line.run(lambda: self._storage_state.get_next_seq())

    def close(self) -> "asyncio.Future[None]":
        if self._close_task is not None:
            return self._close_task
        self._state = "closing"

        async def finish() -> None:
            await self._commit_line.seal(RuntimeError("JsonlStorage is closed"))
            self._state = "closed"

        self._close_task = asyncio.ensure_future(finish())
        return self._close_task

    def _assert_open(self) -> None:
        if self._state != "open":
            raise RuntimeError("JsonlStorage is closed")

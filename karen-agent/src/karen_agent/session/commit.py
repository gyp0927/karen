"""Commit stamping and validation (pi's `harness/session/commit.ts`).

Storage assigns `seq` (monotonic per session) and one `timestamp` per commit.
`commit_write` returns a stamped copy; `validate_committed_writes` enforces
monotonic sequences, unique entry/usage ids, and existing parent entries.
"""

from __future__ import annotations

from typing import List, Protocol

from karen_ai.types import KarenBase

from .types import (
    Entry,
    EntryWrite,
    UsageRow,
    UsageWrite,
    Write,
)


class PreparedCommit(KarenBase):
    writes: List[Write]
    first_seq: int
    seqs: List[int]
    timestamp: int


class CommitValidationState(Protocol):
    def has_entry_or_usage_id(self, id: str) -> bool: ...

    def has_entry_id(self, id: str) -> bool: ...


def insert_entry(entry: Entry) -> EntryWrite:
    return EntryWrite(entry=entry)


def insert_usage(row: UsageRow) -> UsageWrite:
    return UsageWrite(row=row)


def commit_write(write: Write, seq: int, timestamp: int) -> Write:
    """Return a copy of `write` stamped with its committed sequence (and entry timestamp)."""
    if isinstance(write, EntryWrite):
        return EntryWrite(entry=write.entry.model_copy(update={"seq": seq, "timestamp": timestamp}))
    if isinstance(write, UsageWrite):
        return UsageWrite(row=write.row.model_copy(update={"seq": seq}))
    return write.model_copy(update={"seq": seq})


def materialize_committed_entry(entry: Entry, seq: int, timestamp: int) -> Entry:
    return entry.model_copy(update={"seq": seq, "timestamp": timestamp})


def prepare_storage_commit(writes: List[Write], first_seq: int, timestamp: int) -> PreparedCommit:
    committed = [commit_write(write, first_seq + index, timestamp) for index, write in enumerate(writes)]
    return PreparedCommit(
        writes=committed,
        first_seq=first_seq,
        seqs=[write_seq(write) for write in committed],
        timestamp=timestamp,
    )


def write_seq(write: Write) -> int:
    """The committed sequence of a stamped write (entry/usage carry it nested)."""
    if isinstance(write, EntryWrite):
        return write.entry.seq
    if isinstance(write, UsageWrite):
        return write.row.seq
    return write.seq


def write_id(write: Write) -> str:
    if isinstance(write, EntryWrite):
        return write.entry.id
    if isinstance(write, UsageWrite):
        return write.row.id
    raise TypeError(f"Write kind has no id: {write.kind}")


def validate_committed_writes(
    writes: List[Write],
    first_seq: int,
    state: CommitValidationState,
) -> None:
    previous_seq = first_seq - 1
    transaction_ids: set[str] = set()
    transaction_entry_ids: set[str] = set()
    for write in writes:
        seq = write_seq(write)
        if seq <= previous_seq:
            raise ValueError(f"Non-monotonic storage sequence: {seq}")
        previous_seq = seq
        if not isinstance(write, (EntryWrite, UsageWrite)):
            continue
        id = write_id(write)
        if state.has_entry_or_usage_id(id) or id in transaction_ids:
            raise ValueError(f"Duplicate entry or usage id: {id}")
        if isinstance(write, EntryWrite):
            parent_id = write.entry.parent_id
            if parent_id is not None and not state.has_entry_id(parent_id) and parent_id not in transaction_entry_ids:
                raise ValueError(f"Missing parent entry: {parent_id}")
        transaction_ids.add(id)
        if isinstance(write, EntryWrite):
            transaction_entry_ids.add(id)

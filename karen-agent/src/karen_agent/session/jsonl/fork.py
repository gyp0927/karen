"""Streaming two-pass JSONL fork (pi's `jsonl/fork.ts`).

Pass one folds complete source transactions into a lightweight index (entry
parent links, current-row sequences, lane inventory — not payloads) and captures
the sequence boundary. Pass two re-reads the source and streams only selected
entries plus current scalar/list state into an atomically published format-4
destination, preserving source sequences and `nextSeq`. Usage rows and
operation/pending/result state are excluded. The source file must not be
replaced or edited between passes; later append-only writes are excluded by the
captured boundary. The legacy-v3 fork input kind is not ported.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple, Union

from karen_ai.types import KarenBase
from pydantic import ConfigDict

from ..commit import write_seq
from ..fork_policy import (
    UNDEFINED,
    ForkCurrentStatePlan,
    TreeForkPlan,
    project_fork_current_state_write,
    select_branch_fork,
)
from ..types import (
    BranchForkOptions,
    EntryWrite,
    ForkOptions,
    ListAppendWrite,
    ListDeleteWrite,
    UsageWrite,
    ValueDeleteWrite,
    ValueSetWrite,
    Write,
)
from .codec import parse_jsonl_session_header
from .io import iter_file_lines, parse_jsonl_transaction, publish_jsonl, read_jsonl_header_line
from .types import JSONL_STORAGE_VERSION, JsonlSessionMetadata, JsonlStorageHeader


class OpenForkInput(KarenBase):
    """Fork source currently open in this repo; copies stop at the captured boundary."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: str = "open"
    metadata: JsonlSessionMetadata
    next_seq: int


class ClosedForkInput(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: str = "closed"
    metadata: JsonlSessionMetadata


JsonlForkInput = Union[OpenForkInput, ClosedForkInput]


def _physical_key(namespace: str, key: str) -> str:
    return f"{namespace}\x00{key}"


def _read_fork_header(first: Tuple[str, bool], source: JsonlSessionMetadata) -> JsonlStorageHeader:
    format, header = read_jsonl_header_line(first, source.path)
    if format != "v4":
        raise ValueError(f"Invalid JSONL storage {source.path}: expected format 4 header")
    if header.id != source.id or header.cwd != source.cwd:
        raise ValueError(f"Session identity does not match header: {source.id}")
    if header.storage_version != JSONL_STORAGE_VERSION:
        raise ValueError(f"Session {source.id} uses unsupported storage version {header.storage_version}")
    return header


def _reaches_fork_boundary(writes: List[Write], stop_before_seq: Optional[int]) -> bool:
    """Whether this complete transaction reaches the fork boundary (never split one)."""
    if stop_before_seq is None or not writes:
        return False
    if write_seq(writes[0]) >= stop_before_seq:
        return True
    if write_seq(writes[-1]) >= stop_before_seq:
        raise ValueError(f"JSONL transaction crosses fork sequence boundary {stop_before_seq}")
    return False


def _read_fork_transactions(
    lines: Iterator[Tuple[str, bool]],
    path: str,
    stop_before_seq: Optional[int],
) -> Iterator[List[Write]]:
    """Read complete transactions after the header, never splitting at the boundary."""
    for text, terminated in lines:
        if not terminated:
            break
        writes = parse_jsonl_transaction(text)
        if _reaches_fork_boundary(writes, stop_before_seq):
            break
        yield writes


class JsonlForkIndex:
    def __init__(self) -> None:
        self._current_scalar_seqs: Dict[str, int] = {}
        self._branch_tips: Dict[str, Optional[str]] = {}
        self._first_surviving_list_seqs: Dict[str, int] = {}
        self._entry_parents: Dict[str, Optional[str]] = {}
        self._copied_entry_ids: set = set()
        self._lane_configs: set = set()
        self._lane_states: set = set()

    def apply_entry(self, id: str, parent_id: Optional[str]) -> None:
        self._entry_parents[id] = parent_id

    def apply_writes(self, writes: List[Write]) -> None:
        for write in writes:
            if isinstance(write, EntryWrite):
                self.apply_entry(write.entry.id, write.entry.parent_id)
            elif isinstance(write, (ValueSetWrite, ValueDeleteWrite)):
                key = _physical_key(write.namespace, write.key)
                if isinstance(write, ValueDeleteWrite):
                    self._current_scalar_seqs.pop(key, None)
                else:
                    self._current_scalar_seqs[key] = write.seq
                self._apply_lane_value(write)
            elif isinstance(write, (ListAppendWrite, ListDeleteWrite)):
                key = _physical_key(write.namespace, write.key)
                if isinstance(write, ListDeleteWrite):
                    self._first_surviving_list_seqs.pop(key, None)
                elif key not in self._first_surviving_list_seqs:
                    self._first_surviving_list_seqs[key] = write.seq
            # usage rows carry no fork-relevant state

    def _apply_lane_value(self, write) -> None:
        present = isinstance(write, ValueSetWrite)
        if write.namespace == "pi.branch.tip":
            if present:
                self._branch_tips[write.key] = write.value
            else:
                self._branch_tips.pop(write.key, None)
        elif write.namespace == "pi.lane.config":
            if present:
                self._lane_configs.add(write.key)
            else:
                self._lane_configs.discard(write.key)
        elif write.namespace == "pi.lane.state":
            if present:
                self._lane_states.add(write.key)
            else:
                self._lane_states.discard(write.key)

    def get_branch_tip(self, branch: str):
        """Branch tip value, `None` for an empty branch, UNDEFINED when unknown."""
        return self._branch_tips.get(branch, UNDEFINED)

    def has_complete_lane(self, branch: str) -> bool:
        return branch in self._lane_configs and branch in self._lane_states

    def get_current_scalar_seq(self, namespace: str, key: str) -> Optional[int]:
        return self._current_scalar_seqs.get(_physical_key(namespace, key))

    def is_surviving_list_element(self, namespace: str, key: str, seq: int) -> bool:
        first_seq = self._first_surviving_list_seqs.get(_physical_key(namespace, key))
        return first_seq is not None and seq >= first_seq

    def get_parent(self, entry_id: str):
        return self._entry_parents.get(entry_id, UNDEFINED)

    def select_entry(self, entry_id: str) -> None:
        self._copied_entry_ids.add(entry_id)

    def is_entry_selected(self, entry_id: str) -> bool:
        return entry_id in self._copied_entry_ids


def _select_jsonl_fork(index: JsonlForkIndex, options: ForkOptions) -> ForkCurrentStatePlan:
    """Validate the source lanes and return the fork plan (no file I/O)."""
    if not isinstance(options, BranchForkOptions):
        return TreeForkPlan()
    plan = select_branch_fork(
        options,
        tip=index.get_branch_tip(options.branch),
        get_parent=index.get_parent,
        select_entry=index.select_entry,
    )
    if not index.has_complete_lane(options.branch):
        raise ValueError(f"Source branch {options.branch!r} is not a configured AgentLane")
    return plan


def _index_fork_input(input: JsonlForkInput) -> Tuple[JsonlForkIndex, int]:
    """Build the fork index and the source's nextSeq high-water mark."""
    index = JsonlForkIndex()
    path = input.metadata.path
    try:
        lines = iter_file_lines(path)
        try:
            first = next(lines, None)
            if first is None:
                raise ValueError(f"Invalid JSONL storage {path}: missing header")
            header = _read_fork_header(first, input.metadata)
            stop_before_seq = input.next_seq if isinstance(input, OpenForkInput) else None
            highest_complete_seq = 0
            for writes in _read_fork_transactions(lines, path, stop_before_seq):
                index.apply_writes(writes)
                if writes:
                    highest_complete_seq = write_seq(writes[-1])
        finally:
            lines.close()
    except OSError as error:
        raise ValueError(f"Failed to open JSONL fork source {path}: {error}") from error
    if isinstance(input, OpenForkInput):
        return index, input.next_seq
    return index, max(header.next_seq or 1, highest_complete_seq + 1)


def _stream_fork_writes(input: JsonlForkInput, stop_before_seq: int) -> Iterator[Write]:
    """Yield source writes up to the captured boundary; callers own projection."""
    path = input.metadata.path
    try:
        lines = iter_file_lines(path)
        try:
            first = next(lines, None)
            if first is None:
                raise ValueError(f"Invalid JSONL storage {path}: missing header")
            _read_fork_header(first, input.metadata)
            for writes in _read_fork_transactions(lines, path, stop_before_seq):
                yield from writes
        finally:
            lines.close()
    except OSError as error:
        raise ValueError(f"Failed to open JSONL fork source {path}: {error}") from error


def _project_jsonl_fork_write(
    write: Write,
    index: JsonlForkIndex,
    plan: ForkCurrentStatePlan,
    is_entry_copied,
):
    if isinstance(write, EntryWrite):
        return write if is_entry_copied(write.entry.id) else None
    if isinstance(write, (ValueSetWrite, ValueDeleteWrite)):
        if isinstance(write, ValueDeleteWrite):
            return None
        if index.get_current_scalar_seq(write.namespace, write.key) != write.seq:
            return None
        return project_fork_current_state_write(write, plan, is_entry_copied)
    if isinstance(write, (ListAppendWrite, ListDeleteWrite)):
        if isinstance(write, ListDeleteWrite):
            return None
        if not index.is_surviving_list_element(write.namespace, write.key, write.seq):
            return None
        return project_fork_current_state_write(write, plan, is_entry_copied)
    if isinstance(write, UsageWrite):
        return None
    return None


async def run_jsonl_fork(
    *,
    input: JsonlForkInput,
    destination_path: str,
    destination_header: JsonlStorageHeader,
    fork: ForkOptions,
) -> None:
    """Index the source, validate the fork, and stream selected state into the destination."""
    index, next_seq = _index_fork_input(input)
    plan = _select_jsonl_fork(index, fork)
    if isinstance(plan, TreeForkPlan):
        is_entry_copied = lambda entry_id: True
    else:
        is_entry_copied = index.is_entry_selected
    header = destination_header.model_copy(update={"next_seq": next_seq})

    async def transactions(append) -> None:
        for write in _stream_fork_writes(input, next_seq):
            projected = _project_jsonl_fork_write(write, index, plan, is_entry_copied)
            if projected is not None:
                await append([projected])

    await publish_jsonl(destination_path, header, transactions)

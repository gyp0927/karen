"""Complete materialized session state for MemoryStorage and JsonlStorage
(pi's `harness/session/in-memory-storage-state.ts`).

This is intentionally unsuitable for database backends and long-running sessions
that may not fit in memory. Those backends should query indexed durable state and
update durable aggregates within each commit transaction.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

from ..utils import add_usage
from .commit import (
    PreparedCommit,
    prepare_storage_commit,
    validate_committed_writes,
    write_seq,
)
from .fork_policy import (
    UNDEFINED,
    BranchForkPlan,
    ForkCurrentStatePlan,
    TreeForkPlan,
    project_fork_current_state_write,
    select_branch_fork,
)
from .types import (
    Entry,
    EntryScan,
    EntryStructure,
    EntryWrite,
    ForkOptions,
    ListAppendWrite,
    ListDeleteWrite,
    SessionStats,
    StorageBranchScan,
    TreeForkOptions,
    UsageRow,
    UsageScan,
    UsageWrite,
    ValueDeleteWrite,
    ValueSetWrite,
    Write,
)
from .values import (
    ListElement,
    ListReadOptions,
    StoredValue,
    Value,
    ValueList,
    branch_tip,
    lane_config,
    lane_state,
    resolve_list_read_options,
    value,
    value_list,
)


def physical_key(namespace: str, key: str) -> str:
    return f"{namespace}\x00{key}"


class _StoredListSnapshot:
    def __init__(self, address: ValueList, elements: List[ListElement]) -> None:
        self.address = address
        self.elements = elements


class InMemoryStorageState:
    def __init__(self) -> None:
        self._entries: Dict[str, Entry] = {}
        self._entries_by_seq: List[Entry] = []
        self._scalar_values: Dict[str, StoredValue] = {}
        self._list_values: Dict[str, _StoredListSnapshot] = {}
        self._usage: Dict[str, UsageRow] = {}
        self._stats = SessionStats()
        self._next_seq = 1

    # --- commit pipeline ----------------------------------------------------

    def prepare_commit(self, writes: List[Write], timestamp: int) -> PreparedCommit:
        prepared = prepare_storage_commit(writes, self._next_seq, timestamp)
        self.validate_committed(prepared.writes)
        return prepared

    def validate_committed(self, writes: List[Write]) -> None:
        state = self
        validate_committed_writes(writes, self._next_seq, state)  # type: ignore[arg-type]

    def has_entry_or_usage_id(self, id: str) -> bool:
        return id in self._entries or id in self._usage

    def has_entry_id(self, id: str) -> bool:
        return id in self._entries

    def apply_validated(self, writes: List[Write]) -> SessionStats:
        """Apply writes already accepted by validate_committed(); return post-apply totals."""
        for write in writes:
            if isinstance(write, EntryWrite):
                entry = write.entry
                self._entries[entry.id] = entry
                self._entries_by_seq.append(entry)
                if entry.type == "message":
                    self._stats.message_count += 1
            elif isinstance(write, UsageWrite):
                row = write.row
                self._usage[row.id] = row
                self._stats.usage = add_usage(self._stats.usage, row.usage)
            elif isinstance(write, ValueDeleteWrite):
                self._scalar_values.pop(physical_key(write.namespace, write.key), None)
            elif isinstance(write, ValueSetWrite):
                self._apply_value_set(write)
            elif isinstance(write, ListDeleteWrite):
                self._list_values.pop(physical_key(write.namespace, write.key), None)
            elif isinstance(write, ListAppendWrite):
                self._apply_list_append(write)
            self._next_seq = write_seq(write) + 1
        return self._stats

    def _apply_value_set(self, write: ValueSetWrite) -> None:
        self._scalar_values[physical_key(write.namespace, write.key)] = StoredValue(
            address=Value(namespace=write.namespace, key=write.key),
            value=write.value,
            seq=write.seq,
        )

    def _apply_list_append(self, write: ListAppendWrite) -> None:
        element = ListElement(seq=write.seq, value=write.value)
        key = physical_key(write.namespace, write.key)
        stored = self._list_values.get(key)
        if stored is None:
            self._list_values[key] = _StoredListSnapshot(
                address=ValueList(namespace=write.namespace, key=write.key),
                elements=[element],
            )
        else:
            stored.elements.append(element)

    # --- fork ---------------------------------------------------------------

    def create_fork(self, options: ForkOptions) -> "InMemoryStorageState":
        plan, entry_ids = self._select_fork_plan(options)
        is_entry_copied = (lambda entry_id: True) if isinstance(plan, TreeForkPlan) else entry_ids.__contains__

        destination = InMemoryStorageState()
        message_count = 0
        for entry in self._entries_by_seq:
            if not is_entry_copied(entry.id):
                continue
            destination._entries[entry.id] = entry
            destination._entries_by_seq.append(entry)
            if entry.type == "message":
                message_count += 1
        destination._stats = SessionStats(message_count=message_count)

        for stored in self._scalar_values.values():
            projected = project_fork_current_state_write(
                ValueSetWrite(
                    seq=stored.seq,
                    namespace=stored.address.namespace,
                    key=stored.address.key,
                    value=stored.value,
                ),
                plan,
                is_entry_copied,
            )
            if projected is not None:
                destination._apply_value_set(projected)

        for stored in self._list_values.values():
            for element in stored.elements:
                projected = project_fork_current_state_write(
                    ListAppendWrite(
                        seq=element.seq,
                        namespace=stored.address.namespace,
                        key=stored.address.key,
                        value=element.value,
                    ),
                    plan,
                    is_entry_copied,
                )
                if projected is not None:
                    destination._apply_list_append(projected)

        destination._next_seq = self._next_seq
        return destination

    def _select_fork_plan(self, options: ForkOptions) -> Tuple[ForkCurrentStatePlan, set]:
        if isinstance(options, TreeForkOptions):
            return TreeForkPlan(), set()

        entry_ids: set = set()
        stored_tip = self.get_value(branch_tip(options.branch))
        plan = select_branch_fork(
            options,
            tip=stored_tip.value if stored_tip is not None else UNDEFINED,
            get_parent=lambda entry_id: (
                self._entries[entry_id].parent_id if entry_id in self._entries else UNDEFINED
            ),
            select_entry=entry_ids.add,
        )
        if self.get_value(lane_config(options.branch)) is None or self.get_value(lane_state(options.branch)) is None:
            raise ValueError(f"Source branch {options.branch!r} is not a configured AgentLane")
        return plan, entry_ids

    # --- sequence -----------------------------------------------------------

    def advance_next_seq(self, next_seq: int) -> None:
        if not isinstance(next_seq, int) or isinstance(next_seq, bool) or next_seq < 1 or next_seq > (1 << 53) - 1:
            raise ValueError(f"Invalid storage sequence high-water mark: {next_seq}")
        self._next_seq = max(self._next_seq, next_seq)

    def get_next_seq(self) -> int:
        return self._next_seq

    # --- reads ----------------------------------------------------------------

    def get_entries(self, ids: List[str]) -> Dict[str, Entry]:
        return {id: self._entries[id] for id in ids if id in self._entries}

    def get_value(self, address: Value) -> Optional[StoredValue]:
        return self._scalar_values.get(physical_key(address.namespace, address.key))

    def scan_values(self, prefix: Value) -> List[StoredValue]:
        matches = [
            stored
            for stored in self._scalar_values.values()
            if stored.address.namespace == prefix.namespace and stored.address.key.startswith(prefix.key)
        ]
        matches.sort(key=lambda stored: stored.address.key)
        return matches

    def read_list(self, address: ValueList, options: Optional[ListReadOptions] = None) -> List[ListElement]:
        resolved = resolve_list_read_options(options)
        stored = self._list_values.get(physical_key(address.namespace, address.key))
        elements = stored.elements if stored is not None else []
        if resolved.cursor is not None:
            if resolved.order == "asc":
                elements = [e for e in elements if e.seq > resolved.cursor.seq]
            else:
                elements = [e for e in elements if e.seq < resolved.cursor.seq]
        if resolved.order == "desc":
            elements = list(reversed(elements))
        else:
            elements = list(elements)
        return elements[: resolved.limit]

    def scan_branch(self, query: StorageBranchScan) -> List[Entry]:
        start = self._entries.get(query.start)
        if start is None:
            raise ValueError(f"Unknown branch start: {query.start}")

        path: List[Entry] = []
        entry: Optional[Entry] = start
        while entry is not None:
            path.append(entry)
            if entry.parent_id is None:
                break
            entry = self._entries.get(entry.parent_id)
            if entry is None:
                raise ValueError("Corrupt branch: missing parent")
        if query.order == "oldestFirst":
            path.reverse()

        stopped: List[Entry] = []
        for candidate in path:
            stopped.append(candidate)
            if candidate.id == query.stop_at_id or candidate.type == query.stop_at_type:
                break
        filtered = [
            candidate
            for candidate in stopped
            if (query.type is None or candidate.type == query.type)
            and (query.custom_type is None or candidate.custom_type == query.custom_type)
        ]
        if query.cursor is not None:
            if query.order == "oldestFirst":
                filtered = [c for c in filtered if c.seq > query.cursor.seq]
            else:
                filtered = [c for c in filtered if c.seq < query.cursor.seq]
        if query.limit is not None:
            filtered = filtered[: max(0, query.limit)]
        return filtered

    def scan_branch_structure(self, query: StorageBranchScan) -> List[EntryStructure]:
        return [
            EntryStructure(
                id=entry.id,
                parent_id=entry.parent_id,
                seq=entry.seq,
                timestamp=entry.timestamp,
                type=entry.type,
                custom_type=entry.custom_type,
            )
            for entry in self.scan_branch(query)
        ]

    def scan_entries(self, query: EntryScan) -> List[Entry]:
        limit = math.inf if query.limit is None else max(0, math.trunc(query.limit))
        entries: List[Entry] = []
        descending = query.order == "desc"
        index = len(self._entries_by_seq) - 1 if descending else 0
        while 0 <= index < len(self._entries_by_seq) and len(entries) < limit:
            entry = self._entries_by_seq[index]
            if (
                (query.type is None or entry.type == query.type)
                and (query.custom_type is None or entry.custom_type == query.custom_type)
                and (query.from_seq is None or entry.seq >= query.from_seq)
                and (query.to_seq is None or entry.seq <= query.to_seq)
            ):
                entries.append(entry)
            index += -1 if descending else 1
        return entries

    def scan_usage(self, query: UsageScan) -> List[UsageRow]:
        rows = [
            row
            for row in self._usage.values()
            if (query.from_seq is None or row.seq >= query.from_seq)
            and (query.to_seq is None or row.seq <= query.to_seq)
        ]
        rows.sort(key=lambda row: -row.seq if query.order == "desc" else row.seq)
        if query.limit is not None:
            rows = rows[: max(0, query.limit)]
        return rows

    def get_stats(self) -> SessionStats:
        return self._stats

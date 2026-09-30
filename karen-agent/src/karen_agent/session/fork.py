"""Snapshot-based fork construction (pi's `harness/session/fork.ts`).

`create_fork_snapshot` builds the complete logical state for a forked destination
session from a materialized source snapshot. Backends with indexed durable state
(JsonlStorage) use `jsonl/fork.py`'s streaming two-pass fork instead.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from karen_ai.types import KarenBase
from pydantic import ConfigDict, Field

from .fork_policy import (
    UNDEFINED,
    BranchForkPlan,
    ForkCurrentStatePlan,
    TreeForkPlan,
    project_fork_current_state_write,
    select_branch_fork,
)
from .types import BranchForkOptions, Entry, ForkOptions, TreeForkOptions, ValueSetWrite
from .values import StoredValue, branch_tip, lane_config, lane_state, value


class ForkSourceSnapshot(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entries: List[Entry]
    scalar_values: List[StoredValue]
    #: False when a backend supplied only the requested branch rather than the full tree.
    entries_complete: bool = True


class ForkDestinationSnapshot(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entries: Dict[str, Entry]
    scalar_values: List[StoredValue]
    next_seq: int


def _stored_values_in_namespace(values: List[StoredValue], namespace: str) -> List[StoredValue]:
    return [stored for stored in values if stored.address.namespace == namespace]


def _find_stored_value(values: List[StoredValue], namespace: str, key: str) -> Optional[StoredValue]:
    for stored in values:
        if stored.address.namespace == namespace and stored.address.key == key:
            return stored
    return None


def create_fork_snapshot(source: ForkSourceSnapshot, options: ForkOptions) -> ForkDestinationSnapshot:
    """Build the complete logical state for a forked destination session."""
    source_entries = {entry.id: entry for entry in source.entries}
    source_tips = _stored_values_in_namespace(source.scalar_values, branch_tip("").namespace)
    _validate_fork_source_snapshot(source, source_entries, source_tips, options)

    entry_ids, plan = _select_fork_contents(source_entries, source_tips, options)
    entries = {id: source_entries[id] for id in entry_ids}

    scalar_values: List[StoredValue] = []
    next_seq = max([0] + [entry.seq for entry in entries.values()]) + 1
    for stored in source.scalar_values:
        projected = project_fork_current_state_write(
            ValueSetWrite(
                seq=stored.seq,
                namespace=stored.address.namespace,
                key=stored.address.key,
                value=stored.value,
            ),
            plan,
            lambda entry_id: entry_id in entry_ids,
        )
        if projected is not None:
            scalar_values.append(
                StoredValue(
                    address=value(projected.namespace, projected.key),
                    value=projected.value,
                    seq=next_seq,
                )
            )
            next_seq += 1

    return ForkDestinationSnapshot(entries=entries, scalar_values=scalar_values, next_seq=next_seq)


def _select_fork_contents(
    source_entries: Dict[str, Entry],
    source_tips: List[StoredValue],
    options: ForkOptions,
) -> tuple:
    entry_ids: set = set()
    if isinstance(options, TreeForkOptions):
        entry_ids.update(source_entries.keys())
        return entry_ids, TreeForkPlan()

    assert isinstance(options, BranchForkOptions)
    source_tip = next((stored for stored in source_tips if stored.address.key == options.branch), None)
    plan = select_branch_fork(
        options,
        tip=source_tip.value if source_tip is not None else UNDEFINED,
        get_parent=lambda entry_id: (source_entries[entry_id].parent_id if entry_id in source_entries else UNDEFINED),
        select_entry=entry_ids.add,
    )
    return entry_ids, plan


def _validate_fork_source_snapshot(
    source: ForkSourceSnapshot,
    source_entries: Dict[str, Entry],
    source_tips: List[StoredValue],
    options: ForkOptions,
) -> None:
    source_tip_keys = {stored.address.key for stored in source_tips}

    lane_config_ns = lane_config("").namespace
    lane_state_ns = lane_state("").namespace
    for stored in source.scalar_values:
        if stored.address.namespace in (lane_config_ns, lane_state_ns) and stored.address.key not in source_tip_keys:
            raise ValueError(f"Source session branch {stored.address.key!r} is missing branch.tip")
    for tip in source_tips:
        configuration = _find_stored_value(source.scalar_values, lane_config_ns, tip.address.key)
        state = _find_stored_value(source.scalar_values, lane_state_ns, tip.address.key)
        if (configuration is None) != (state is None):
            raise ValueError(f"Source session branch {tip.address.key!r} has incomplete lane state")
        if isinstance(options, BranchForkOptions) and tip.address.key == options.branch and configuration is None:
            raise ValueError(f"Source branch {options.branch!r} is not a configured AgentLane")
        if (
            (source.entries_complete or isinstance(options, TreeForkOptions))
            and tip.value is not None
            and tip.value not in source_entries
        ):
            raise ValueError(f"Source session branch {tip.address.key!r} has an unknown tip")

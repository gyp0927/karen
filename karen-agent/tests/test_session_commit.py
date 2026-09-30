"""Commit stamping and validation (pi's commit.ts behaviors)."""

import pytest

from karen_ai import TextContent, Usage, UserMessage
from karen_agent.session.commit import (
    commit_write,
    insert_entry,
    insert_usage,
    prepare_storage_commit,
    validate_committed_writes,
)
from karen_agent.session.types import (
    CustomEntry,
    EntryWrite,
    MessageEntry,
    UsageRow,
    UsageWrite,
)
from karen_agent.session.values import set_value, value


def _entry(id, parent_id=None):
    return MessageEntry(id=id, parent_id=parent_id, message=UserMessage(content=f"msg {id}", timestamp=0))


class _State:
    def __init__(self, entry_ids=(), usage_ids=()):
        self.entry_ids = set(entry_ids)
        self.usage_ids = set(usage_ids)

    def has_entry_or_usage_id(self, id):
        return id in self.entry_ids or id in self.usage_ids

    def has_entry_id(self, id):
        return id in self.entry_ids


def test_insert_and_stamp():
    write = insert_entry(_entry("a"))
    assert isinstance(write, EntryWrite)
    committed = commit_write(write, 7, 1234)
    assert committed.entry.seq == 7
    assert committed.entry.timestamp == 1234
    # The source write is untouched.
    assert write.entry.seq == 0

    usage = insert_usage(UsageRow(id="u1", usage=Usage(input=1), adjustment=False))
    assert isinstance(usage, UsageWrite)
    assert commit_write(usage, 8, 1234).row.seq == 8

    scalar = set_value(value("app.c"), 1)
    assert commit_write(scalar, 9, 1234).seq == 9
    assert scalar.seq == 0


def test_prepare_storage_commit_assigns_monotonic_seqs():
    prepared = prepare_storage_commit([insert_entry(_entry("a")), insert_entry(_entry("b", "a"))], 3, 99)
    assert prepared.first_seq == 3
    assert prepared.seqs == [3, 4]
    assert prepared.timestamp == 99
    assert [w.entry.seq for w in prepared.writes] == [3, 4]


def test_validate_rejects_non_monotonic_seq():
    committed = prepare_storage_commit([insert_entry(_entry("a"))], 5, 1).writes
    # first_seq 5 implies previous seq 4; replaying the same write at seq 5 is fine...
    validate_committed_writes(committed, 5, _State())
    # ...but claiming first_seq 6 makes the write's seq 5 non-monotonic.
    with pytest.raises(ValueError, match="Non-monotonic storage sequence: 5"):
        validate_committed_writes(committed, 6, _State())


def test_validate_rejects_duplicate_ids():
    writes = prepare_storage_commit(
        [insert_entry(_entry("a")), insert_entry(_entry("a"))], 1, 1
    ).writes
    with pytest.raises(ValueError, match="Duplicate entry or usage id: a"):
        validate_committed_writes(writes, 1, _State())

    existing = prepare_storage_commit([insert_entry(_entry("x"))], 1, 1).writes
    with pytest.raises(ValueError, match="Duplicate entry or usage id: x"):
        validate_committed_writes(existing, 1, _State(entry_ids={"x"}))

    # Entry and usage share one id namespace.
    usage = prepare_storage_commit(
        [insert_usage(UsageRow(id="x", usage=Usage(), adjustment=False))], 1, 1
    ).writes
    with pytest.raises(ValueError, match="Duplicate entry or usage id: x"):
        validate_committed_writes(usage, 1, _State(entry_ids={"x"}))


def test_validate_requires_known_parents():
    orphan = prepare_storage_commit([insert_entry(_entry("b", "missing"))], 1, 1).writes
    with pytest.raises(ValueError, match="Missing parent entry: missing"):
        validate_committed_writes(orphan, 1, _State())

    # Parent earlier in the same transaction is fine.
    chained = prepare_storage_commit(
        [insert_entry(_entry("a")), insert_entry(_entry("b", "a"))], 1, 1
    ).writes
    validate_committed_writes(chained, 1, _State())

    # Parent already durable is fine.
    child = prepare_storage_commit([insert_entry(_entry("c", "a"))], 3, 1).writes
    validate_committed_writes(child, 3, _State(entry_ids={"a"}))


def test_custom_entry_round_trips_through_commit():
    write = insert_entry(CustomEntry(id="c1", parent_id=None, custom_type="note", data={"x": 1}))
    committed = commit_write(write, 2, 10)
    assert committed.entry.custom_type == "note"
    assert committed.entry.seq == 2

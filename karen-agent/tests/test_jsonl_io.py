"""JSONL IO primitives (pi's jsonl-io.test.ts): atomic publication and transaction framing."""

import json
import os

import pytest

from karen_ai import Usage, UserMessage
from karen_agent.session import insert_entry, insert_usage
from karen_agent.session.commit import commit_write
from karen_agent.session.types import MessageEntry, UsageRow
from karen_agent.session.jsonl.io import (
    parse_jsonl_transaction,
    publish_file_atomically,
    publish_jsonl,
    serialize_jsonl_transaction,
)
from karen_agent.session.jsonl.types import JSONL_STORAGE_VERSION, JsonlStorageHeader
from karen_agent.session.values import append_list, session_name, set_value, value_list


def _header():
    return JsonlStorageHeader(
        v=4, kind="header", id="s1", storage_version=JSONL_STORAGE_VERSION, created_at=1, cwd="E:/repo"
    )


def _stamped_entry(id, seq, timestamp=10):
    return commit_write(
        insert_entry(MessageEntry(id=id, parent_id=None, message=UserMessage(content=id, timestamp=0))),
        seq,
        timestamp,
    )


async def test_atomic_publish_keeps_destination_unchanged_until_complete(tmp_path):
    destination = tmp_path / "f.jsonl"
    destination.write_text("original\n", encoding="utf-8")
    seen_mid_write = None

    async def body(append):
        nonlocal seen_mid_write
        await append("staged\n")
        seen_mid_write = destination.read_text(encoding="utf-8")

    await publish_file_atomically(str(destination), body)
    assert seen_mid_write == "original\n"
    assert destination.read_text(encoding="utf-8") == "staged\n"
    assert not (tmp_path / "f.jsonl.tmp").exists()


async def test_atomic_publish_failure_discards_partial_content(tmp_path):
    destination = tmp_path / "f.jsonl"
    destination.write_text("original\n", encoding="utf-8")

    async def body(append):
        await append("partial\n")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await publish_file_atomically(str(destination), body)
    assert destination.read_text(encoding="utf-8") == "original\n"
    assert not (tmp_path / "f.jsonl.tmp").exists()


async def test_publish_jsonl_framing(tmp_path):
    destination = tmp_path / "s.jsonl"

    async def transactions(append):
        await append([_stamped_entry("a", 1)])
        await append([_stamped_entry("b", 2), commit_write(set_value(session_name, "x"), 3, 10)])

    await publish_jsonl(str(destination), _header(), transactions)
    lines = destination.read_text(encoding="utf-8").split("\n")
    assert json.loads(lines[0])["kind"] == "header"
    assert isinstance(json.loads(lines[1]), dict)  # single write is a bare object
    assert isinstance(json.loads(lines[2]), list)  # multi-write transaction stays an array
    assert lines[3] == ""


async def test_publish_jsonl_header_only(tmp_path):
    destination = tmp_path / "s.jsonl"

    async def transactions(append):
        return None

    await publish_jsonl(str(destination), _header(), transactions)
    assert destination.read_text(encoding="utf-8").count("\n") == 1


def test_transaction_round_trip_all_write_kinds():
    writes = [
        _stamped_entry("a", 1),
        commit_write(insert_usage(UsageRow(id="u1", usage=Usage(input=2), adjustment=True, details={"s": "x"})), 2, 10),
        commit_write(set_value(session_name, "name"), 3, 10),
        commit_write(append_list(value_list("app.events"), {"n": 1}), 4, 10),
    ]
    line = serialize_jsonl_transaction(writes)
    parsed = parse_jsonl_transaction(line)
    assert [w.kind for w in parsed] == ["entry", "usage", "value", "list"]
    assert parsed[0].entry.id == "a"
    assert parsed[0].entry.message.content == "a"
    assert parsed[1].row.usage.input == 2 and parsed[1].row.adjustment is True
    assert parsed[2].value == "name"
    assert parsed[3].value == {"n": 1}

    # Single write round-trips as a bare object.
    single = parse_jsonl_transaction(serialize_jsonl_transaction([_stamped_entry("z", 9)]))
    assert single[0].entry.id == "z" and single[0].entry.seq == 9


def test_parse_rejects_garbage():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_jsonl_transaction("{nope")
    with pytest.raises(ValueError, match="write seq"):
        parse_jsonl_transaction('{"kind":"usage","seq":0}')
    with pytest.raises(ValueError, match="transaction write"):
        parse_jsonl_transaction("[42]")

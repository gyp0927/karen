"""JsonlStorage persistence and torn-tail behaviors (pi's jsonl-storage.test.ts)."""

import json
import os

import pytest

from karen_ai import Usage, UserMessage
from karen_agent.session import insert_entry, insert_usage
from karen_agent.session.types import MessageEntry, UsageRow
from karen_agent.session.jsonl import JSONL_FORMAT_VERSION, JSONL_STORAGE_VERSION, JsonlStorage
from karen_agent.session.jsonl.codec import LegacyV3UnsupportedError
from karen_agent.session.jsonl.types import JsonlStorageHeader
from karen_agent.session.values import append_list, delete_list, session_name, set_value, value_list


def _header(id="s1", **overrides):
    kwargs = dict(
        v=JSONL_FORMAT_VERSION,
        kind="header",
        id=id,
        storage_version=JSONL_STORAGE_VERSION,
        created_at=1,
        cwd="E:/repo",
    )
    kwargs.update(overrides)
    return JsonlStorageHeader(**kwargs)


def _entry_write(id, parent_id=None, text=None):
    return insert_entry(
        MessageEntry(id=id, parent_id=parent_id, message=UserMessage(content=text or f"msg {id}", timestamp=0))
    )


def _lines(path):
    return path.read_text(encoding="utf-8").split("\n")


async def test_create_writes_header_only_and_commit_appends_one_line(tmp_path):
    path = str(tmp_path / "s.jsonl")
    storage = await JsonlStorage.create(path, _header(), [])
    header = json.loads(_lines(tmp_path / "s.jsonl")[0])
    assert _lines(tmp_path / "s.jsonl")[1:] == [""]  # header-only file
    assert header["kind"] == "header" and header["v"] == 4 and header["id"] == "s1"
    assert header["storageVersion"] == 1 and header["cwd"] == "E:/repo"
    assert "nextSeq" not in header and "parentSessionId" not in header

    await storage.commit([_entry_write("a"), set_value(session_name, "demo")])
    await storage.commit([_entry_write("b", "a")])
    lines = _lines(tmp_path / "s.jsonl")
    assert len(lines) == 4  # header + 2 transactions + trailing ""
    multi = json.loads(lines[1])
    assert isinstance(multi, list) and [w["kind"] for w in multi] == ["entry", "value"]
    single = json.loads(lines[2])
    assert isinstance(single, dict) and single["kind"] == "entry"
    assert single["seq"] == 3 and single["parentId"] == "a"
    assert single["message"]["role"] == "user"
    assert single["timestamp"] > 0
    await storage.close()


async def test_reopen_replays_state_and_continues_seq(tmp_path):
    path = str(tmp_path / "s.jsonl")
    storage = await JsonlStorage.create(path, _header(), [])
    await storage.commit([_entry_write("a", None, "hello")])
    await storage.commit(
        [insert_usage(UsageRow(id="u1", usage=Usage(input=3, output=2, total_tokens=5), adjustment=False))]
    )
    await storage.close()

    reopened = await JsonlStorage.open(path)
    entries = await reopened.get_entries(["a"])
    assert entries["a"].message.content == "hello"
    assert isinstance(entries["a"].message, UserMessage)  # coerced back into karen-ai models
    stats = await reopened.get_stats()
    assert stats.message_count == 1 and stats.usage.input == 3

    result = await reopened.commit([_entry_write("b", "a")])
    assert result.first_seq == 3
    assert result.stats.message_count == 2
    await reopened.close()


async def test_whole_list_deletion_does_not_resurrect_appends(tmp_path):
    path = str(tmp_path / "s.jsonl")
    events = value_list("app.events")
    storage = await JsonlStorage.create(path, _header(), [])
    await storage.commit([append_list(events, "x"), append_list(events, "y")])
    await storage.commit([delete_list(events)])
    await storage.close()

    reopened = await JsonlStorage.open(path)
    assert await reopened.read_list(events) == []
    await reopened.commit([append_list(events, "after")])
    await reopened.close()

    reread = await JsonlStorage.open(path)
    assert [e.value for e in await reread.read_list(events)] == ["after"]


async def test_header_next_seq_advances_sequences(tmp_path):
    # Simulate a snapshot rewrite (as fork publishes): header carries nextSeq.
    path = tmp_path / "s.jsonl"
    header = _header(next_seq=50)
    path.write_text(
        json.dumps(header.model_dump(mode="json", by_alias=True, exclude_none=True)) + "\n",
        encoding="utf-8",
    )
    storage = await JsonlStorage.open(str(path))
    result = await storage.commit([_entry_write("a")])
    assert result.first_seq == 50


# ---------------------------------------------------------------------------
# Torn tails and malformed lines
# ---------------------------------------------------------------------------


async def test_torn_final_line_discarded_and_file_repaired(tmp_path):
    path = tmp_path / "s.jsonl"
    storage = await JsonlStorage.create(str(path), _header(), [])
    await storage.commit([_entry_write("kept")])
    await storage.close()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        handle.write('{"kind":"entry","id":"torn"')  # unterminated, partial JSON

    reopened = await JsonlStorage.open(str(path))
    assert await reopened.get_entries(["torn"]) == {}
    assert (await reopened.get_entries(["kept"]))["kept"].seq == 1
    # The file was truncated back to complete lines before admitting writes.
    assert not path.read_text(encoding="utf-8").endswith('{"kind":"entry","id":"torn"')
    result = await reopened.commit([_entry_write("after", "kept")])
    assert result.first_seq == 2


async def test_torn_array_line_discarded_wholly(tmp_path):
    path = tmp_path / "s.jsonl"
    events = value_list("app.events")
    storage = await JsonlStorage.create(str(path), _header(), [])
    await storage.commit([append_list(events, "real")])
    await storage.close()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        handle.write('[{"kind":"list","op":"append","seq":2,"namespace":"app.events"')

    reopened = await JsonlStorage.open(str(path))
    assert [e.value for e in await reopened.read_list(events)] == ["real"]


async def test_malformed_interior_line_rejected_without_rewrite(tmp_path):
    path = tmp_path / "s.jsonl"
    storage = await JsonlStorage.create(str(path), _header(), [])
    await storage.commit([_entry_write("a")])
    await storage.close()
    original = path.read_text(encoding="utf-8")
    with open(path, "a", encoding="utf-8", newline="") as handle:
        handle.write("not json\n")

    with pytest.raises(ValueError, match=r"line 3"):
        await JsonlStorage.open(str(path))
    assert path.read_text(encoding="utf-8") == original + "not json\n"


async def test_complete_malformed_final_line_rejected(tmp_path):
    path = tmp_path / "s.jsonl"
    await (await JsonlStorage.create(str(path), _header(), [])).close()
    original = path.read_text(encoding="utf-8")
    with open(path, "a", encoding="utf-8", newline="") as handle:
        handle.write('{"kind":"bogus","seq":1}\n')
    with pytest.raises(ValueError, match=r"line 2") as excinfo:
        await JsonlStorage.open(str(path))
    assert "Invalid JSONL write kind" in str(excinfo.value.__cause__)
    assert path.read_text(encoding="utf-8") == original + '{"kind":"bogus","seq":1}\n'


async def test_invalid_write_shapes_rejected(tmp_path):
    path = tmp_path / "s.jsonl"
    await (await JsonlStorage.create(str(path), _header(), [])).close()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        handle.write('{"kind":"value","op":"upsert","seq":1,"namespace":"x","key":"y"}\n')
    with pytest.raises(ValueError, match=r"line 2") as excinfo:
        await JsonlStorage.open(str(path))
    assert "Invalid JSONL value operation: upsert" in str(excinfo.value.__cause__)


async def test_unterminated_header_rejected(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"kind":"header"', encoding="utf-8")
    with pytest.raises(ValueError, match="missing header"):
        await JsonlStorage.open(str(path))


async def test_unsupported_storage_version_rejected_without_repair(tmp_path):
    path = tmp_path / "s.jsonl"
    header = _header(storage_version=99)
    path.write_text(
        json.dumps(header.model_dump(mode="json", by_alias=True, exclude_none=True))
        + "\n"
        + '{"kind":"entry","id":"torn"',  # torn tail must NOT be repaired on this failure
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported storage version 99"):
        await JsonlStorage.open(str(path))
    assert path.read_text(encoding="utf-8").endswith('{"kind":"entry","id":"torn"')


async def test_legacy_v3_header_rejected(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps(
            {"type": "session", "version": 3, "id": "old", "timestamp": "2024-01-01T00:00:00.000Z", "cwd": "/x"}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(LegacyV3UnsupportedError, match="legacy v3"):
        await JsonlStorage.open(str(path))


async def test_closed_storage_rejects_operations(tmp_path):
    path = str(tmp_path / "s.jsonl")
    storage = await JsonlStorage.create(path, _header(), [])
    await storage.close()
    await storage.close()  # idempotent
    with pytest.raises(RuntimeError, match="JsonlStorage is closed"):
        await storage.commit([])
    with pytest.raises(RuntimeError, match="JsonlStorage is closed"):
        await storage.get_stats()

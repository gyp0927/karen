"""JsonlSessionRepo lifecycle, cwd scoping, fork, and resume (pi's jsonl-session-repo.test.ts)."""

import json
import os

import pytest

from karen_ai import Usage, UserMessage
from karen_agent.session import (
    BranchForkOptions,
    BranchScan,
    LaneConfiguration,
    LaneModelRef,
    LaneState,
    TreeForkOptions,
    insert_usage,
)
from karen_agent.session.types import UsageRow
from karen_agent.session.jsonl import (
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionRepo,
    session_directory_name,
    session_file_name,
)
from karen_agent.session.values import lane_config, lane_state, session_name


def _user(text):
    return UserMessage(content=text, timestamp=0)


async def _make_lane(session, name="main"):
    await session.create_branch(name, None)
    await session.set_value(
        lane_config(name),
        LaneConfiguration(
            model=LaneModelRef(provider="deepseek", model_id="deepseek-v4-pro"),
            thinking_level="medium",
        ).model_dump(mode="json", by_alias=True),
    )
    await session.set_value(lane_state(name), LaneState().model_dump(mode="json", by_alias=True))


def test_directory_and_file_naming():
    assert session_directory_name("/repo") == "--repo--"
    assert session_directory_name("E:\\code\\karen") == "--E--code-karen--"
    name = session_file_name(1_700_000_000_000, "abc-123")
    assert name == "2023-11-14T22-13-20-000Z_abc-123.jsonl"


async def test_create_persists_metadata_and_list_filters_by_cwd(tmp_path):
    repo = JsonlSessionRepo(str(tmp_path / "sessions"), now=lambda: 1_700_000_000_000)
    cwd_a = str(tmp_path / "proj-a")
    cwd_b = str(tmp_path / "proj-b")

    session = await repo.create(JsonlSessionCreateOptions(cwd=cwd_a, id="a1"))
    assert session.metadata.id == "a1"
    assert session.metadata.created_at == 1_700_000_000_000
    assert session.metadata.cwd == os.path.abspath(cwd_a)
    assert session.metadata.path.endswith("_a1.jsonl")
    assert os.path.basename(os.path.dirname(session.metadata.path)) == session_directory_name(
        os.path.abspath(cwd_a)
    )
    await session.close()

    other = await repo.create(JsonlSessionCreateOptions(cwd=cwd_b, id="b1"))
    await other.close()

    everything = await repo.list()
    assert [m.id for m in everything] == ["a1", "b1"]  # created_at tie -> id ascending
    only_a = await repo.list(JsonlSessionListOptions(cwd=cwd_a))
    assert [m.id for m in only_a] == ["a1"]
    assert only_a[0].modified_at > 0

    missing_root = JsonlSessionRepo(str(tmp_path / "nope"))
    assert await missing_root.list() == []


async def test_resume_round_trip(tmp_path):
    root = str(tmp_path / "sessions")
    cwd = str(tmp_path / "proj")
    repo = JsonlSessionRepo(root)

    session = await repo.create(JsonlSessionCreateOptions(cwd=cwd))
    branch = await session.create_branch("main", None)
    first = await branch.append_message(_user("hello"))
    await session.set_name("demo")
    session_id = session.metadata.id
    await session.close()

    # Resume: discover, reopen, verify state, keep appending.
    listed = await repo.list(JsonlSessionListOptions(cwd=cwd))
    assert [m.id for m in listed] == [session_id]
    resumed = await repo.open(listed[0])
    assert await resumed.get_name() == "demo"
    entries = await (await resumed.branch("main")).find_entries(BranchScan(order="oldestFirst"))
    assert [e.id for e in entries] == [first]
    assert entries[0].message.content == "hello"
    assert isinstance(entries[0].message, UserMessage)

    second = await (await resumed.branch("main")).append_message(_user("world"))
    await resumed.close()

    again = await repo.open(listed[0])
    entries = await (await again.branch("main")).find_entries(BranchScan(order="oldestFirst"))
    assert [e.id for e in entries] == [first, second]
    assert entries[1].parent_id == first


async def test_open_rejects_identity_mismatch_and_missing_files(tmp_path):
    repo = JsonlSessionRepo(str(tmp_path / "sessions"))
    session = await repo.create(JsonlSessionCreateOptions(cwd=str(tmp_path)))
    metadata = session.metadata
    await session.close()

    wrong = metadata.model_copy(update={"id": "different"})
    with pytest.raises(ValueError, match="identity does not match"):
        await repo.open(wrong)

    missing = metadata.model_copy(update={"path": str(tmp_path / "gone.jsonl")})
    with pytest.raises(ValueError, match="does not exist"):
        await repo.open(missing)

    open_session = await repo.open(metadata)
    with pytest.raises(ValueError, match="already open"):
        await repo.open(metadata)
    with pytest.raises(ValueError, match="Session is open"):
        await repo.delete(metadata)
    await open_session.close()
    await repo.delete(metadata)
    assert await repo.list() == []
    with pytest.raises(ValueError, match="does not exist"):
        await repo.delete(metadata)


async def test_create_identity_rules(tmp_path):
    repo = JsonlSessionRepo(str(tmp_path / "sessions"))
    cwd = str(tmp_path / "proj")

    session = await repo.create(JsonlSessionCreateOptions(cwd=cwd, id="dup"))
    # Same cwd + id: rejected while open, and after close by the file-existence check.
    with pytest.raises(ValueError, match="Session already exists"):
        await repo.create(JsonlSessionCreateOptions(cwd=cwd, id="dup"))
    await session.close()
    with pytest.raises(ValueError, match="Session already exists"):
        await repo.create(JsonlSessionCreateOptions(cwd=cwd, id="dup"))

    # Same id in a different cwd is fine.
    other = await repo.create(JsonlSessionCreateOptions(cwd=str(tmp_path / "other"), id="dup"))
    await other.close()


# ---------------------------------------------------------------------------
# Fork
# ---------------------------------------------------------------------------


async def _fork_source(repo, cwd):
    session = await repo.create(JsonlSessionCreateOptions(cwd=cwd))
    await _make_lane(session, "main")
    branch = await session.branch("main")
    a = await branch.append_message(_user("a"))
    b = await branch.append_message(_user("b"))
    await session.set_name("source")
    await session.set_label(b, "mid")

    async def usage_commit(mutator):
        await mutator.commit([insert_usage(UsageRow(id="u1", usage=Usage(input=9), adjustment=False))])

    await session.mutate(usage_commit)
    return session, (a, b)


async def test_fork_open_source_branch_scope(tmp_path):
    repo = JsonlSessionRepo(str(tmp_path / "sessions"))
    cwd = str(tmp_path / "proj")
    session, (a, b) = await _fork_source(repo, cwd)

    fork = await repo.fork(session.metadata, BranchForkOptions(branch="main", entry_id=a))
    assert fork.metadata.parent_session_id == session.metadata.id
    entries = await (await fork.branch("main")).find_entries(BranchScan(order="oldestFirst"))
    assert [e.id for e in entries] == [a]
    assert await fork.get_name() == "source"
    assert (await fork.get_stats()).usage.input == 0  # usage rows are not copied

    # Source is untouched and still writable.
    await (await session.branch("main")).append_message(_user("c"))
    entries = await (await fork.branch("main")).find_entries()
    assert len(entries) == 1

    # The fork destination id is claimed until close.
    with pytest.raises(ValueError, match="Session already exists"):
        await repo.create(JsonlSessionCreateOptions(cwd=cwd, id=fork.metadata.id))
    await fork.close()
    await session.close()


async def test_fork_closed_source_tree_scope(tmp_path):
    repo = JsonlSessionRepo(str(tmp_path / "sessions"))
    cwd = str(tmp_path / "proj")
    session, (a, b) = await _fork_source(repo, cwd)
    source_metadata = session.metadata
    await session.close()

    tree = await repo.fork(source_metadata, TreeForkOptions(id="forked"))
    assert tree.metadata.parent_session_id == source_metadata.id
    entries = await (await tree.branch("main")).find_entries(BranchScan(order="oldestFirst"))
    assert [e.id for e in entries] == [a, b]
    assert entries[0].message.content == "a"
    # Sequence numbers are preserved from the source file (branch create, lane
    # config/state, and per-append tip writes occupy interleaved sequences).
    assert [e.seq for e in entries] == [4, 6]

    await tree.close()

    # The fork file itself is a valid standalone session.
    listed = await repo.list(JsonlSessionListOptions(cwd=cwd))
    by_id = {m.id: m for m in listed}
    assert set(by_id) == {source_metadata.id, "forked"}
    reopened = await repo.open(by_id["forked"])
    assert (await reopened.get_stats()).message_count == 2
    await reopened.close()


async def test_fork_unknown_source_file(tmp_path):
    repo = JsonlSessionRepo(str(tmp_path / "sessions"))
    session, _ = await _fork_source(repo, str(tmp_path / "proj"))
    metadata = session.metadata
    await session.close()
    os.remove(metadata.path)
    with pytest.raises(ValueError):
        await repo.fork(metadata, TreeForkOptions())

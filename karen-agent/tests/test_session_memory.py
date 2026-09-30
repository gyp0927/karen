"""StorageBackedSession + MemorySessionRepo behaviors (pi's storage-backed-session,
session-create-branch, memory-session-repo, and fork-policy tests)."""

import asyncio

import pytest

from karen_ai import Usage, UserMessage
from karen_ai.providers import faux_assistant_message
from karen_agent.session import (
    BranchForkOptions,
    BranchScan,
    EntryCursor,
    EntryQuery,
    LaneConfiguration,
    LaneModelRef,
    LaneState,
    MemorySessionRepo,
    SessionBranchExistsError,
    SessionCreateOptions,
    SessionInvariantError,
    SessionInvalidBranchError,
    SessionPendingAssistantMessageError,
    SessionUnknownTargetError,
    TreeForkOptions,
    insert_entry,
    insert_usage,
)
from karen_agent.session.commit import commit_write
from karen_agent.session.types import EntryWrite, MessageEntry, UsageRow
from karen_agent.session.values import (
    ListReadOptions,
    ListCursor,
    append_list,
    branch_tip,
    entry_label,
    lane_config,
    lane_state,
    operation_result,
    operation_state,
    pending_entry,
    session_name,
    set_value,
    value,
    value_list,
)


def _user(text, timestamp=0):
    return UserMessage(content=text, timestamp=timestamp)


def _lane_config():
    return LaneConfiguration(
        model=LaneModelRef(provider="deepseek", model_id="deepseek-v4-pro"),
        thinking_level="medium",
        active_tool_names=["read"],
    )


def _usage(n):
    return Usage(input=n, output=n, total_tokens=2 * n)


async def _make_lane(session, name="main"):
    """Create a branch that also counts as a configured AgentLane."""
    await session.create_branch(name, None)
    await session.set_value(lane_config(name), _lane_config().model_dump(mode="json", by_alias=True))
    await session.set_value(lane_state(name), LaneState().model_dump(mode="json", by_alias=True))


# ---------------------------------------------------------------------------
# Basic session behaviors
# ---------------------------------------------------------------------------


async def test_create_uses_injected_clock_for_identity():
    repo = MemorySessionRepo(now=lambda: 1_700_000_000_000)
    session = await repo.create()
    assert session.metadata.created_at == 1_700_000_000_000
    assert session.metadata.storage_version == 1
    # uuidv7 embeds the timestamp in the first 48 bits.
    assert session.metadata.id.startswith("018bcfe5-6800-7")


async def test_append_message_advances_tip_and_stats():
    repo = MemorySessionRepo()
    session = await repo.create()
    branch = await session.create_branch("main", None)
    assert await branch.get_tip_id() is None

    first = await branch.append_message(_user("one"))
    second = await branch.append_message(_user("two"))
    assert await branch.get_tip_id() == second

    entries = await branch.find_entries()
    assert [e.id for e in entries] == [second, first]  # newestFirst by default
    assert entries[1].parent_id is None
    assert entries[0].parent_id == first
    assert entries[0].message.content == "two"

    stats = await session.get_stats()
    assert stats.message_count == 2


async def test_append_custom_entry_and_branch_scans():
    repo = MemorySessionRepo()
    session = await repo.create()
    branch = await session.create_branch("main", None)
    a = await branch.append_message(_user("a"))
    b = await branch.append_custom_entry("note", {"n": 1})
    c = await branch.append_message(_user("c"))

    all_entries = await branch.find_entries(BranchScan(order="oldestFirst"))
    assert [e.id for e in all_entries] == [a, b, c]
    assert all_entries[1].custom_type == "note"
    assert all_entries[1].data == {"n": 1}

    # stopAtId includes the stop entry itself.
    stopped = await branch.find_entries(BranchScan(order="oldestFirst", stop_at_id=b))
    assert [e.id for e in stopped] == [a, b]

    stopped_by_type = await branch.find_entries(BranchScan(stop_at_type="custom"))
    assert [e.id for e in stopped_by_type] == [c, b]  # default newestFirst

    only_messages = await branch.find_entries(BranchScan(type="message"))
    assert [e.id for e in only_messages] == [c, a]

    custom_only = await branch.find_entries(BranchScan(custom_type="note"))
    assert [e.id for e in custom_only] == [b]

    limited = await branch.find_entries(BranchScan(limit=2))
    assert [e.id for e in limited] == [c, b]

    # find_entry clamps limit to 1.
    assert (await branch.find_entry()).id == c


async def test_create_branch_validation():
    repo = MemorySessionRepo()
    session = await repo.create()
    with pytest.raises(SessionInvalidBranchError):
        await session.create_branch("", None)
    with pytest.raises(SessionInvalidBranchError):
        await session.create_branch("a\x00b", None)

    await session.create_branch("main", None)
    with pytest.raises(SessionBranchExistsError):
        await session.create_branch("main", None)
    with pytest.raises(SessionUnknownTargetError):
        await session.create_branch("other", "missing-entry")

    assert (await session.branch("missing")) is None
    assert (await session.branch("main")).name == "main"


async def test_get_branch_tip_unknown_branch_invariant():
    # get_branch_tip is a package-internal on StorageBackedSession (pi: same).
    from karen_agent.session import SessionMetadata, StorageBackedSession
    from karen_agent.session.memory import MemoryStorage

    session = StorageBackedSession(
        SessionMetadata(id="s", created_at=0, storage_version=1), MemoryStorage()
    )
    with pytest.raises(SessionInvariantError, match="Unknown branch"):
        await session.get_branch_tip("missing")


async def test_values_lists_name_labels():
    repo = MemorySessionRepo()
    session = await repo.create()

    counter = value("app.counter")
    assert await session.get_value(counter) is None
    await session.set_value(counter, 41)
    stored = await session.get_value(counter)
    assert stored.value == 41 and stored.seq == 1
    await session.delete_value(counter)
    assert await session.get_value(counter) is None

    events = value_list("app.events")
    await session.append_list(events, "a")
    await session.append_list(events, "b")
    await session.append_list(events, "c")
    assert [e.value for e in await session.read_list(events)] == ["a", "b", "c"]
    desc = await session.read_list(events, ListReadOptions(order="desc", limit=2))
    assert [e.value for e in desc] == ["c", "b"]
    paged = await session.read_list(events, ListReadOptions(cursor=ListCursor(seq=desc[-1].seq), order="desc"))
    assert [e.value for e in paged] == ["a"]
    await session.delete_list(events)
    assert await session.read_list(events) == []

    assert await session.get_name() is None
    await session.set_name("demo")
    assert await session.get_name() == "demo"
    await session.set_name(None)
    assert await session.get_name() is None

    await session.set_label("target-1", "checkpoint")
    assert await session.get_label("target-1") == "checkpoint"
    await session.set_label("target-1", None)
    assert await session.get_label("target-1") is None


async def test_scan_values_prefix():
    repo = MemorySessionRepo()
    session = await repo.create()
    await session.set_value(value("app.cfg", "b"), 2)
    await session.set_value(value("app.cfg", "a"), 1)
    await session.set_value(value("other", "a"), 3)
    rows = await session.scan_values(value("app.cfg"))
    assert [(r.address.key, r.value) for r in rows] == [("a", 1), ("b", 2)]


async def test_find_entries_cursor_paging():
    repo = MemorySessionRepo()
    session = await repo.create()
    branch = await session.create_branch("main", None)
    ids = [await branch.append_message(_user(f"m{i}")) for i in range(5)]
    # Each append commits entry + branch-tip write, so entry seqs are 1, 3, 5, 7, 9.
    seqs = [(await session.get_entry(id)).seq for id in ids]

    newest = await session.find_entries()
    assert [e.id for e in newest] == list(reversed(ids))

    oldest_first = await session.find_entries(EntryQuery(order="asc"))
    assert [e.id for e in oldest_first] == ids

    page2 = await session.find_entries(EntryQuery(order="asc", limit=2, cursor=EntryCursor(seq=seqs[1])))
    assert [e.id for e in page2] == ids[2:4]

    desc_page = await session.find_entries(EntryQuery(order="desc", cursor=EntryCursor(seq=seqs[3])))
    assert [e.id for e in desc_page] == list(reversed(ids[:3]))

    assert await session.find_entries(EntryQuery(order="desc", cursor=EntryCursor(seq=1))) == []
    assert await session.find_entries(EntryQuery(order="asc", cursor=EntryCursor(seq=2**53 - 1))) == []

    assert (await session.find_entry()).id == ids[-1]


# ---------------------------------------------------------------------------
# Mutation line semantics
# ---------------------------------------------------------------------------


async def test_serialized_read_modify_write():
    repo = MemorySessionRepo()
    session = await repo.create()
    counter = value("app.counter")

    async def increment():
        async def job(mutator):
            stored = await mutator.get_value(counter)
            await asyncio.sleep(0)  # force interleaving opportunities
            await mutator.commit([set_value(counter, (stored.value if stored else 0) + 1)])

        await session.mutate(job)

    await asyncio.gather(*(increment() for _ in range(20)))
    assert (await session.get_value(counter)).value == 20


async def test_one_commit_per_mutation_and_guard_consumed_on_failure():
    repo = MemorySessionRepo()
    session = await repo.create()
    await session.create_branch("main", None)

    async def two_commits(mutator):
        await mutator.commit([set_value(session_name, "committed")])
        with pytest.raises(RuntimeError, match="commit already attempted"):
            await mutator.commit([])

    await session.mutate(two_commits)
    assert await session.get_name() == "committed"

    async def failing(mutator):
        bad = insert_entry(MessageEntry(id="orphan", parent_id="missing", message=_user("x")))
        with pytest.raises(ValueError, match="Missing parent entry"):
            await mutator.commit([bad])
        with pytest.raises(RuntimeError, match="commit already attempted"):
            await mutator.commit([])

    await session.mutate(failing)


async def test_mutator_invalid_after_end():
    repo = MemorySessionRepo()
    session = await repo.create()
    mutation = await session.begin_mutation()
    await mutation.commit([set_value(session_name, "x")])
    await mutation.end()
    with pytest.raises(RuntimeError, match="outside its mutation callback"):
        await mutation.commit([])
    with pytest.raises(RuntimeError, match="outside its mutation callback"):
        await mutation.get_value(session_name)
    # end is idempotent.
    await mutation.end()


async def test_pending_assistant_message_rejected():
    repo = MemorySessionRepo()
    session = await repo.create()
    branch = await session.create_branch("main", None)
    pending = faux_assistant_message("incomplete", api="openai-responses", provider="openai", model="mock")
    pending.stop_reason = "pending"

    with pytest.raises(SessionPendingAssistantMessageError):
        await branch.append_message(pending)

    async def bad_commit(mutator):
        with pytest.raises(SessionPendingAssistantMessageError):
            await mutator.commit([insert_entry(MessageEntry(id="p1", parent_id=None, message=pending))])

    await session.mutate(bad_commit)

    # A dict-shaped pending assistant (camelCase wire form) is rejected too.
    with pytest.raises(SessionPendingAssistantMessageError):
        await branch.append_message(pending.model_dump(mode="json", by_alias=True))


async def test_nested_public_writer_queues_behind_callback():
    repo = MemorySessionRepo()
    session = await repo.create()
    events = value_list("app.events")

    async def outer(mutator):
        await mutator.commit([append_list(events, "callback")])
        # Awaiting this inside the callback would deadlock; fire-and-forget queues it.
        asyncio.get_running_loop().create_task(session.append_list(events, "public"))
        await asyncio.sleep(0.01)
        assert [e.value for e in await session.read_list(events)] == ["callback"]

    await session.mutate(outer)
    await asyncio.sleep(0.05)
    assert [e.value for e in await session.read_list(events)] == ["callback", "public"]


async def test_direct_reads_observe_commit_before_mutation_ends():
    repo = MemorySessionRepo()
    session = await repo.create()
    mutation = await session.begin_mutation()
    await mutation.commit([set_value(session_name, "visible")])
    assert (await session.get_value(session_name)).value == "visible"
    await mutation.end()


async def test_close_is_idempotent_and_rejects_late_operations():
    repo = MemorySessionRepo()
    session = await repo.create()
    await session.close()
    await session.close()
    with pytest.raises(RuntimeError, match="Session is closed"):
        await session.get_stats()
    with pytest.raises(RuntimeError, match="Session is closed"):
        await session.set_value(session_name, "late")
    with pytest.raises(RuntimeError, match="Session is closed"):
        await session.begin_mutation()


# ---------------------------------------------------------------------------
# Fork
# ---------------------------------------------------------------------------


async def _build_fork_source(repo):
    session = await repo.create()
    await _make_lane(session, "main")
    branch = await session.branch("main")
    a = await branch.append_message(_user("a"))
    b = await branch.append_message(_user("b"))
    c = await branch.append_message(_user("c"))
    await session.set_name("source")
    await session.set_label(b, "middle")
    await session.set_value(operation_state("op-1"), {"at": "starting"})
    await session.set_value(pending_entry("p-1"), {"type": "message", "payload": {}})

    async def add_usage_row(mutator):
        await mutator.commit([insert_usage(UsageRow(id="u1", usage=_usage(5), adjustment=False))])

    await session.mutate(add_usage_row)
    return session, (a, b, c)


async def test_fork_branch_scope_at_and_before():
    repo = MemorySessionRepo()
    session, (a, b, c) = await _build_fork_source(repo)

    fork = await repo.fork(session.metadata, BranchForkOptions(branch="main", entry_id=b))
    fork_branch = await fork.branch("main")
    assert fork_branch is not None
    entries = await fork_branch.find_entries(BranchScan(order="oldestFirst"))
    assert [e.id for e in entries] == [a, b]
    assert await fork.get_name() == "source"
    assert await fork.get_label(b) == "middle"
    # Copied lane restarts idle.
    assert (await fork.get_value(lane_state("main"))).value["currentOperationId"] is None
    # Operation/pending/usage state is excluded.
    assert await fork.get_value(operation_state("op-1")) is None
    assert await fork.get_value(pending_entry("p-1")) is None
    assert (await fork.get_stats()).usage.input == 0
    assert (await fork.get_stats()).message_count == 2

    before = await repo.fork(session.metadata, BranchForkOptions(branch="main", entry_id=b, position="before"))
    entries = await (await before.branch("main")).find_entries(BranchScan(order="oldestFirst"))
    assert [e.id for e in entries] == [a]
    assert await before.get_label(b) is None  # label of a non-copied entry is dropped


async def test_fork_branch_scope_errors():
    repo = MemorySessionRepo()
    session, (a, b, c) = await _build_fork_source(repo)
    other = await repo.create()
    foreign = await (await other.create_branch("foreign", None)).append_message(_user("f"))

    with pytest.raises(ValueError, match="Unknown source branch"):
        await repo.fork(session.metadata, BranchForkOptions(branch="missing"))
    with pytest.raises(ValueError, match="is not on source branch"):
        await repo.fork(session.metadata, BranchForkOptions(branch="main", entry_id=foreign))

    # A data-only branch (no lane config/state) cannot be branch-forked.
    await session.create_branch("data", None)
    with pytest.raises(ValueError, match="not a configured AgentLane"):
        await repo.fork(session.metadata, BranchForkOptions(branch="data"))


async def test_fork_tree_scope_copies_all_branches():
    repo = MemorySessionRepo()
    session, (a, b, c) = await _build_fork_source(repo)
    data_branch = await session.create_branch("data", a)
    d = await data_branch.append_message(_user("d"))

    tree = await repo.fork(session.metadata, TreeForkOptions())
    main_entries = await (await tree.branch("main")).find_entries(BranchScan(order="oldestFirst"))
    assert [e.id for e in main_entries] == [a, b, c]
    data_entries = await (await tree.branch("data")).find_entries(BranchScan(order="oldestFirst"))
    assert [e.id for e in data_entries] == [a, d]
    assert tree.metadata.parent_session_id == session.metadata.id
    # Tree forks still exclude usage/operation state.
    assert (await tree.get_stats()).usage.input == 0
    assert await tree.get_value(operation_state("op-1")) is None


async def test_memory_repo_lifecycle():
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="s1"))
    assert [m.id for m in await repo.list()] == ["s1"]

    with pytest.raises(ValueError, match="already open"):
        await repo.open(session.metadata)
    await session.close()
    reopened = await repo.open(session.metadata)
    assert reopened.metadata.id == "s1"

    with pytest.raises(ValueError, match="Unknown session"):
        await repo.open(type(session.metadata)(id="nope", created_at=0, storage_version=1))

    # Cannot delete while open; can after close.
    with pytest.raises(ValueError, match="Session is open"):
        await repo.delete(session.metadata)
    await reopened.close()
    await repo.delete(session.metadata)
    assert await repo.list() == []

    with pytest.raises(ValueError, match="Session already exists"):
        await repo.create(SessionCreateOptions(id="dup"))
        await repo.create(SessionCreateOptions(id="dup"))

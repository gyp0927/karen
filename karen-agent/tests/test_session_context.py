"""Session entry -> context message projection (pi's session/context.ts)."""

import pytest
from karen_ai import UserMessage
from karen_ai.providers import faux_assistant_message

from karen_agent.messages import BranchSummaryMessage, CompactionSummaryMessage
from karen_agent.session.context import (
    SessionContextBuildOptions,
    build_context_entries,
    build_session_context,
    session_entry_to_context_messages,
)
from karen_agent.session.types import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    MessageEntry,
)


def _message_entry(id, message):
    return MessageEntry(id=id, message=message, timestamp=1)


def _user(text):
    return UserMessage(content=text, timestamp=1)


def test_build_context_entries_without_compaction():
    entries = [_message_entry("a", _user("1")), _message_entry("b", _user("2"))]
    assert build_context_entries(entries) == entries


def test_build_context_entries_keeps_latest_compaction_and_tail():
    c1 = CompactionEntry(id="c1", summary="s1", tokens_before=10, from_hook=False, timestamp=1)
    c2 = CompactionEntry(id="c2", summary="s2", tokens_before=20, from_hook=False, timestamp=2)
    tail = _message_entry("m", _user("after"))
    entries = [c1, _message_entry("x", _user("old")), c2, tail]
    assert build_context_entries(entries) == [c2, tail]


def test_session_entry_to_context_messages_filters_bad_assistant():
    for stop in ("error", "aborted", "deferred"):
        bad = faux_assistant_message("oops", stop_reason=stop)
        assert session_entry_to_context_messages(_message_entry("e", bad)) == []
    good = faux_assistant_message("fine", stop_reason="stop")
    assert session_entry_to_context_messages(_message_entry("e", good)) == [good]


def test_session_entry_to_context_messages_compaction():
    retained = [_user("keep"), faux_assistant_message("drop", stop_reason="error")]
    entry = CompactionEntry(
        id="c", summary="sum", tokens_before=42, retained_tail=retained, from_hook=False, timestamp=5
    )
    messages = session_entry_to_context_messages(entry)
    assert isinstance(messages[0], CompactionSummaryMessage)
    assert messages[0].summary == "sum"
    assert messages[0].tokens_before == 42
    assert messages[0].timestamp == 5
    # retained tail is filtered like regular messages
    assert messages[1:] == [retained[0]]


def test_session_entry_to_context_messages_branch_summary():
    entry = BranchSummaryEntry(id="b", summary="s", from_id="x", from_hook=False, timestamp=2)
    [message] = session_entry_to_context_messages(entry)
    assert isinstance(message, BranchSummaryMessage)
    assert message.from_id == "x"
    empty = BranchSummaryEntry(id="b", summary="", from_id=None, from_hook=False, timestamp=2)
    assert session_entry_to_context_messages(empty) == []


async def test_build_session_context_with_projectors():
    entries = [
        _message_entry("a", _user("hi")),
        CustomEntry(id="c1", custom_type="todo", data={"items": 2}),
        CustomEntry(id="c2", custom_type="unknown"),
    ]

    def projector(entry, context):
        assert context == "ctx"
        return [_user(f"todo:{entry.data['items']}")]

    async def async_projector(entry, context):
        return [_user("async")]

    options = SessionContextBuildOptions(entry_projectors={"todo": projector})
    messages = await build_session_context(entries, options, "ctx")
    assert [m.content for m in messages] == ["hi", "todo:2"]

    options2 = SessionContextBuildOptions(entry_projectors={"todo": async_projector, "unknown": lambda e, c: None})
    messages2 = await build_session_context(entries, options2)
    assert [m.content for m in messages2] == ["hi", "async"]


async def test_build_session_context_applies_latest_compaction_only():
    c = CompactionEntry(id="c", summary="s", tokens_before=1, from_hook=False, timestamp=1)
    entries = [_message_entry("old", _user("old")), c, _message_entry("new", _user("new"))]
    messages = await build_session_context(entries)
    assert isinstance(messages[0], CompactionSummaryMessage)
    assert messages[1].content == "new"
    assert len(messages) == 2

"""Unit tests for the pure session-tree helpers (`navigation.py`)."""

from karen_ai import TextContent, UserMessage
from karen_ai.providers import faux_assistant_message
from karen_agent.session import CompactionEntry, CustomEntry, MessageEntry
from karen_coding_agent.navigation import (
    PREVIEW_WIDTH,
    build_tree,
    entry_kind,
    entry_preview,
    entry_text,
    forkable_user_messages,
    render_tree,
)


def _message_entry(entry_id, parent_id, message, timestamp=1):
    return MessageEntry(id=entry_id, parent_id=parent_id, message=message, timestamp=timestamp)


def _user(entry_id, parent_id, text, timestamp=1):
    return _message_entry(
        entry_id, parent_id, UserMessage(content=[TextContent(text=text)], timestamp=timestamp), timestamp
    )


def _assistant(entry_id, parent_id, text="ok", timestamp=1):
    return _message_entry(entry_id, parent_id, faux_assistant_message(text), timestamp)


def _compaction(entry_id, parent_id, summary, tokens_before=9):
    return CompactionEntry(
        id=entry_id, parent_id=parent_id, summary=summary, tokens_before=tokens_before, from_hook=False
    )


# ---------------------------------------------------------------------------
# build_tree
# ---------------------------------------------------------------------------


def test_build_tree_links_children_and_keeps_root_order():
    entries = [
        _user("a", None, "one", 1),
        _assistant("b", "a", "two", 2),
        _user("c", "b", "three", 3),
    ]
    roots = build_tree(entries)
    assert [root.entry.id for root in roots] == ["a"]
    assert [child.entry.id for child in roots[0].children] == ["b"]
    assert [child.entry.id for child in roots[0].children[0].children] == ["c"]


def test_build_tree_branches_sort_children_by_timestamp():
    entries = [
        _user("a", None, "q", 1),
        _assistant("b", "a", "answer b", 30),
        _assistant("c", "a", "answer c", 20),
    ]
    roots = build_tree(entries)
    assert [child.entry.id for child in roots[0].children] == ["c", "b"]


def test_build_tree_missing_parent_becomes_a_root():
    entries = [_user("a", None, "q", 1), _assistant("b", "gone", "orphan", 2)]
    roots = build_tree(entries)
    assert [root.entry.id for root in roots] == ["a", "b"]


def test_build_tree_self_parent_is_a_root():
    entries = [_user("a", "a", "self", 1)]
    assert [root.entry.id for root in build_tree(entries)] == ["a"]


def test_build_tree_resolves_labels():
    entries = [_user("a", None, "q", 1)]
    roots = build_tree(entries, {"a": "first"})
    assert roots[0].label == "first"
    assert build_tree(entries)[0].label is None


# ---------------------------------------------------------------------------
# text / kinds / previews
# ---------------------------------------------------------------------------


def test_entry_text_reads_message_blocks_and_summaries():
    assert entry_text(_user("a", None, "hello")) == "hello"
    plain = _message_entry("b", None, UserMessage(content="bare string", timestamp=1))
    assert entry_text(plain) == "bare string"
    assert entry_text(_compaction("c", None, "compacted")) == "compacted"
    custom = CustomEntry(id="d", parent_id=None, custom_type="note", data={"a": 1})
    assert entry_text(custom) == '[note] {"a":1}'


def test_entry_kind_labels():
    assert entry_kind(_user("a", None, "hi")) == "user"
    assert entry_kind(_assistant("b", None)) == "assistant"
    assert entry_kind(_compaction("c", None, "s")) == "compaction"
    assert entry_kind(CustomEntry(id="d", parent_id=None, custom_type="note")) == "custom:note"


def test_entry_preview_collapses_and_truncates():
    entry = _user("a", None, "line one\n\n  line two")
    assert entry_preview(entry) == "line one line two"
    long_entry = _user("b", None, "x" * 200)
    preview = entry_preview(long_entry, width=10)
    assert preview == "x" * 9 + "…"
    assert len(preview) == 10


def test_forkable_user_messages_skips_empty_and_non_user_entries():
    entries = [
        _user("a", None, "first", 1),
        _assistant("b", "a", "reply", 2),
        _user("c", "b", "second", 3),
        _user("d", "c", "", 4),
        _compaction("e", "d", "compacted"),
    ]
    assert forkable_user_messages(entries) == [("a", "first"), ("c", "second")]


# ---------------------------------------------------------------------------
# render_tree
# ---------------------------------------------------------------------------


def test_render_tree_marks_tips_and_indents_children():
    entries = [
        _user("aaaaaaaa-1", None, "question", 1),
        _assistant("bbbbbbbb-2", "aaaaaaaa-1", "answer", 2),
        _user("cccccccc-3", "aaaaaaaa-1", "side question", 3),
    ]
    roots = build_tree(entries)
    rendered = render_tree(roots, leaf_id="bbbbbbbb-2", tips=["bbbbbbbb-2", "cccccccc-3"])
    lines = rendered.splitlines()
    # tree lines show the id's last 8 characters (uuid7 heads are shared)
    assert lines[0].startswith("· aaaaaa-1 user: question")  # root, not a tip
    assert lines[1].startswith("  ├─ ● bbbbbb-2 assistant: answer")
    assert lines[2].startswith("  └─ ○ cccccc-3 user: side question")


def test_render_tree_shows_labels():
    entries = [_user("aaaaaaaa-1", None, "question", 1)]
    rendered = render_tree(build_tree(entries, {"aaaaaaaa-1": "baseline"}), leaf_id="aaaaaaaa-1")
    assert "[baseline]" in rendered
    assert rendered.startswith("● aaaaaa-1 user [baseline]: question")


def test_render_tree_preview_width_is_respected():
    entries = [_user("aaaaaaaa-1", None, "y" * 200, 1)]
    rendered = render_tree(build_tree(entries), leaf_id="aaaaaaaa-1")
    assert rendered.endswith("…")
    assert len(rendered) < PREVIEW_WIDTH + 40

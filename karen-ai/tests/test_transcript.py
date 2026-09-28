"""Tests for transcript normalization and replay."""

from karen_ai import (
    Context,
    SystemMessage,
    Tool,
    ToolReference,
    UserMessage,
    get_current_system_message,
    get_current_system_prompt,
    get_current_tools,
    get_declared_tools,
    has_non_additive_tool_changes,
    has_tool_redefinitions,
    normalize_context,
    resolve_transcript_tools,
)

SEARCH = Tool(name="search", description="Search", parameters={"type": "object", "properties": {"q": {"type": "string"}}})
CALC = Tool(name="calc", description="Calculate", parameters={"type": "object", "properties": {"expr": {"type": "string"}}})


def test_normalize_context_folds_prompt_and_tools_into_leading_system_message():
    context = Context(
        system_prompt="You are helpful.",
        tools=[SEARCH],
        messages=[UserMessage(content="hi", timestamp=1)],
    )
    transcript = normalize_context(context)

    assert len(transcript.messages) == 2
    head = transcript.messages[0]
    assert head.role == "system"
    assert head.content == "You are helpful."
    assert [t.name for t in head.tools_added] == ["search"]
    assert transcript.messages[1].role == "user"


def test_normalize_context_without_prompt_or_tools_stays_unchanged():
    context = Context(messages=[UserMessage(content="hi", timestamp=1)])
    transcript = normalize_context(context)
    assert len(transcript.messages) == 1
    assert transcript.messages[0].role == "user"


def test_replay_applies_content_sections_and_tool_deltas():
    messages = [
        SystemMessage(content="Base prompt.", sections={"env": "prod"}, tools_added=[SEARCH], timestamp=0),
        UserMessage(content="q", timestamp=1),
        SystemMessage(content="More instructions.", sections={"env": None, "style": "concise"}, tools_added=[CALC], timestamp=2),
        SystemMessage(tools_removed=[ToolReference(name="search")], timestamp=3),
    ]

    current = get_current_system_message(messages)
    assert current is not None
    assert "Base prompt." in get_current_system_prompt(messages)
    assert "More instructions." in get_current_system_prompt(messages)
    assert "prod" not in get_current_system_prompt(messages)
    assert "concise" in get_current_system_prompt(messages)
    assert [t.name for t in get_current_tools(messages)] == ["calc"]
    assert [t.name for t in get_declared_tools(messages)] == ["search", "calc"]


def test_tool_redefinition_detection():
    changed_search = Tool(name="search", description="Search v2", parameters={"type": "object"})
    messages = [
        SystemMessage(tools_added=[SEARCH], timestamp=0),
        SystemMessage(tools_added=[changed_search], timestamp=1),
    ]
    assert has_tool_redefinitions(messages) is True
    assert has_non_additive_tool_changes(messages) is True


def test_resolve_transcript_tools_anchors_additions_only_when_additive():
    additive = [
        SystemMessage(tools_added=[SEARCH], timestamp=0),
        SystemMessage(tools_added=[CALC], timestamp=1),
    ]
    resolved = resolve_transcript_tools(additive, supports_tool_additions=True)
    assert resolved.anchors_additions is True
    assert [t.name for t in resolved.request_tools] == ["search"]

    resolved_no_support = resolve_transcript_tools(additive, supports_tool_additions=False)
    assert resolved_no_support.anchors_additions is False
    assert {t.name for t in resolved_no_support.request_tools} == {"search", "calc"}


def test_collapse_later_system_messages_when_unsupported():
    from karen_ai import TranscriptContext, collapse_system_messages

    context = TranscriptContext(
        messages=[
            SystemMessage(content="Base.", timestamp=0),
            UserMessage(content="q", timestamp=1),
            SystemMessage(content="Update.", timestamp=2),
            UserMessage(content="q2", timestamp=3),
        ]
    )
    collapsed = collapse_system_messages(context)
    assert [m.role for m in collapsed.messages] == ["system", "user", "user"]
    head = collapsed.messages[0]
    assert "Base." in head.content
    assert "Update." in head.content

"""Transcript normalization and replay helpers, mirroring pi-ai's utils/transcript.ts.

The leading system message is the system prompt. Later system messages change
it: `content` adds instructions from that point on, `sections` replace or
remove named prompt sections, and `tools_added`/`tools_removed` change the
tool set. Replaying every system message in order yields the current prompt
and tools.
"""

from __future__ import annotations

import json
from typing import List, Optional, Sequence

from .types import (
    Context,
    Message,
    SystemMessage,
    Tool,
    ToolReference,
    TranscriptContext,
)
from .utils.text import content_text, get_system_message_text


def create_initial_system_message(
    system_prompt: Optional[str],
    tools: Optional[List[Tool]],
) -> Optional[SystemMessage]:
    """Build the leading system message for a prompt and tool set.

    Returns None when both are empty, so an empty transcript stays empty.
    """
    has_system_prompt = system_prompt is not None and len(system_prompt) > 0
    has_tools = tools is not None and len(tools) > 0
    if not has_system_prompt and not has_tools:
        return None
    return SystemMessage(
        role="system",
        content=system_prompt or "",
        tools_added=tools if has_tools else None,
        timestamp=0,
    )


def normalize_context(context: Context) -> TranscriptContext:
    """Fold `Context.system_prompt` and `Context.tools` into a leading system message.

    This is the only entry point that produces a TranscriptContext; every
    provider-facing function expects the result.
    """
    initial_message = create_initial_system_message(context.system_prompt, context.tools)
    messages = [initial_message, *context.messages] if initial_message else list(context.messages)
    return TranscriptContext(messages=messages)


def _is_system_message(message) -> bool:
    return getattr(message, "role", None) == "system"


def get_initial_system_message(messages: Sequence) -> Optional[SystemMessage]:
    """Return the leading system message, if the transcript starts with one."""
    first = messages[0] if messages else None
    return first if first is not None and _is_system_message(first) else None


def without_initial_system_message(messages: List[Message]) -> List[Message]:
    """Drop the leading system message for APIs that carry the prompt outside the message list."""
    return messages[1:] if get_initial_system_message(messages) else messages


def get_current_tools(messages: Sequence) -> List[Tool]:
    """Resolve the tools available after applying every transcript delta in order."""
    tools: dict[str, Tool] = {}
    for message in messages:
        if not _is_system_message(message):
            continue
        for tool in message.tools_removed or []:
            tools.pop(tool.name, None)
        for tool in message.tools_added or []:
            tools[tool.name] = tool
    return list(tools.values())


def get_current_system_message(messages: Sequence) -> Optional[SystemMessage]:
    """Replay every system message into one leading system message holding the
    current prompt and tools."""
    content: list[str] = []
    sections: dict[str, str] = {}
    timestamp: Optional[int] = None
    for message in messages:
        if not _is_system_message(message):
            continue
        if timestamp is None:
            timestamp = message.timestamp
        text = content_text(message.content)
        if len(text) > 0:
            content.append(text)
        for name, value in (message.sections or {}).items():
            if value is None:
                sections.pop(name, None)
            else:
                sections[name] = value
    tools = get_current_tools(messages)
    if timestamp is None and len(tools) == 0:
        return None
    return SystemMessage(
        role="system",
        content="\n\n".join(content),
        sections=sections if sections else None,
        tools_added=tools if tools else None,
        timestamp=timestamp or 0,
    )


def get_current_system_prompt(messages: Sequence) -> str:
    """Render the current system prompt text after replaying every system message."""
    message = get_current_system_message(messages)
    return get_system_message_text(message) if message else ""


def collapse_system_messages(context: TranscriptContext) -> TranscriptContext:
    """Rebuild the transcript for APIs without mid-conversation system messages:
    the replayed system message leads, and every later system message is dropped."""
    head = get_current_system_message(context.messages)
    messages = [m for m in context.messages if m.role != "system"]
    return TranscriptContext(messages=[head, *messages] if head else messages)


def resolve_transcript(context: TranscriptContext, supports_mid_convo_system_messages: Optional[bool]) -> TranscriptContext:
    """Keep later system messages in place when the model accepts them; otherwise collapse."""
    return context if supports_mid_convo_system_messages else collapse_system_messages(context)


def to_tool_declaration(tool: Tool) -> Tool:
    """Strip executable and display-only fields from a tool before comparison or persistence."""
    return Tool(
        name=tool.name,
        description=tool.description,
        parameters=json.loads(json.dumps(tool.parameters)),
        constrained_sampling=tool.constrained_sampling,
    )


def declarations_equal(left: Tool, right: Tool) -> bool:
    """Whether two tools declare the same interface to the model."""
    return (
        to_tool_declaration(left).model_dump_json(by_alias=True)
        == to_tool_declaration(right).model_dump_json(by_alias=True)
    )


class ToolStateChanges:
    def __init__(self, tools_added: List[Tool], tools_removed: List[ToolReference]):
        self.tools_added = tools_added
        self.tools_removed = tools_removed


def get_tool_state_changes(previous: Sequence[Tool], current: Sequence[Tool]) -> ToolStateChanges:
    """Compare two complete tool states. A changed definition is a removal followed by an addition."""
    previous_tools = {tool.name: tool for tool in previous}
    current_tools = {tool.name: tool for tool in current}
    return ToolStateChanges(
        tools_added=[
            to_tool_declaration(tool)
            for tool in current
            if tool.name not in previous_tools or not declarations_equal(previous_tools[tool.name], tool)
        ],
        tools_removed=[
            ToolReference(name=tool.name)
            for tool in previous
            if tool.name not in current_tools or not declarations_equal(tool, current_tools[tool.name])
        ],
    )


def get_declared_tools(messages: Sequence) -> List[Tool]:
    """Every definition referenced by transcript tool state, in first-declaration order."""
    definitions: dict[str, Tool] = {}
    for message in messages:
        if not _is_system_message(message):
            continue
        for tool in message.tools_added or []:
            definitions.setdefault(tool.name, tool)
    return list(definitions.values())


def has_tool_redefinitions(messages: Sequence) -> bool:
    """Whether a tool name was declared twice with different definitions."""
    declared: dict[str, Tool] = {}
    for message in messages:
        if not _is_system_message(message):
            continue
        for tool in message.tools_added or []:
            previous = declared.get(tool.name)
            if previous is not None and not declarations_equal(previous, tool):
                return True
            declared[tool.name] = tool
    return False


def has_non_additive_tool_changes(messages: Sequence) -> bool:
    """Whether tool history contains a removal or same-name redeclaration that an
    addition-only transport cannot replay."""
    declared: set[str] = set()
    for message in messages:
        if not _is_system_message(message):
            continue
        if message.tools_removed:
            return True
        for tool in message.tools_added or []:
            if tool.name in declared:
                return True
            declared.add(tool.name)
    return False


class TranscriptTools:
    def __init__(self, request_tools: List[Tool], anchors_additions: bool):
        #: Tools sent in the top-level request field.
        self.request_tools = request_tools
        #: Whether later system messages carry their own tools_added as in-place additions.
        self.anchors_additions = anchors_additions


def resolve_transcript_tools(messages: Sequence, supports_tool_additions: bool) -> TranscriptTools:
    """Split tool declarations between the top-level request field and in-place additions."""
    anchors_additions = supports_tool_additions and not has_non_additive_tool_changes(messages)
    initial = get_initial_system_message(messages)
    return TranscriptTools(
        request_tools=(initial.tools_added or []) if anchors_additions and initial else get_current_tools(messages),
        anchors_additions=anchors_additions,
    )

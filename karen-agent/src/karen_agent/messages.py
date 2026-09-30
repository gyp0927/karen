"""Harness message shapes and LLM conversion (pi's `harness/messages.ts`).

The agent loop works with ``AgentMessage`` throughout; besides the karen-ai
message roles (``system`` / ``user`` / ``assistant`` / ``toolResult``) the
harness adds four custom roles carried as plain transcript entries:
``bashExecution``, ``custom``, ``branchSummary`` and ``compactionSummary``.
:func:`convert_to_llm` maps them to user messages at the LLM call boundary.

Messages loaded back from a session file stay plain dicts (their roles are not
part of karen-ai's ``Message`` union), so every accessor here is duck-typed:
pydantic models expose snake_case attributes while stored dicts carry the
camelCase wire keys.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from karen_ai import ImageContent, Message, TextContent, UserMessage
from karen_ai.types import KarenBase

from .types import AgentMessage

__all__ = [
    "COMPACTION_SUMMARY_PREFIX",
    "COMPACTION_SUMMARY_SUFFIX",
    "BRANCH_SUMMARY_PREFIX",
    "BRANCH_SUMMARY_SUFFIX",
    "BashExecutionMessage",
    "CustomMessage",
    "BranchSummaryMessage",
    "CompactionSummaryMessage",
    "bash_execution_to_text",
    "create_branch_summary_message",
    "create_compaction_summary_message",
    "create_custom_message",
    "convert_to_llm",
]

COMPACTION_SUMMARY_PREFIX = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

BRANCH_SUMMARY_PREFIX = "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"
BRANCH_SUMMARY_SUFFIX = "</summary>"


class BashExecutionMessage(KarenBase):
    """A bash command executed outside the tool loop, kept in the transcript."""

    role: Literal["bashExecution"] = "bashExecution"
    command: str
    output: str
    exit_code: Optional[int] = None
    cancelled: bool
    truncated: bool
    full_output_path: Optional[str] = None
    timestamp: int
    exclude_from_context: Optional[bool] = None


class CustomMessage(KarenBase):
    """Application-defined message; always converted to a user message for the LLM."""

    role: Literal["custom"] = "custom"
    custom_type: str
    content: Union[str, List[Union[TextContent, ImageContent]]]
    display: bool
    details: Optional[Any] = None
    timestamp: int


class BranchSummaryMessage(KarenBase):
    """Summary of a branch the conversation came back from."""

    role: Literal["branchSummary"] = "branchSummary"
    summary: str
    from_id: Optional[str] = None
    timestamp: int


class CompactionSummaryMessage(KarenBase):
    """Summary replacing compacted history."""

    role: Literal["compactionSummary"] = "compactionSummary"
    summary: str
    tokens_before: int
    timestamp: int


def message_field(message: Any, *names: str, default: Any = None) -> Any:
    """Read a field from a pydantic message model or a stored plain dict.

    Accepts both snake_case attribute names and camelCase wire keys so callers
    can handle live models and JSONL-loaded dicts uniformly.
    """
    for name in names:
        if isinstance(message, dict):
            if name in message:
                return message[name]
        else:
            value = getattr(message, name, None)
            if value is not None:
                return value
            if hasattr(message, name):
                return value
    return default


def _to_timestamp_ms(timestamp: Union[str, int, float]) -> int:
    if isinstance(timestamp, (int, float)):
        return int(timestamp)
    text = timestamp
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return int(datetime.fromisoformat(text).timestamp() * 1000)


def bash_execution_to_text(message: Any) -> str:
    """Render a bashExecution message as the user-message text the LLM sees."""
    text = f"Ran `{message_field(message, 'command')}`\n"
    output = message_field(message, "output")
    if output:
        text += f"```\n{output}\n```"
    else:
        text += "(no output)"
    if message_field(message, "cancelled"):
        text += "\n\n(command cancelled)"
    else:
        exit_code = message_field(message, "exit_code", "exitCode")
        if exit_code is not None and exit_code != 0:
            text += f"\n\nCommand exited with code {exit_code}"
    full_output_path = message_field(message, "full_output_path", "fullOutputPath")
    if message_field(message, "truncated") and full_output_path:
        text += f"\n\n[Output truncated. Full output: {full_output_path}]"
    return text


def create_branch_summary_message(
    summary: str, from_id: Optional[str], timestamp: Union[str, int, float]
) -> BranchSummaryMessage:
    return BranchSummaryMessage(summary=summary, from_id=from_id, timestamp=_to_timestamp_ms(timestamp))


def create_compaction_summary_message(
    summary: str, tokens_before: int, timestamp: Union[str, int, float]
) -> CompactionSummaryMessage:
    return CompactionSummaryMessage(
        summary=summary, tokens_before=tokens_before, timestamp=_to_timestamp_ms(timestamp)
    )


def create_custom_message(
    custom_type: str,
    content: Union[str, List[Union[TextContent, ImageContent]]],
    display: bool,
    details: Any,
    timestamp: Union[str, int, float],
) -> CustomMessage:
    return CustomMessage(
        custom_type=custom_type,
        content=content,
        display=display,
        details=details,
        timestamp=_to_timestamp_ms(timestamp),
    )


def convert_to_llm(messages: List[AgentMessage]) -> List[Message]:
    """Map harness messages to LLM messages, dropping non-context roles."""
    result: List[Message] = []
    for message in messages:
        role = message_field(message, "role")
        timestamp = message_field(message, "timestamp", default=0)
        if role == "bashExecution":
            if message_field(message, "exclude_from_context", "excludeFromContext"):
                continue
            result.append(
                UserMessage(
                    content=[TextContent(text=bash_execution_to_text(message))],
                    timestamp=timestamp,
                )
            )
        elif role == "custom":
            content = message_field(message, "content", default="")
            if isinstance(content, str):
                content = [TextContent(text=content)]
            result.append(UserMessage(content=content, timestamp=timestamp))
        elif role == "branchSummary":
            summary = message_field(message, "summary", default="")
            result.append(
                UserMessage(
                    content=[TextContent(text=BRANCH_SUMMARY_PREFIX + summary + BRANCH_SUMMARY_SUFFIX)],
                    timestamp=timestamp,
                )
            )
        elif role == "compactionSummary":
            summary = message_field(message, "summary", default="")
            result.append(
                UserMessage(
                    content=[TextContent(text=COMPACTION_SUMMARY_PREFIX + summary + COMPACTION_SUMMARY_SUFFIX)],
                    timestamp=timestamp,
                )
            )
        elif role in ("system", "user", "assistant", "toolResult"):
            result.append(message)
        # Unknown roles never reach the model.
    return result


def safe_json_stringify(value: Any) -> str:
    """``JSON.stringify`` with pi's compaction fallbacks ("undefined" / "[unserializable]")."""
    try:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return "[unserializable]"

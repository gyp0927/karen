"""File-operation tracking and conversation serialization (pi's `harness/compaction/utils.ts`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Set, Tuple

from karen_ai.utils import content_text

from ..messages import message_field, safe_json_stringify
from ..types import AgentMessage

__all__ = [
    "FileOperations",
    "create_file_ops",
    "extract_file_ops_from_message",
    "compute_file_lists",
    "format_file_operations",
    "serialize_conversation",
]

TOOL_RESULT_MAX_CHARS = 2000


@dataclass
class FileOperations:
    """File paths touched by a session branch or compaction range."""

    #: Files read but not necessarily modified.
    read: Set[str] = field(default_factory=set)
    #: Files written by full-file write operations.
    written: Set[str] = field(default_factory=set)
    #: Files modified by edit operations.
    edited: Set[str] = field(default_factory=set)


def create_file_ops() -> FileOperations:
    """Create an empty file-operation accumulator."""
    return FileOperations()


def _block_field(block: Any, name: str) -> Any:
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def extract_file_ops_from_message(message: AgentMessage, file_ops: FileOperations) -> None:
    """Add file operations from assistant tool calls to an accumulator."""
    if message_field(message, "role") != "assistant":
        return
    content = message_field(message, "content")
    if not isinstance(content, list):
        return
    for block in content:
        if _block_field(block, "type") != "toolCall":
            continue
        args = _block_field(block, "arguments")
        if not isinstance(args, dict):
            continue
        path = args.get("path")
        if not isinstance(path, str) or not path:
            continue
        name = _block_field(block, "name")
        if name == "read":
            file_ops.read.add(path)
        elif name == "write":
            file_ops.written.add(path)
        elif name == "edit":
            file_ops.edited.add(path)


def compute_file_lists(file_ops: FileOperations) -> Tuple[List[str], List[str]]:
    """Compute sorted (read-only, modified) file lists from accumulated operations."""
    modified = file_ops.edited | file_ops.written
    read_only = sorted(f for f in file_ops.read if f not in modified)
    return read_only, sorted(modified)


def format_file_operations(read_files: List[str], modified_files: List[str]) -> str:
    """Format file lists as summary metadata tags."""
    sections: List[str] = []
    if read_files:
        sections.append(f"<read-files>\n{'\n'.join(read_files)}\n</read-files>")
    if modified_files:
        sections.append(f"<modified-files>\n{'\n'.join(modified_files)}\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


def _truncate_for_summary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated_chars = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated_chars} more characters truncated]"


def serialize_conversation(messages: List[Any]) -> str:
    """Serialize LLM messages to plain text for summarization prompts."""
    parts: List[str] = []

    for message in messages:
        role = message_field(message, "role")
        content = message_field(message, "content", default="")
        if role == "user":
            text = content_text(content, "")
            if text:
                parts.append(f"[User]: {text}")
        elif role == "assistant":
            thinking_parts: List[str] = []
            tool_calls: List[str] = []
            has_text = False
            for block in content if isinstance(content, list) else []:
                block_type = _block_field(block, "type")
                if block_type == "thinking":
                    thinking_parts.append(_block_field(block, "thinking") or "")
                elif block_type == "toolCall":
                    args = _block_field(block, "arguments") or {}
                    args_str = ", ".join(f"{k}={safe_json_stringify(v)}" for k, v in args.items())
                    tool_calls.append(f"{_block_field(block, 'name')}({args_str})")
                elif block_type == "text":
                    has_text = True
            if thinking_parts:
                parts.append(f"[Assistant thinking]: {'\n'.join(thinking_parts)}")
            if has_text:
                parts.append(f"[Assistant]: {content_text(content)}")
            if tool_calls:
                parts.append(f"[Assistant tool calls]: {'; '.join(tool_calls)}")
        elif role == "toolResult":
            text = content_text(content, "")
            if text:
                parts.append(f"[Tool result]: {_truncate_for_summary(text, TOOL_RESULT_MAX_CHARS)}")

    return "\n\n".join(parts)

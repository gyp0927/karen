"""Build model-context message lists from session entries (pi's `harness/session/context.ts`)."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

from karen_ai.types import KarenBase

from ..messages import (
    create_branch_summary_message,
    create_compaction_summary_message,
    message_field,
)
from ..types import AgentMessage
from .types import CompactionEntry, Entry, EntryProjector

__all__ = [
    "SessionContextBuildOptions",
    "build_context_entries",
    "session_entry_to_context_messages",
    "build_session_context",
]


class SessionContextBuildOptions(KarenBase):
    model_config = {"arbitrary_types_allowed": True}

    #: custom_type -> projector turning application-defined custom entries into context messages.
    entry_projectors: Optional[Dict[str, EntryProjector]] = None


def build_context_entries(path_entries: List[Entry]) -> List[Entry]:
    """Keep only the latest compaction entry and everything after it."""
    compaction: Optional[CompactionEntry] = None
    compaction_index = -1
    for index in range(len(path_entries) - 1, -1, -1):
        entry = path_entries[index]
        if entry.type == "compaction":
            compaction = entry  # type: ignore[assignment]
            compaction_index = index
            break
    if compaction is None:
        return list(path_entries)
    return [compaction, *path_entries[compaction_index + 1 :]]


def _is_context_message(message: AgentMessage) -> bool:
    if message_field(message, "role") != "assistant":
        return True
    return message_field(message, "stop_reason", "stopReason") not in ("error", "aborted", "deferred")


def session_entry_to_context_messages(entry: Entry) -> List[AgentMessage]:
    """Convert one session entry into the messages it contributes to model context."""
    if entry.type == "message":
        return [entry.message] if _is_context_message(entry.message) else []  # type: ignore[attr-defined]
    if entry.type == "compaction":
        compaction = entry  # type: ignore[assignment]
        return [
            create_compaction_summary_message(compaction.summary, compaction.tokens_before, compaction.timestamp),
            *[m for m in compaction.retained_tail if _is_context_message(m)],
        ]
    if entry.type == "branch_summary":
        if not entry.summary:  # type: ignore[attr-defined]
            return []
        return [create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)]  # type: ignore[attr-defined]
    return []  # custom entries contribute only through entry_projectors


async def build_session_context(
    path_entries: List[Entry],
    options: Optional[SessionContextBuildOptions] = None,
    context: Any = None,
) -> List[AgentMessage]:
    """Build the full context message list for a branch path.

    ``context`` is forwarded to entry projectors (pi passes its chord Context;
    karen keeps it an opaque application value).
    """
    entries = build_context_entries(path_entries)
    messages: List[AgentMessage] = []
    for entry in entries:
        if entry.type != "custom":
            messages.extend(session_entry_to_context_messages(entry))
            continue
        projector = (options.entry_projectors or {}).get(entry.custom_type) if options else None  # type: ignore[attr-defined]
        if projector is None:
            continue
        projected = projector(entry, context)
        if inspect.isawaitable(projected):
            projected = await projected
        if projected:
            messages.extend(projected)
    return messages

"""Session tree and fork views (pi's `session-manager.ts` `getTree` plus the
fork-selector helpers of `agent-session.ts`).

Pure functions over karen-agent session entries: the parent/child tree, the
one-line previews `/tree` and RPC `get_tree` show, the user messages a fork
can target, and a terminal rendering of the tree. The operations that move
the branch tip or create sessions live on `AgentSession`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from karen_agent.messages import message_field
from karen_agent.session import Entry

__all__ = [
    "PREVIEW_WIDTH",
    "TreeNode",
    "build_tree",
    "entry_kind",
    "entry_preview",
    "entry_text",
    "forkable_user_messages",
    "render_tree",
]

#: Longest preview kept for a tree line (pi's TUI truncates to the pane width).
PREVIEW_WIDTH = 72


@dataclass
class TreeNode:
    """pi's `SessionTreeNode`: an entry, its children, and its resolved label."""

    entry: Entry
    children: List["TreeNode"] = field(default_factory=list)
    label: Optional[str] = None


def build_tree(entries: Iterable[Entry], labels: Optional[Mapping[str, str]] = None) -> List[TreeNode]:
    """Build pi's `getTree()`: nodes in entry order, children oldest-first.

    Entries whose parent is missing from the scan (an orphaned branch that was
    rewound away) become roots, exactly like pi.
    """
    labels = labels or {}
    ordered = list(entries)
    nodes: Dict[str, TreeNode] = {
        entry.id: TreeNode(entry=entry, label=labels.get(entry.id)) for entry in ordered
    }

    roots: List[TreeNode] = []
    for entry in ordered:
        node = nodes[entry.id]
        parent_id = entry.parent_id
        if parent_id is None or parent_id == entry.id:
            roots.append(node)
            continue
        parent = nodes.get(parent_id)
        if parent is None:
            roots.append(node)  # orphan: treat as root, like pi
        else:
            parent.children.append(node)

    stack = list(roots)
    while stack:
        node = stack.pop()
        node.children.sort(key=lambda child: child.entry.timestamp)
        stack.extend(node.children)
    return roots


def entry_text(entry: Entry) -> str:
    """The model-visible text of an entry (pi's `contentText` over messages)."""
    entry_type = getattr(entry, "type", None)
    if entry_type == "message":
        message = getattr(entry, "message", None)
        if message is None:
            return ""
        content = message_field(message, "content", default=[])
        if isinstance(content, str):  # user messages may carry a bare string
            return content
        return "".join(
            block.text
            for block in content or []
            if message_field(block, "type") == "text" and message_field(block, "text")
        )
    if entry_type in ("compaction", "branch_summary"):
        return getattr(entry, "summary", "") or ""
    if entry_type == "custom":
        data = getattr(entry, "data", None)
        return f"[{getattr(entry, 'custom_type', 'custom')}] {_compact_json(data)}"
    return ""


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def entry_kind(entry: Entry) -> str:
    """Short kind label: the message role, or the entry type."""
    entry_type = getattr(entry, "type", None)
    if entry_type == "message":
        return str(message_field(getattr(entry, "message", None), "role", default="message"))
    if entry_type == "branch_summary":
        return "branchSummary"
    if entry_type == "custom":
        return f"custom:{getattr(entry, 'custom_type', 'custom')}"
    return str(entry_type)


def entry_preview(entry: Entry, width: int = PREVIEW_WIDTH) -> str:
    """One collapsed line for the entry, truncated like pi's TUI rows."""
    text = " ".join(entry_text(entry).split())
    if len(text) > width:
        text = text[: width - 1] + "…"
    return text


def forkable_user_messages(entries: Iterable[Entry]) -> List[Tuple[str, str]]:
    """pi's `getUserMessagesForForking`: (entryId, text) for every user message."""
    result: List[Tuple[str, str]] = []
    for entry in entries:
        if getattr(entry, "type", None) != "message":
            continue
        message = getattr(entry, "message", None)
        if message_field(message, "role") != "user":
            continue
        text = entry_text(entry)
        if text:
            result.append((entry.id, text))
    return result


def render_tree(
    roots: Sequence[TreeNode],
    *,
    leaf_id: Optional[str] = None,
    tips: Sequence[str] = (),
    width: int = PREVIEW_WIDTH,
) -> str:
    """Render the tree for the terminal: `●` leaf, `○` other branch tips."""
    tip_set = set(tips)
    lines: List[str] = []

    def marker(node: TreeNode) -> str:
        if node.entry.id == leaf_id:
            return "●"
        if node.entry.id in tip_set:
            return "○"
        return "·"

    def walk(node: TreeNode, prefix: str, is_last: Optional[bool]) -> None:
        connector = "" if is_last is None else ("└─ " if is_last else "├─ ")
        label = f" [{node.label}]" if node.label else ""
        lines.append(
            f"{prefix}{connector}{marker(node)} {node.entry.id[-8:]} "
            f"{entry_kind(node.entry)}{label}: {entry_preview(node.entry, width)}"
        )
        if is_last is None:
            child_prefix = "  "
        else:
            child_prefix = prefix + ("   " if is_last else "│  ")
        for index, child in enumerate(node.children):
            walk(child, child_prefix, index == len(node.children) - 1)

    for root in roots:
        walk(root, "", None)
    return "\n".join(lines)

"""JSON event stream for `karen --mode json` (pi coding-agent's
`modes/json-event.ts`).

Every agent event becomes one JSON line on stdout. `message_update` events
are reshaped exactly like pi: the cumulative `partial` assistant snapshot is
stripped from the streaming sub-event (`message_start` provides the initial
message, deltas build it, `message_end` provides the final authoritative
one), cumulative `usage` is kept (constant size), and `toolcall_start`
gains the tool call's `id`/`toolName`. All other events serialize as-is in
camelCase wire form (`None` fields omitted, like pi's JSON.stringify).
"""

from __future__ import annotations

from typing import Any, Dict

from karen_ai import ToolCall
from karen_agent.session.jsonl import to_jsonable


def to_json_event(event: Any) -> Dict[str, Any]:
    """Convert one agent event to its JSON wire shape (pi's `toJsonEvent`)."""
    if getattr(event, "type", None) != "message_update":
        return to_jsonable(event)
    message = event.message
    if getattr(message, "role", None) != "assistant":
        raise ValueError("message_update message is not an assistant message")
    assistant_event = event.assistant_message_event
    data = to_jsonable(assistant_event)
    data.pop("partial", None)
    if assistant_event.type == "toolcall_start":
        tool_call = assistant_event.partial.content[assistant_event.content_index]
        if not isinstance(tool_call, ToolCall):
            raise ValueError(
                f"toolcall_start content at index {assistant_event.content_index} is not a tool call"
            )
        data["id"] = tool_call.id
        data["toolName"] = tool_call.name
    return {
        "type": "message_update",
        "usage": to_jsonable(message.usage),
        "assistantMessageEvent": data,
    }


__all__ = ["to_json_event"]

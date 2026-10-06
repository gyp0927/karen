"""Unit tests for the `--mode json` event conversion (pi's json-event.ts)."""

from karen_ai import DoneEvent, TextDeltaEvent, ToolCallStartEvent
from karen_ai.providers import faux_assistant_message, faux_tool_call
from karen_agent.types import MessageUpdateEvent, ToolExecutionStartEvent
from karen_coding_agent.json_events import to_json_event


def test_passthrough_event_serializes_camel_case():
    event = ToolExecutionStartEvent(tool_call_id="t1", tool_name="find", args={"pattern": "*.py"})
    assert to_json_event(event) == {
        "type": "tool_execution_start",
        "toolCallId": "t1",
        "toolName": "find",
        "args": {"pattern": "*.py"},
    }


def test_message_update_strips_partial_and_keeps_usage():
    message = faux_assistant_message("hel")
    event = MessageUpdateEvent(
        message=message,
        assistant_message_event=TextDeltaEvent(content_index=0, delta="hel", partial=message),
    )
    data = to_json_event(event)
    assert data["type"] == "message_update"
    assert "message" not in data  # only cumulative usage survives
    assert "usage" in data
    assert data["assistantMessageEvent"] == {"type": "text_delta", "contentIndex": 0, "delta": "hel"}


def test_message_update_toolcall_start_gains_id_and_tool_name():
    message = faux_assistant_message([faux_tool_call("find", {"pattern": "*.py"}, id="t1")])
    event = MessageUpdateEvent(
        message=message,
        assistant_message_event=ToolCallStartEvent(content_index=0, partial=message),
    )
    data = to_json_event(event)["assistantMessageEvent"]
    assert data == {"type": "toolcall_start", "contentIndex": 0, "id": "t1", "toolName": "find"}


def test_message_update_done_keeps_the_final_message():
    message = faux_assistant_message("hi")
    event = MessageUpdateEvent(
        message=message,
        assistant_message_event=DoneEvent(reason="stop", message=message),
    )
    data = to_json_event(event)["assistantMessageEvent"]
    assert data["type"] == "done"
    assert data["reason"] == "stop"
    assert data["message"]["role"] == "assistant"
    assert "partial" not in data

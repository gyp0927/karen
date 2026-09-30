"""Compaction utils: file operations + conversation serialization (pi's compaction/utils.ts)."""

from karen_ai import TextContent, ThinkingContent, ToolCall, UserMessage
from karen_ai.providers import faux_assistant_message

from karen_agent.compaction.utils import (
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)
from karen_agent.messages import create_custom_message


def _assistant_with_calls(*calls):
    blocks = [
        ToolCall(id=f"call-{i}", name=name, arguments=args) for i, (name, args) in enumerate(calls)
    ]
    return faux_assistant_message(blocks)


def test_extract_file_ops():
    ops = create_file_ops()
    extract_file_ops_from_message(
        _assistant_with_calls(
            ("read", {"path": "a.py"}),
            ("write", {"path": "b.py"}),
            ("edit", {"path": "c.py"}),
            ("bash", {"command": "ls"}),  # not a file op
            ("read", {"limit": 10}),  # no path -> skipped
            ("read", {"path": 42}),  # non-string path -> skipped
        ),
        ops,
    )
    assert ops.read == {"a.py"}
    assert ops.written == {"b.py"}
    assert ops.edited == {"c.py"}


def test_extract_file_ops_ignores_non_assistant():
    ops = create_file_ops()
    extract_file_ops_from_message(UserMessage(content="read x", timestamp=0), ops)
    assert ops.read == set()


def test_extract_file_ops_handles_stored_dict_message():
    ops = create_file_ops()
    extract_file_ops_from_message(
        {"role": "assistant", "content": [{"type": "toolCall", "name": "edit", "arguments": {"path": "d.py"}}]},
        ops,
    )
    assert ops.edited == {"d.py"}


def test_compute_file_lists_modified_wins():
    ops = create_file_ops()
    ops.read.update(["z.py", "a.py", "m.py"])
    ops.written.add("m.py")
    ops.edited.add("b.py")
    read_files, modified_files = compute_file_lists(ops)
    assert read_files == ["a.py", "z.py"]
    assert modified_files == ["b.py", "m.py"]


def test_format_file_operations():
    assert format_file_operations(["a"], ["b"]) == (
        "\n\n<read-files>\na\n</read-files>\n\n<modified-files>\nb\n</modified-files>"
    )
    assert format_file_operations(["a"], []) == "\n\n<read-files>\na\n</read-files>"
    assert format_file_operations([], []) == ""


def test_serialize_conversation_full():
    assistant = faux_assistant_message(
        [
            ThinkingContent(thinking="let me think"),
            TextContent(text="answer"),
            ToolCall(id="t1", name="read", arguments={"path": "a.py", "limit": 5}),
        ]
    )
    text = serialize_conversation([UserMessage(content="hello", timestamp=0), assistant])
    assert text == (
        "[User]: hello\n\n"
        "[Assistant thinking]: let me think\n\n"
        "[Assistant]: answer\n\n"
        '[Assistant tool calls]: read(path="a.py", limit=5)'
    )


def test_serialize_conversation_tool_result_truncation():
    from karen_ai import ToolResultMessage

    long_output = "x" * 2500
    result = ToolResultMessage(
        tool_call_id="t", tool_name="bash", content=[TextContent(text=long_output)], is_error=False, timestamp=0
    )
    text = serialize_conversation([result])
    assert text == f"[Tool result]: {'x' * 2000}\n\n[... 500 more characters truncated]"


def test_serialize_conversation_skips_empty_content():
    text = serialize_conversation([UserMessage(content="", timestamp=0)])
    assert text == ""


def test_serialize_conversation_custom_message_as_user():
    """convert_to_llm output feeds serialize_conversation: custom became user."""
    custom = create_custom_message("note", "note text", True, None, 0)
    from karen_agent.messages import convert_to_llm

    text = serialize_conversation(convert_to_llm([custom]))
    assert text == "[User]: note text"

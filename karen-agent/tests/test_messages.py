"""Harness message shapes and convert_to_llm (pi's messages.ts)."""

from karen_ai import TextContent, UserMessage

from karen_agent.messages import (
    BRANCH_SUMMARY_PREFIX,
    BRANCH_SUMMARY_SUFFIX,
    COMPACTION_SUMMARY_PREFIX,
    COMPACTION_SUMMARY_SUFFIX,
    BashExecutionMessage,
    CompactionSummaryMessage,
    bash_execution_to_text,
    convert_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
    safe_json_stringify,
)


def test_bash_execution_to_text_full():
    msg = BashExecutionMessage(command="ls", output="a\nb", exit_code=0, cancelled=False, truncated=False, timestamp=1)
    assert bash_execution_to_text(msg) == "Ran `ls`\n```\na\nb\n```"


def test_bash_execution_to_text_no_output():
    msg = BashExecutionMessage(command="true", output="", exit_code=0, cancelled=False, truncated=False, timestamp=1)
    assert bash_execution_to_text(msg) == "Ran `true`\n(no output)"


def test_bash_execution_to_text_cancelled():
    msg = BashExecutionMessage(command="x", output="", exit_code=None, cancelled=True, truncated=False, timestamp=1)
    assert bash_execution_to_text(msg).endswith("\n\n(command cancelled)")


def test_bash_execution_to_text_nonzero_exit():
    msg = BashExecutionMessage(command="x", output="", exit_code=3, cancelled=False, truncated=False, timestamp=1)
    assert bash_execution_to_text(msg).endswith("\n\nCommand exited with code 3")


def test_bash_execution_to_text_truncated_footer():
    msg = BashExecutionMessage(
        command="x", output="o", exit_code=0, cancelled=False, truncated=True, full_output_path="/tmp/f", timestamp=1
    )
    assert bash_execution_to_text(msg).endswith("\n\n[Output truncated. Full output: /tmp/f]")
    # truncated without a path prints no footer
    msg2 = BashExecutionMessage(command="x", output="o", exit_code=0, cancelled=False, truncated=True, timestamp=1)
    assert "truncated" not in bash_execution_to_text(msg2)


def test_create_messages_accept_str_timestamps():
    branch = create_branch_summary_message("s", "e1", "2026-09-30T00:00:00Z")
    assert branch.role == "branchSummary"
    assert branch.from_id == "e1"
    assert branch.timestamp > 0
    compaction = create_compaction_summary_message("s", 123, 1000)
    assert compaction.tokens_before == 123
    assert compaction.timestamp == 1000
    custom = create_custom_message("note", "hello", True, None, 5)
    assert custom.custom_type == "note"
    assert custom.display is True


def test_convert_to_llm_bash_execution():
    msg = BashExecutionMessage(command="ls", output="o", exit_code=0, cancelled=False, truncated=False, timestamp=7)
    [converted] = convert_to_llm([msg])
    assert converted.role == "user"
    assert converted.timestamp == 7
    assert converted.content[0].text == "Ran `ls`\n```\no\n```"


def test_convert_to_llm_bash_execution_excluded():
    msg = BashExecutionMessage(
        command="ls", output="o", exit_code=0, cancelled=False, truncated=False, timestamp=7,
        exclude_from_context=True,
    )
    assert convert_to_llm([msg]) == []


def test_convert_to_llm_custom_string_and_blocks():
    custom = create_custom_message("note", "hello", True, None, 3)
    [converted] = convert_to_llm([custom])
    assert converted.role == "user"
    assert converted.content[0].text == "hello"

    blocks = [TextContent(text="block")]
    custom2 = create_custom_message("note", blocks, True, None, 3)
    [converted2] = convert_to_llm([custom2])
    assert converted2.content[0].text == "block"


def test_convert_to_llm_summary_messages():
    branch = create_branch_summary_message("B", None, 1)
    compaction = create_compaction_summary_message("C", 10, 2)
    converted = convert_to_llm([branch, compaction])
    assert converted[0].content[0].text == BRANCH_SUMMARY_PREFIX + "B" + BRANCH_SUMMARY_SUFFIX
    assert converted[1].content[0].text == COMPACTION_SUMMARY_PREFIX + "C" + COMPACTION_SUMMARY_SUFFIX


def test_convert_to_llm_passes_standard_roles_and_drops_unknown():
    user = UserMessage(content="hi", timestamp=1)
    weird = {"role": "notification", "text": "x"}
    converted = convert_to_llm([user, weird])
    assert converted == [user]


def test_convert_to_llm_handles_stored_dicts():
    """Messages loaded from a session file stay plain dicts with camelCase keys."""
    stored = {"role": "bashExecution", "command": "ls", "output": "", "exitCode": 2,
              "cancelled": False, "truncated": False, "timestamp": 9}
    [converted] = convert_to_llm([stored])
    assert converted.content[0].text == "Ran `ls`\n(no output)\n\nCommand exited with code 2"

    stored_excluded = {**stored, "excludeFromContext": True}
    assert convert_to_llm([stored_excluded]) == []

    stored_compaction = {"role": "compactionSummary", "summary": "S", "tokensBefore": 5, "timestamp": 1}
    [c] = convert_to_llm([stored_compaction])
    assert c.content[0].text == COMPACTION_SUMMARY_PREFIX + "S" + COMPACTION_SUMMARY_SUFFIX


def test_safe_json_stringify():
    assert safe_json_stringify({"a": 1}) == '{"a":1}'
    assert safe_json_stringify("x") == '"x"'

    class Unserializable:
        pass

    assert safe_json_stringify(Unserializable()) == "[unserializable]"


def test_compaction_summary_message_wire_keys():
    msg = CompactionSummaryMessage(summary="s", tokens_before=3, timestamp=1)
    dumped = msg.model_dump(by_alias=True)
    assert dumped["tokensBefore"] == 3
    assert dumped["role"] == "compactionSummary"

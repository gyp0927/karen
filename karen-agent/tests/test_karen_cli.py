"""Tests for the demo CLI's pure logic (examples/karen_cli.py).

The example module is loaded by path; no network or credentials are touched —
KarenCli construction only registers providers and reads local template dirs.
"""

import importlib.util
import os
import types
from pathlib import Path

import pytest

from karen_agent.hooks import AfterToolEvent, BeforeToolEvent
from karen_agent.session import BranchScan
from karen_ai import AssistantMessage, TextContent, Usage, UserMessage

SPEC = importlib.util.spec_from_file_location(
    "karen_cli", Path(__file__).parent.parent / "examples" / "karen_cli.py"
)
karen_cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(karen_cli)


# ---------------------------------------------------------------------------
# parse_command
# ---------------------------------------------------------------------------


def test_parse_plain_prompt():
    assert karen_cli.parse_command("hello there", set()) == ("prompt", None, "hello there")


def test_parse_builtins():
    assert karen_cli.parse_command("/quit", set())[0] == "quit"
    assert karen_cli.parse_command("/exit", set())[0] == "quit"
    assert karen_cli.parse_command("/q", set())[0] == "quit"
    assert karen_cli.parse_command("/help", set())[0] == "help"
    assert karen_cli.parse_command("/new", set())[0] == "new"
    assert karen_cli.parse_command("/templates", set())[0] == "templates"


def test_parse_compact_with_instructions():
    assert karen_cli.parse_command("/compact", set()) == ("compact", None, "")
    assert karen_cli.parse_command("/compact focus on tests", set()) == ("compact", None, "focus on tests")


def test_parse_template_invocation():
    assert karen_cli.parse_command("/review a.py b.py", {"review"}) == ("template", "review", "a.py b.py")
    # unknown slash commands are sent to the model verbatim
    assert karen_cli.parse_command("/nope x", {"review"}) == ("prompt", None, "/nope x")


# ---------------------------------------------------------------------------
# format_args_preview
# ---------------------------------------------------------------------------


def test_format_args_preview():
    assert karen_cli.format_args_preview({"path": "a.py", "limit": 5}) == "path='a.py', limit=5"
    assert "\\n" in karen_cli.format_args_preview({"command": "a\nb"})
    long = karen_cli.format_args_preview({"content": "x" * 500})
    assert len(long) <= 72 and long.endswith("…")


# ---------------------------------------------------------------------------
# CLI hooks (constructed offline)
# ---------------------------------------------------------------------------


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setattr(karen_cli, "SESSIONS_ROOT", tmp_path / "sessions")
    return karen_cli.KarenCli(cwd=str(tmp_path), model_id="deepseek-v4-pro", fresh=True)


def test_system_message_seeds_prompt_and_tools(cli):
    message = cli._system_message()
    assert message.role == "system"
    assert "coding assistant" in message.content
    assert {t.name for t in message.tools_added} == {"read", "write", "edit", "bash"}


async def test_path_guard_blocks_writes_outside_cwd(cli, tmp_path):
    result = await cli.hooks.run(
        "before_tool",
        BeforeToolEvent(tool_call_id="t", tool_name="write", args={"path": "../escape.txt"}),
    )
    assert result.block is not None
    assert "escapes the working directory" in result.block.reason

    ok_result = await cli.hooks.run(
        "before_tool",
        BeforeToolEvent(tool_call_id="t", tool_name="write", args={"path": "inside.txt"}),
    )
    assert ok_result.block is None


async def test_before_tool_call_bridge_maps_block(cli):
    hook_context = types.SimpleNamespace(
        tool_call=types.SimpleNamespace(id="t1", name="edit"),
        args={"path": "../../outside.txt"},
    )
    result = await cli._before_tool_call(hook_context, None)
    assert result.block is True
    assert result.reason

    allowed = types.SimpleNamespace(
        tool_call=types.SimpleNamespace(id="t2", name="read"),
        args={"path": "anything.txt"},
    )
    assert await cli._before_tool_call(allowed, None) is None


async def test_after_tool_counter(cli):
    for _ in range(2):
        await cli.hooks.run(
            "after_tool",
            AfterToolEvent(tool_call_id="t", tool_name="read", args={}, content=[], is_error=False),
        )
    await cli.hooks.run(
        "after_tool",
        AfterToolEvent(tool_call_id="t", tool_name="bash", args={}, content=[], is_error=False),
    )
    assert cli.tool_counts == {"read": 2, "bash": 1}


def test_templates_loaded_from_project_dir(tmp_path, monkeypatch):
    prompts = tmp_path / ".karen" / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "review.md").write_text("---\ndescription: Review a file\n---\nReview $1", encoding="utf-8")
    monkeypatch.setattr(karen_cli, "SESSIONS_ROOT", tmp_path / "sessions")
    cli = karen_cli.KarenCli(cwd=str(tmp_path), model_id="deepseek-v4-pro", fresh=True)
    assert [t.name for t in cli.templates] == ["review"]
    assert cli.templates[0].description == "Review a file"


# ---------------------------------------------------------------------------
# overflow recovery (M7)
# ---------------------------------------------------------------------------


def _assistant(cli, *, stop_reason="stop", error_message=None, usage=None, model=None, provider="deepseek"):
    return AssistantMessage(
        content=[TextContent(text="answer")] if stop_reason == "stop" else [],
        api=cli.model.api,
        provider=provider,
        model=model or cli.model.id,
        usage=usage or Usage(),
        stop_reason=stop_reason,
        error_message=error_message,
        timestamp=2,
    )


def test_overflow_action_error_same_model_retries(cli):
    message = _assistant(cli, stop_reason="error", error_message="400 prompt is too long: 1200000 tokens")
    assert cli._overflow_action(message) == "retry"


def test_overflow_action_skips_different_model(cli):
    message = _assistant(
        cli, stop_reason="error", error_message="prompt is too long", model="deepseek-v3-older"
    )
    assert cli._overflow_action(message) is None


def test_overflow_action_ignores_non_overflow_errors(cli):
    message = _assistant(cli, stop_reason="error", error_message="500 internal server error")
    assert cli._overflow_action(message) is None


def test_overflow_action_ignores_aborted(cli):
    message = _assistant(cli, stop_reason="aborted", error_message="prompt is too long")
    assert cli._overflow_action(message) is None


def test_overflow_action_silent_overflow_compacts_without_retry(cli):
    window = cli.model.context_window
    message = _assistant(
        cli, stop_reason="stop", usage=Usage(input=window + 1, output=10, total_tokens=window + 11)
    )
    assert cli._overflow_action(message) == "compact_only"


def test_overflow_action_recoverable_length_retries(cli):
    message = _assistant(cli, stop_reason="length", usage=Usage(input=100, output=16, total_tokens=116))
    assert cli._overflow_action(message) == "retry"


def test_overflow_action_ignores_non_assistant(cli):
    assert cli._overflow_action(UserMessage(content="hi", timestamp=1)) is None


def test_omit_final_attempt_drops_last_assistant_turn(cli):
    user = UserMessage(content="hi", timestamp=1)
    first = _assistant(cli)
    tool_result = types.SimpleNamespace(role="toolResult")
    failed = _assistant(cli, stop_reason="error", error_message="prompt is too long")
    assert cli._omit_final_attempt([user, first, tool_result, failed]) == [user, first, tool_result]
    assert cli._omit_final_attempt([user, failed]) == [user]
    # the failed attempt's own tool results go with it
    assert cli._omit_final_attempt([user, failed, tool_result]) == [user]


class _FakeStream:
    """Stand-in for the loop's EventStream: no events, scripted result messages."""

    def __init__(self, messages):
        self._messages = messages

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        return
        yield

    async def result(self):
        return self._messages


async def _session_message_roles(cli):
    branch = await cli.session.branch("main")
    entries = await branch.find_entries(BranchScan(order="oldestFirst"))
    return [entry.message for entry in entries if entry.type == "message"]


def _patch_streams(monkeypatch, first_messages, retry_messages, calls):
    def fake_loop(prompts, context, config, signal, stream_fn):
        calls["loop"] += 1
        return _FakeStream([*prompts, *first_messages])

    def fake_continue(context, config, signal, stream_fn):
        calls["continue"] += 1
        return _FakeStream(retry_messages)

    monkeypatch.setattr(karen_cli, "agent_loop", fake_loop)
    monkeypatch.setattr(karen_cli, "agent_loop_continue", fake_continue)


def _patch_compaction(monkeypatch, cli, record):
    async def fake_compact(reason, custom_instructions=None):
        record.append(reason)
        return True

    monkeypatch.setattr(cli, "run_compaction", fake_compact)


async def test_run_turn_overflow_omits_attempt_and_retries(cli, monkeypatch):
    await cli.open_session()
    overflow = _assistant(cli, stop_reason="error", error_message="400 prompt is too long: 1200000 tokens")
    recovered = _assistant(cli)
    calls = {"loop": 0, "continue": 0}
    _patch_streams(monkeypatch, [overflow], [recovered], calls)
    compactions = []
    _patch_compaction(monkeypatch, cli, compactions)

    await cli.run_turn("hello")

    assert calls == {"loop": 1, "continue": 1}
    assert compactions == ["overflow"]
    persisted = await _session_message_roles(cli)
    assert [m.role for m in persisted] == ["user", "assistant"]
    assert all(getattr(m, "stop_reason", None) != "error" for m in persisted)


async def test_run_turn_overflow_gives_up_after_one_retry(cli, monkeypatch, capsys):
    await cli.open_session()
    first = _assistant(cli, stop_reason="error", error_message="prompt is too long")
    second = _assistant(cli, stop_reason="error", error_message="prompt is too long")
    calls = {"loop": 0, "continue": 0}
    _patch_streams(monkeypatch, [first], [second], calls)
    compactions = []
    _patch_compaction(monkeypatch, cli, compactions)

    await cli.run_turn("hello")

    assert calls == {"loop": 1, "continue": 1}
    assert compactions == ["overflow"]  # no second compact-and-retry attempt
    assert "recovery failed after one compact-and-retry attempt" in capsys.readouterr().err
    persisted = await _session_message_roles(cli)
    # the second failure stays in the transcript; the first was omitted
    assert [getattr(m, "stop_reason", None) for m in persisted] == [None, "error"]


async def test_run_turn_silent_overflow_compacts_without_retry(cli, monkeypatch):
    await cli.open_session()
    window = cli.model.context_window
    message = _assistant(
        cli, stop_reason="stop", usage=Usage(input=window + 1, output=10, total_tokens=window + 11)
    )
    calls = {"loop": 0, "continue": 0}
    _patch_streams(monkeypatch, [message], [], calls)
    compactions = []
    _patch_compaction(monkeypatch, cli, compactions)
    # the compaction stub doesn't rewrite the context, so the still-huge usage
    # estimate would chain a threshold compaction; isolate the overflow wiring
    async def no_auto_compact():
        return None

    monkeypatch.setattr(cli, "maybe_auto_compact", no_auto_compact)

    await cli.run_turn("hello")

    assert calls == {"loop": 1, "continue": 0}
    assert compactions == ["overflow"]
    persisted = await _session_message_roles(cli)
    # the completed response is preserved
    assert [m.role for m in persisted] == ["user", "assistant"]


async def test_run_turn_no_overflow_skips_overflow_compaction(cli, monkeypatch):
    await cli.open_session()
    calls = {"loop": 0, "continue": 0}
    _patch_streams(monkeypatch, [_assistant(cli)], [], calls)
    compactions = []
    _patch_compaction(monkeypatch, cli, compactions)

    await cli.run_turn("hello")

    assert calls == {"loop": 1, "continue": 0}
    assert compactions == []

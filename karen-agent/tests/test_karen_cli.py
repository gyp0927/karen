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

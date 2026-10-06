"""Tests for the karen CLI's pure logic, a piped REPL run, and the headless
print/JSON modes — all driven by the scripted faux provider, no network."""

import io
import json
import sys

import pytest

from karen_ai import create_models
from karen_ai.providers import faux_assistant_message, faux_tool_call, register_faux_provider
from karen_coding_agent import cli as karen_cli


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


def test_parse_skill_invocation():
    assert karen_cli.parse_command("/greet now", set(), {"greet"}) == ("skill", "greet", "now")
    assert karen_cli.parse_command("/greet", set(), {"greet"}) == ("skill", "greet", "")
    assert karen_cli.parse_command("/skills", set(), {"greet"})[0] == "skills"
    # templates and skills share one namespace; templates win
    assert karen_cli.parse_command("/greet", {"greet"}, {"greet"})[0] == "template"


# ---------------------------------------------------------------------------
# format_args_preview
# ---------------------------------------------------------------------------


def test_format_args_preview():
    assert karen_cli.format_args_preview({"path": "a.py", "limit": 5}) == "path='a.py', limit=5"
    assert "\\n" in karen_cli.format_args_preview({"command": "a\nb"})
    long = karen_cli.format_args_preview({"content": "x" * 500})
    assert len(long) <= 72 and long.endswith("…")


# ---------------------------------------------------------------------------
# piped REPL, offline (faux provider)
# ---------------------------------------------------------------------------


@pytest.fixture
def faux_models():
    def build():
        models = create_models()
        registration = register_faux_provider(responses=[faux_assistant_message("pong")])
        models.set_provider(registration.provider)
        return models, "faux"

    return build


def test_piped_repl_round_trip(tmp_path, monkeypatch, capsys, faux_models):
    monkeypatch.setattr(karen_cli, "build_models", faux_models)
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello\n/quit\n"))

    exit_code = karen_cli.main(["--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "pong" in out
    assert "new session" in out
    assert "[context ~" in out


def test_print_mode_prints_final_reply_only(tmp_path, monkeypatch, capsys, faux_models):
    monkeypatch.setattr(karen_cli, "build_models", faux_models)
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))

    exit_code = karen_cli.main(
        ["-p", "ping", "--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "pong\n"  # pi: stdout carries the final message text only
    assert "new session" in captured.err  # chatter goes to stderr in headless mode


def _faux_factory(responses):
    def build():
        models = create_models()
        registration = register_faux_provider(responses=responses)
        models.set_provider(registration.provider)
        return models, "faux"

    return build


def test_print_mode_runs_multiple_prompts_sequentially(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        karen_cli,
        "build_models",
        _faux_factory([faux_assistant_message("first"), faux_assistant_message("second")]),
    )
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))

    exit_code = karen_cli.main(
        ["-p", "one", "two", "--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == "second\n"  # the last reply is the final message


def test_print_mode_error_reply_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        karen_cli,
        "build_models",
        _faux_factory([faux_assistant_message("", stop_reason="error", error_message="boom")]),
    )
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))

    exit_code = karen_cli.main(
        ["-p", "ping", "--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "boom" in captured.err


def test_json_mode_emits_header_and_event_stream(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        karen_cli,
        "build_models",
        _faux_factory(
            [
                faux_assistant_message([faux_tool_call("find", {"pattern": "*.py"})]),
                faux_assistant_message("done"),
            ]
        ),
    )
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))

    exit_code = karen_cli.main(
        ["--mode", "json", "find files", "--new", "--cwd", str(tmp_path),
         "--provider", "faux", "--model", "faux-1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    lines = [json.loads(line) for line in captured.out.splitlines()]
    assert lines[0]["kind"] == "header"
    assert lines[0]["v"] == 4
    events = lines[1:]
    assert all("type" in event for event in events)
    updates = [e for e in events if e["type"] == "message_update"]
    assert updates, "expected streaming message_update events"
    for update in updates:
        assert "message" not in update
        assert "usage" in update
        assert "partial" not in update["assistantMessageEvent"]
    tool_starts = [
        u for u in updates if u["assistantMessageEvent"]["type"] == "toolcall_start"
    ]
    assert tool_starts and tool_starts[0]["assistantMessageEvent"]["toolName"] == "find"
    assert tool_starts[0]["assistantMessageEvent"]["id"]
    tool_execs = [e for e in events if e["type"] == "tool_execution_start"]
    assert tool_execs and tool_execs[0]["toolName"] == "find"
    assert events[-1]["type"] == "agent_end"


def test_json_mode_without_prompt_emits_header_only(tmp_path, monkeypatch, capsys, faux_models):
    monkeypatch.setattr(karen_cli, "build_models", faux_models)
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))

    exit_code = karen_cli.main(
        ["--mode", "json", "--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    lines = [json.loads(line) for line in captured.out.splitlines()]
    assert [line.get("kind") for line in lines] == ["header"]


def test_unknown_model_exits(tmp_path, monkeypatch, faux_models):
    monkeypatch.setattr(karen_cli, "build_models", faux_models)
    with pytest.raises(SystemExit):
        karen_cli.KarenCli(cwd=str(tmp_path), model_id="nope", fresh=True, provider="faux")


# ---------------------------------------------------------------------------
# system prompt assembly (M6)
# ---------------------------------------------------------------------------


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _capturing_factory(captured, reply="pong"):
    """A faux response step that records the messages the model was called with."""

    def factory(context, options, state, model):
        captured.append(list(context.messages))
        return faux_assistant_message(reply)

    return factory


def test_print_mode_prompt_carries_context_files_and_skills(tmp_path, monkeypatch, capsys):
    cwd = tmp_path / "proj"
    _write(cwd / "AGENTS.md", "Always run pytest before answering.")
    _write(
        cwd / ".karen" / "skills" / "greet" / "SKILL.md",
        "---\nname: greet\ndescription: Say hi warmly\n---\n\nGreet the user.",
    )
    monkeypatch.setattr(karen_cli, "DEFAULT_AGENT_DIR", tmp_path / "agent")
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    captured = []
    monkeypatch.setattr(karen_cli, "build_models", _faux_factory([_capturing_factory(captured)]))

    exit_code = karen_cli.main(
        ["-p", "hi", "--new", "--cwd", str(cwd), "--provider", "faux", "--model", "faux-1"]
    )

    assert exit_code == 0 and captured
    system = captured[0][0]
    assert system.role == "system"
    assert system.content == ""  # the prompt lives in sections, like pi
    assert "You are an expert coding assistant operating inside karen" in system.sections["preamble"]
    assert "Always run pytest before answering." in system.sections["project_context"]
    assert "<name>greet</name>" in system.sections["skills"]
    assert system.sections["cwd"] == f"<cwd>\n{cwd.as_posix()}\n</cwd>"
    tool_names = [tool.name for tool in system.tools_added]
    assert {"read", "bash", "edit", "write", "grep", "find", "ls"} <= set(tool_names)
    assert "- read: Read file contents" in system.sections["tools"]


def test_system_md_replaces_preamble_and_append_system_md_adds_addendum(tmp_path, monkeypatch, capsys):
    cwd = tmp_path / "proj"
    _write(cwd / ".karen" / "SYSTEM.md", "You are a terse test agent.")
    _write(cwd / ".karen" / "APPEND_SYSTEM.md", "Always answer in one line.")
    monkeypatch.setattr(karen_cli, "DEFAULT_AGENT_DIR", tmp_path / "agent")
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    captured = []
    monkeypatch.setattr(karen_cli, "build_models", _faux_factory([_capturing_factory(captured)]))

    assert karen_cli.main(["-p", "hi", "--new", "--cwd", str(cwd), "--provider", "faux", "--model", "faux-1"]) == 0

    sections = captured[0][0].sections
    assert sections["preamble"] == "You are a terse test agent."
    assert "tools" not in sections and "rules" not in sections
    assert sections["addendum"] == "<addendum>\nAlways answer in one line.\n</addendum>"


def test_settings_default_tools_limit_the_session(tmp_path, monkeypatch, capsys):
    cwd = tmp_path / "proj"
    _write(cwd / ".karen" / "settings.json", '{"defaultTools": ["read", "grep"]}')
    monkeypatch.setattr(karen_cli, "DEFAULT_AGENT_DIR", tmp_path / "agent")
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    captured = []
    monkeypatch.setattr(karen_cli, "build_models", _faux_factory([_capturing_factory(captured)]))

    assert karen_cli.main(["-p", "hi", "--new", "--cwd", str(cwd), "--provider", "faux", "--model", "faux-1"]) == 0

    system = captured[0][0]
    assert [tool.name for tool in system.tools_added] == ["read", "grep"]
    assert "- read: Read file contents" in system.sections["tools"]
    assert "- bash:" not in system.sections["tools"]


def test_repl_skill_listing_and_invocation(tmp_path, monkeypatch, capsys, faux_models):
    cwd = tmp_path / "proj"
    _write(
        cwd / ".karen" / "skills" / "greet" / "SKILL.md",
        "---\nname: greet\ndescription: Say hi warmly\n---\n\nGreet the user.",
    )
    monkeypatch.setattr(karen_cli, "DEFAULT_AGENT_DIR", tmp_path / "agent")
    monkeypatch.setattr(karen_cli, "build_models", faux_models)
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setattr(sys, "stdin", io.StringIO("/skills\n/greet be nice\n/quit\n"))

    exit_code = karen_cli.main(["--new", "--cwd", str(cwd), "--provider", "faux", "--model", "faux-1"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "/greet  Say hi warmly" in out
    # the skill block reached the model: it is persisted as the user message
    session_files = list((tmp_path / "sessions").rglob("*.jsonl"))
    assert len(session_files) == 1
    transcript = session_files[0].read_text(encoding="utf-8")
    assert 'skill name=' in transcript and "be nice" in transcript
    assert "Greet the user." in transcript

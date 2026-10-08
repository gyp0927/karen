"""Tests for the karen CLI's pure logic, a piped REPL run, and the headless
print/JSON modes — all driven by the scripted faux provider, no network."""

import io
import json
import os
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


def test_parse_navigation_commands():
    assert karen_cli.parse_command("/tree", set()) == ("tree", None, "")
    assert karen_cli.parse_command("/tree --summarize ab12", set()) == ("tree", None, "--summarize ab12")
    assert karen_cli.parse_command("/fork", set()) == ("fork", None, "")
    assert karen_cli.parse_command("/fork 2", set()) == ("fork", None, "2")
    assert karen_cli.parse_command("/clone", set()) == ("clone", None, "")
    assert karen_cli.parse_command("/sessions", set()) == ("sessions", None, "")
    assert karen_cli.parse_command("/resume abc123", set()) == ("resume", None, "abc123")
    assert karen_cli.parse_command("/name my task", set()) == ("name", None, "my task")
    assert karen_cli.parse_command("/session", set()) == ("session", None, "")


def test_parse_skill_invocation():
    assert karen_cli.parse_command("/greet now", set(), {"greet"}) == ("skill", "greet", "now")
    assert karen_cli.parse_command("/greet", set(), {"greet"}) == ("skill", "greet", "")
    assert karen_cli.parse_command("/skills", set(), {"greet"})[0] == "skills"
    # templates and skills share one namespace; templates win
    assert karen_cli.parse_command("/greet", {"greet"}, {"greet"})[0] == "template"


def test_parse_retry_command():
    assert karen_cli.parse_command("/retry", set()) == ("retry", None, "")
    assert karen_cli.parse_command("/retry off", set()) == ("retry", None, "off")
    assert karen_cli.parse_command("/retry on", set()) == ("retry", None, "on")


# ---------------------------------------------------------------------------
# format_args_preview
# ---------------------------------------------------------------------------


def test_format_args_preview():
    assert karen_cli.format_args_preview({"path": "a.py", "limit": 5}) == "path='a.py', limit=5"
    assert "\\n" in karen_cli.format_args_preview({"command": "a\nb"})
    long = karen_cli.format_args_preview({"content": "x" * 500})
    assert len(long) <= 72 and long.endswith("…")


# ---------------------------------------------------------------------------
# interactive-mode resolution (auto TUI)
# ---------------------------------------------------------------------------


def _resolve(**kwargs):
    arguments = {"tui_flag": False, "repl_flag": False, "setting": None, "interactive": True}
    arguments.update(kwargs)
    return karen_cli.resolve_interactive_mode(**arguments)


def test_the_tui_is_the_default_on_a_terminal():
    assert _resolve() == (True, "")


def test_no_terminal_falls_back_to_the_repl_with_a_notice():
    use_tui, notice = _resolve(interactive=False)
    assert use_tui is False
    assert "plain REPL" in notice


def test_explicit_flags_outrank_the_setting():
    assert _resolve(tui_flag=True, setting=False) == (True, "")
    assert _resolve(repl_flag=True, setting=True) == (False, "")


def test_the_setting_outranks_the_default():
    assert _resolve(setting=False) == (False, "")
    assert _resolve(setting=True) == (True, "")


def test_an_explicit_repl_choice_is_never_explained():
    # the user asked for the plain REPL (or the setting did): there is nothing
    # to warn about, even without a terminal
    assert _resolve(repl_flag=True, interactive=False) == (False, "")
    assert _resolve(setting=False, interactive=False) == (False, "")


def test_an_explicit_tui_without_a_terminal_still_explains_itself():
    use_tui, notice = _resolve(tui_flag=True, interactive=False)
    assert use_tui is False
    assert "plain REPL" in notice


def test_tui_and_repl_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as exit_info:
        karen_cli.main(["--tui", "--repl"])
    assert exit_info.value.code == 2


def _recording_repl_tui(opened):
    """A `KarenCli.repl_tui` stand-in that records that it was entered.

    `main`'s choice would otherwise be unobservable: both it and `repl_tui`'s
    own guard print the same sentence and both end up in the plain REPL, so
    only "was `repl_tui` entered at all" pins the decision down.
    """

    async def repl_tui(self):
        opened.append(True)
        return 0

    return repl_tui


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


def test_main_opens_the_tui_on_a_terminal(tmp_path, monkeypatch, capsys, faux_models):
    """The headline behaviour: with a terminal and no flags, `karen` enters the
    alt-screen TUI (not merely "the helper returns True")."""
    monkeypatch.setattr(karen_cli, "build_models", faux_models)
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setenv("KAREN_SETTINGS_PATH", str(tmp_path / "no-global.json"))
    monkeypatch.setattr(sys, "stdin", io.StringIO("/quit\n"))
    monkeypatch.setattr(karen_cli, "supports_tty", lambda: True)
    opened = []
    monkeypatch.setattr(karen_cli.KarenCli, "repl_tui", _recording_repl_tui(opened))

    exit_code = karen_cli.main(
        ["--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"]
    )

    assert exit_code == 0
    assert opened == [True]
    assert "plain REPL" not in capsys.readouterr().err


def test_the_tui_is_not_started_without_a_terminal(tmp_path, monkeypatch, capsys, faux_models):
    monkeypatch.setattr(karen_cli, "build_models", faux_models)
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setenv("KAREN_SETTINGS_PATH", str(tmp_path / "no-global.json"))
    monkeypatch.setattr(sys, "stdin", io.StringIO("/quit\n"))
    monkeypatch.setattr(karen_cli, "supports_tty", lambda: False)
    opened = []
    monkeypatch.setattr(karen_cli.KarenCli, "repl_tui", _recording_repl_tui(opened))

    exit_code = karen_cli.main(
        ["--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "plain REPL" in captured.err  # the auto-TUI explains its fallback
    assert "type /help for commands" in captured.out  # the plain REPL ran
    assert opened == []  # main decided, not repl_tui's own guard


def test_a_tui_false_setting_keeps_the_repl_without_a_notice(tmp_path, monkeypatch, capsys, faux_models):
    monkeypatch.setattr(karen_cli, "build_models", faux_models)
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setenv("KAREN_SETTINGS_PATH", str(tmp_path / "no-global.json"))
    monkeypatch.setattr(sys, "stdin", io.StringIO("/quit\n"))
    # a terminal is available, so only the setting can keep the TUI away
    monkeypatch.setattr(karen_cli, "supports_tty", lambda: True)
    opened = []
    monkeypatch.setattr(karen_cli.KarenCli, "repl_tui", _recording_repl_tui(opened))
    project = tmp_path / ".karen" / "settings.json"
    project.parent.mkdir(parents=True, exist_ok=True)
    project.write_text(json.dumps({"tui": False}), encoding="utf-8")

    exit_code = karen_cli.main(
        ["--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert opened == []
    assert "plain REPL" not in captured.err  # nothing to explain: it was asked for
    assert "type /help for commands" in captured.out  # the plain REPL really ran


def test_the_repl_flag_is_silent_about_the_tui(tmp_path, monkeypatch, capsys, faux_models):
    monkeypatch.setattr(karen_cli, "build_models", faux_models)
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setenv("KAREN_SETTINGS_PATH", str(tmp_path / "no-global.json"))
    monkeypatch.setattr(sys, "stdin", io.StringIO("/quit\n"))
    monkeypatch.setattr(karen_cli, "supports_tty", lambda: True)
    opened = []
    monkeypatch.setattr(karen_cli.KarenCli, "repl_tui", _recording_repl_tui(opened))

    exit_code = karen_cli.main(
        ["--repl", "--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert opened == []
    assert "plain REPL" not in captured.err
    assert "type /help for commands" in captured.out  # the plain REPL ran


def test_the_tui_flag_beats_a_tui_false_setting(tmp_path, monkeypatch, capsys, faux_models):
    monkeypatch.setattr(karen_cli, "build_models", faux_models)
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setenv("KAREN_SETTINGS_PATH", str(tmp_path / "no-global.json"))
    monkeypatch.setattr(sys, "stdin", io.StringIO("/quit\n"))
    monkeypatch.setattr(karen_cli, "supports_tty", lambda: True)
    opened = []
    monkeypatch.setattr(karen_cli.KarenCli, "repl_tui", _recording_repl_tui(opened))
    project = tmp_path / ".karen" / "settings.json"
    project.parent.mkdir(parents=True, exist_ok=True)
    project.write_text(json.dumps({"tui": False}), encoding="utf-8")

    exit_code = karen_cli.main(
        ["--tui", "--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"]
    )

    assert exit_code == 0
    assert opened == [True]


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


def _write_settings(tmp_path, retry):
    directory = tmp_path / ".karen"
    directory.mkdir(exist_ok=True)
    (directory / "settings.json").write_text(json.dumps({"retry": retry}), encoding="utf-8")


def test_print_mode_auto_retries_a_transient_failure(tmp_path, monkeypatch, capsys):
    _write_settings(tmp_path, {"baseDelayMs": 1, "maxAgentDelayMs": 1})
    monkeypatch.setattr(
        karen_cli,
        "build_models",
        _faux_factory(
            [
                faux_assistant_message([], stop_reason="error", error_message="Error 503 Service Unavailable"),
                faux_assistant_message("recovered"),
            ]
        ),
    )
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))

    exit_code = karen_cli.main(
        ["-p", "ping", "--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "recovered\n"
    assert "[retrying (attempt 1/3)" in captured.err
    assert "[retry succeeded on attempt 1]" in captured.err
    # the failing run is not reported as a failure
    assert "[run failed" not in captured.err


def test_retry_settings_disable_auto_retry(tmp_path, monkeypatch, capsys):
    _write_settings(tmp_path, {"enabled": False})
    monkeypatch.setattr(
        karen_cli,
        "build_models",
        _faux_factory(
            [
                faux_assistant_message([], stop_reason="error", error_message="Error 503 Service Unavailable"),
                faux_assistant_message("never used"),
            ]
        ),
    )
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))

    exit_code = karen_cli.main(
        ["-p", "ping", "--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "[retrying" not in captured.err
    assert captured.out == ""
    assert "Error 503 Service Unavailable" in captured.err  # print mode reports the failure itself


def test_repl_retry_command_toggles_and_reports(tmp_path, monkeypatch, capsys, faux_models):
    monkeypatch.setattr(karen_cli, "build_models", faux_models)
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setattr(sys, "stdin", io.StringIO("/retry\n/retry off\n/retry on\n/retry maybe\n/quit\n"))

    exit_code = karen_cli.main(["--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "auto-retry: on (maxRetries 3, baseDelayMs 2000)" in captured.out
    assert "auto-retry: off (maxRetries 3, baseDelayMs 2000)" in captured.out
    assert captured.out.count("auto-retry: on") == 2  # initial report and the toggle back
    assert "usage: /retry [on|off]" in captured.err


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


# ---------------------------------------------------------------------------
# session navigation in the REPL (M7)
# ---------------------------------------------------------------------------


async def _nav_cli(tmp_path, monkeypatch, responses):
    monkeypatch.setattr(karen_cli, "build_models", _faux_factory(responses))
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    cli = karen_cli.KarenCli(cwd=str(tmp_path), model_id="faux-1", fresh=True, provider="faux")
    await cli._open_session(fresh=True)
    return cli


async def test_cli_tree_command_renders_and_navigates(tmp_path, monkeypatch, capsys):
    cli = await _nav_cli(
        tmp_path,
        monkeypatch,
        [faux_assistant_message("first reply"), faux_assistant_message("second reply")],
    )
    await cli.session.prompt("first question")
    await cli.session.prompt("second question")
    entries = await cli.session.entries()
    capsys.readouterr()

    await cli._handle_tree("")

    listed = capsys.readouterr().out
    assert "user: first question" in listed
    assert "assistant: second reply" in listed
    assert "● current tip" in listed

    await cli._handle_tree(entries[2].id[-8:])

    after = capsys.readouterr().out
    assert "second question" in after  # the message came back for editing
    assert await cli.session.branch_tip_id() == entries[1].id
    await cli.session.close()


async def test_cli_tree_reports_bad_ids(tmp_path, monkeypatch, capsys):
    cli = await _nav_cli(
        tmp_path,
        monkeypatch,
        [faux_assistant_message("first reply"), faux_assistant_message("second reply")],
    )
    await cli.session.prompt("first question")
    await cli.session.prompt("second question")
    ids = [entry.id for entry in await cli.session.entries()]
    shared = os.path.commonprefix(ids)
    capsys.readouterr()

    with pytest.raises(ValueError, match="no entry matches"):
        await cli._handle_tree("zzzzzzzz")

    assert len(shared) >= 1
    with pytest.raises(ValueError, match="ambiguous entry prefix"):
        await cli._handle_tree(shared)

    # the tip never moved
    assert await cli.session.branch_tip_id() == ids[-1]
    await cli.session.close()


def test_repl_keeps_running_after_a_bad_tree_id(tmp_path, monkeypatch, capsys, faux_models):
    monkeypatch.setattr(karen_cli, "build_models", faux_models)
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setattr(sys, "stdin", io.StringIO("question\n/tree zzzzzzzz\n/quit\n"))

    exit_code = karen_cli.main(["--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "pong" in captured.out
    assert "[error: no entry matches" in captured.err


async def test_cli_resolve_entry_reports_bad_prefixes(tmp_path, monkeypatch):
    cli = await _nav_cli(tmp_path, monkeypatch, [faux_assistant_message("reply")])
    await cli.session.prompt("question")

    with pytest.raises(ValueError, match="no entry matches"):
        await cli._resolve_entry("zzzzzzzz")

    entry = (await cli.session.entries())[0]
    assert await cli._resolve_entry(entry.id[-8:]) == entry.id
    assert await cli._resolve_entry(entry.id) == entry.id
    await cli.session.close()


async def test_cli_fork_lists_targets_and_forks(tmp_path, monkeypatch, capsys):
    cli = await _nav_cli(
        tmp_path,
        monkeypatch,
        [faux_assistant_message("first reply"), faux_assistant_message("second reply")],
    )
    await cli.session.prompt("first question")
    await cli.session.prompt("second question")
    original_id = cli.session.session.metadata.id
    capsys.readouterr()

    await cli._handle_fork("")

    listed = capsys.readouterr().out
    assert "1. [" in listed and "first question" in listed
    assert "fork from one of these with /fork <n|id>" in listed

    await cli._handle_fork("2")

    after = capsys.readouterr().out
    assert "forked into session" in after
    assert "second question" in after  # pi hands the message back for editing
    assert cli.session.session.metadata.id != original_id
    assert len(await cli.session.entries()) == 2
    await cli.session.close()


async def test_cli_clone_and_resume_round_trip(tmp_path, monkeypatch, capsys):
    cli = await _nav_cli(tmp_path, monkeypatch, [faux_assistant_message("reply")])
    await cli.session.prompt("question")
    original_id = cli.session.session.metadata.id
    capsys.readouterr()

    await cli._handle_clone()

    cloned_id = cli.session.session.metadata.id
    assert cloned_id != original_id
    assert "cloned into session" in capsys.readouterr().out

    await cli._handle_sessions("")

    listed = capsys.readouterr().out
    assert "switch with /resume <n|id>" in listed
    assert f"* 1. {cloned_id[-8:]}" in listed  # newest first, current marked
    assert original_id[-8:] in listed

    await cli._handle_sessions("2")

    after = capsys.readouterr().out
    assert f"switched to session {original_id}" in after
    assert cli.session.session.metadata.id == original_id
    await cli.session.close()


async def test_cli_clone_without_entries_hints(tmp_path, monkeypatch, capsys):
    cli = await _nav_cli(tmp_path, monkeypatch, [])
    capsys.readouterr()

    await cli._handle_clone()

    assert "nothing to clone yet" in capsys.readouterr().out
    await cli.session.close()


async def test_cli_name_and_session_info(tmp_path, monkeypatch, capsys):
    cli = await _nav_cli(tmp_path, monkeypatch, [faux_assistant_message("reply")])
    await cli.session.prompt("question")
    capsys.readouterr()

    await cli._handle_name("")

    assert "session name: (unnamed)" in capsys.readouterr().out

    await cli._handle_name("release prep")

    assert "session name: release prep" in capsys.readouterr().out

    await cli._print_session_info()

    info = capsys.readouterr().out
    assert f"id: {cli.session.session.metadata.id}" in info
    assert "name: release prep" in info
    assert "messages: 2 (1 user, 1 assistant, 0 tool calls, 0 tool results)" in info
    assert "context: ~" in info
    await cli.session.close()


# ---------------------------------------------------------------------------
# images settings wiring + REPL resilience (M9 follow-ups)
# ---------------------------------------------------------------------------


async def test_cli_applies_image_settings_to_the_session(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"images": {"autoResize": False, "blockImages": True}}), encoding="utf-8"
    )
    monkeypatch.setenv("KAREN_SETTINGS_PATH", str(settings_path))
    cli = await _nav_cli(tmp_path, monkeypatch, [])
    assert cli.session.auto_resize_images is False
    assert cli.session.block_images is True
    await cli.session.close()


async def test_cli_settings_command_applies_images_live(tmp_path, monkeypatch, capsys):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setenv("KAREN_SETTINGS_PATH", str(settings_path))
    cli = await _nav_cli(tmp_path, monkeypatch, [])
    assert cli.session.block_images is False
    capsys.readouterr()

    cli._handle_settings('global images={"blockImages":true}')

    assert "updated" in capsys.readouterr().out
    assert cli.session.block_images is True  # takes effect without a restart
    await cli.session.close()


async def test_cli_settings_write_survives_the_next_new(tmp_path, monkeypatch, capsys):
    """pi reads its settings manager on every use, so a `/settings` write is not
    a property of the open session: `/new` must rebuild from the file."""
    import base64
    import io

    from PIL import Image

    settings_path = tmp_path / "settings.json"
    monkeypatch.setenv("KAREN_SETTINGS_PATH", str(settings_path))
    cli = await _nav_cli(tmp_path, monkeypatch, [])
    capsys.readouterr()

    cli._handle_settings('global images={"autoResize":false,"blockImages":true}')
    capsys.readouterr()

    await cli._open_session(fresh=True)  # the REPL's `/new`

    assert karen_cli.image_auto_resize(cli.settings.images) is False
    assert cli.session.auto_resize_images is False
    assert cli.session.block_images is True

    # and the rebuilt tool set honours it: with auto-resize off the read tool
    # hands the image through untouched
    buffer = io.BytesIO()
    Image.new("RGB", (4000, 3000), (10, 20, 30)).save(buffer, "PNG")
    payload = buffer.getvalue()
    (tmp_path / "big.png").write_bytes(payload)
    read_tool = next(tool for tool in cli.session.tools if tool.name == "read")
    result = await read_tool.execute("call-1", {"path": "big.png"}, None, None)
    image = next(block for block in result.content if getattr(block, "type", None) == "image")
    assert image.data == base64.b64encode(payload).decode()
    await cli.session.close()


def test_piped_repl_survives_a_broken_settings_file(tmp_path, monkeypatch, capsys):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("KAREN_SETTINGS_PATH", str(settings_path))
    monkeypatch.setattr(karen_cli, "build_models", _faux_factory([faux_assistant_message("pong")]))
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setattr(sys, "stdin", io.StringIO("/settings global defaultModel=x\n/quit\n"))

    exit_code = karen_cli.main(["--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "refusing to overwrite" in captured.err
    assert settings_path.read_text(encoding="utf-8") == "{not json"  # left alone


def test_piped_repl_survives_a_failed_export(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "blocker.txt"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(karen_cli, "build_models", _faux_factory([faux_assistant_message("pong")]))
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setattr(
        sys, "stdin", io.StringIO(f"/export html {blocker}/out.html\nhello\n/quit\n")
    )

    exit_code = karen_cli.main(["--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[error:" in captured.err
    assert "pong" in captured.out  # the session kept working after the failure


async def test_cli_read_tool_honors_image_settings(tmp_path, monkeypatch):
    """`images.autoResize=false` reaches the read tool's image processor."""
    import base64
    import io

    from PIL import Image

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"images": {"autoResize": False}}), encoding="utf-8")
    monkeypatch.setenv("KAREN_SETTINGS_PATH", str(settings_path))
    cli = await _nav_cli(tmp_path, monkeypatch, [])

    buffer = io.BytesIO()
    Image.new("RGB", (4000, 3000), (10, 20, 30)).save(buffer, "PNG")
    payload = buffer.getvalue()
    (tmp_path / "big.png").write_bytes(payload)

    read_tool = next(tool for tool in cli.session.tools if tool.name == "read")
    result = await read_tool.execute("call-1", {"path": "big.png"}, None, None)

    image = next(block for block in result.content if getattr(block, "type", None) == "image")
    assert image.data == base64.b64encode(payload).decode()  # untouched, not resized
    await cli.session.close()

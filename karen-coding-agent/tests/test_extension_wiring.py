"""Integration tests for the M10 extension wiring: how `cli.py` loads
extensions (conventional dirs, the settings `extensions` list, the
`--extension` flag), how the REPL/TUI route extension commands, how extension
tools reach the model, the session lifecycle events extensions observe, the
`resources_discover` merge, and RPC `new_session` rebinding — all offline,
driven by the scripted faux provider and helper extension modules written into
tmp dirs (the loader imports from real file paths)."""

import asyncio
import io
import json
import sys
from types import SimpleNamespace

import pytest

from karen_ai import create_models
from karen_ai.providers import faux_assistant_message, faux_tool_call, register_faux_provider
from karen_agent import Agent
from karen_coding_agent import cli as karen_cli
from karen_coding_agent import rpc as karen_rpc
from karen_coding_agent.extensions import Command, Extension, ExtensionHost, ExtensionRunner
from karen_coding_agent.rpc import RpcSession
from karen_coding_agent.tui.app import LiveTui


# ---------------------------------------------------------------------------
# helpers: filesystem isolation + a scripted faux provider
# ---------------------------------------------------------------------------


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _isolate(tmp_path, monkeypatch, responses=(), registrations=None):
    """Point the CLI at a scripted faux provider and a throwaway filesystem.

    `KAREN_SETTINGS_PATH` names a file that does not exist (no global settings
    bleed in) and `DEFAULT_AGENT_DIR` a tmp dir (no user-level extensions,
    skills or templates); `registrations`, when given, collects the faux
    registration so the test can count unconsumed scripted responses.
    """

    def build():
        models = create_models()
        registration = register_faux_provider(responses=list(responses))
        models.set_provider(registration.provider)
        if registrations is not None:
            registrations.append(registration)
        return models, "faux"

    monkeypatch.setattr(karen_cli, "build_models", build)
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setenv("KAREN_SETTINGS_PATH", str(tmp_path / "no-global-settings.json"))
    monkeypatch.setattr(karen_cli, "DEFAULT_AGENT_DIR", tmp_path / "agent")


def _run_repl(tmp_path, monkeypatch, stdin_text, extra_args=()):
    """Drive `karen_cli.main` synchronously with a piped stdin, like test_cli."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
    return karen_cli.main(
        ["--new", "--cwd", str(tmp_path), "--provider", "faux", "--model", "faux-1", *extra_args]
    )


def _recorded(record_path):
    """The JSON lines an extension appended to its record file, in order."""
    if not record_path.exists():
        return []
    return [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# helpers: extension modules (imported by the loader from real file paths)
# ---------------------------------------------------------------------------


def _recording_extension(record_path, command="hello", notify="hi"):
    """A `/command` extension: records its args, then `ctx.notify`s the user."""
    return (
        "import json\n"
        f"RECORD = {json.dumps(str(record_path))}\n"
        "def extension(pi):\n"
        "    def handler(args, ctx):\n"
        "        with open(RECORD, 'a', encoding='utf-8') as fh:\n"
        "            fh.write(json.dumps({'args': args}) + '\\n')\n"
        f"        ctx.notify({json.dumps(notify)})\n"
        f"    pi.register_command({json.dumps(command)}, handler, description='test command')\n"
    )


def _lifecycle_extension(record_path):
    """Records the session_start/session_shutdown events it observes."""
    return (
        "import json\n"
        f"RECORD = {json.dumps(str(record_path))}\n"
        "def extension(pi):\n"
        "    def make(name):\n"
        "        def handler(event, ctx):\n"
        "            with open(RECORD, 'a', encoding='utf-8') as fh:\n"
        "                fh.write(json.dumps({'event': name, 'reason': getattr(event, 'reason', None)}) + '\\n')\n"
        "        return handler\n"
        "    pi.on('session_start', make('session_start'))\n"
        "    pi.on('session_shutdown', make('session_shutdown'))\n"
    )


def _tool_extension(record_path, tool_name="mytool", output="tool-output-marker", extra_tool=None):
    """Registers one tool (plus `extra_tool`, a bare name, when given).

    Both registrations live in the same factory: a second `def extension(pi)`
    would just be shadowed by the first.
    """
    names = [tool_name] + ([extra_tool] if extra_tool else [])
    return (
        "import json\n"
        f"RECORD = {json.dumps(str(record_path))}\n"
        "def extension(pi):\n"
        "    def execute(tool_call_id, args, signal, on_update):\n"
        "        with open(RECORD, 'a', encoding='utf-8') as fh:\n"
        "            fh.write(json.dumps({'tool_call_id': tool_call_id, 'args': args}) + '\\n')\n"
        f"        return {json.dumps(output)}\n"
        + "".join(
            f"    pi.register_tool(name={json.dumps(name)}, description='record things',\n"
            "                     parameters={'type': 'object', 'properties': {'x': {'type': 'number'}}},\n"
            "                     execute=execute)\n"
            for name in names
        )
    )


def _resources_extension(prompts_dir, skills_dir):
    """Contributes prompt-template and skill directories via resources_discover."""
    return (
        "def extension(pi):\n"
        "    def discover(event, ctx):\n"
        f"        return {{'prompt_paths': [{json.dumps(str(prompts_dir))}],\n"
        f"                'skill_paths': [{json.dumps(str(skills_dir))}]}}\n"
        "    pi.on('resources_discover', discover)\n"
    )


# ---------------------------------------------------------------------------
# loading: conventional project directory
# ---------------------------------------------------------------------------


async def test_project_conventional_extension_loads_on_session_open(tmp_path, monkeypatch):
    record = tmp_path / "record.jsonl"
    _write(tmp_path / ".karen" / "extensions" / "greeter.py", _recording_extension(record))
    _isolate(tmp_path, monkeypatch, [faux_assistant_message("pong")])

    cli = karen_cli.KarenCli(cwd=str(tmp_path), model_id="faux-1", fresh=True, provider="faux")
    # loading is deferred to the first session open (factories may be async)
    assert cli.extension_runner is None
    assert cli.loaded_extensions is None

    await cli._open_session(fresh=True)

    assert cli.extension_runner is not None
    assert "hello" in cli.extension_runner.commands()
    loaded = cli.loaded_extensions
    assert len(loaded.extensions) == 1
    assert loaded.extensions[0].source == "project"
    assert loaded.diagnostics == []
    summary = cli._extension_summary_line()
    assert "extensions: 1 loaded" in summary and "/hello" in summary
    await cli.session.close()


def test_a_broken_extension_reports_a_diagnostic_without_killing_startup(tmp_path, monkeypatch, capsys):
    _write(tmp_path / ".karen" / "extensions" / "broken.py", "raise RuntimeError('boom on import')\n")
    _isolate(tmp_path, monkeypatch, [faux_assistant_message("pong")])

    exit_code = _run_repl(tmp_path, monkeypatch, "/quit\n")

    captured = capsys.readouterr()
    assert exit_code == 0
    # the notify sink of the plain REPL sends error-level diagnostics to stderr
    assert "[extension error:" in captured.err
    assert "boom on import" in captured.err
    assert "type /help for commands" in captured.out  # the REPL still started


# ---------------------------------------------------------------------------
# loading: settings `extensions` list and the --extension flag
# ---------------------------------------------------------------------------


def test_settings_extensions_list_loads(tmp_path, monkeypatch, capsys):
    extension_file = tmp_path / "extras" / "from_settings.py"
    _write(extension_file, _recording_extension(tmp_path / "record.jsonl", command="sethello"))
    _isolate(tmp_path, monkeypatch, [faux_assistant_message("pong")])
    settings_path = tmp_path / "global-settings.json"
    settings_path.write_text(json.dumps({"extensions": [str(extension_file)]}), encoding="utf-8")
    monkeypatch.setenv("KAREN_SETTINGS_PATH", str(settings_path))

    exit_code = _run_repl(tmp_path, monkeypatch, "/quit\n")

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "extensions: 1 loaded" in out
    assert "/sethello" in out


def test_the_extension_flag_loads(tmp_path, monkeypatch, capsys):
    extension_file = tmp_path / "extras" / "from_flag.py"
    _write(extension_file, _recording_extension(tmp_path / "record.jsonl", command="flaghello"))
    _isolate(tmp_path, monkeypatch, [faux_assistant_message("pong")])

    exit_code = _run_repl(tmp_path, monkeypatch, "/quit\n", ["--extension", str(extension_file)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "extensions: 1 loaded" in out
    assert "/flaghello" in out


def test_a_missing_extension_flag_path_is_a_startup_diagnostic(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "extras" / "nope.py"
    _isolate(tmp_path, monkeypatch, [faux_assistant_message("pong")])

    exit_code = _run_repl(tmp_path, monkeypatch, "/quit\n", ["--extension", str(missing)])

    captured = capsys.readouterr()
    assert exit_code == 0  # a missing path is reported, never fatal
    assert "[extension error:" in captured.err
    assert "extension path does not exist" in captured.err
    assert "type /help for commands" in captured.out


# ---------------------------------------------------------------------------
# REPL command dispatch
# ---------------------------------------------------------------------------


def test_repl_extension_command_runs_the_handler_not_the_model(tmp_path, monkeypatch, capsys):
    record = tmp_path / "record.jsonl"
    _write(tmp_path / ".karen" / "extensions" / "greeter.py", _recording_extension(record))
    registrations = []
    _isolate(tmp_path, monkeypatch, [faux_assistant_message("pong")], registrations)

    exit_code = _run_repl(tmp_path, monkeypatch, "/hello world\n/quit\n")

    out = capsys.readouterr().out
    assert exit_code == 0
    assert [entry["args"] for entry in _recorded(record)] == ["world"]
    # ctx.notify reached the user (printed right after the input prompt)
    assert "karen> hi" in out
    # the command line never became a run: the scripted response is unconsumed
    assert registrations[0].get_pending_response_count() == 1


def test_an_unknown_slash_line_goes_to_the_model_verbatim(tmp_path, monkeypatch, capsys):
    _write(tmp_path / ".karen" / "extensions" / "greeter.py",
           _recording_extension(tmp_path / "record.jsonl"))
    registrations = []
    _isolate(tmp_path, monkeypatch, [faux_assistant_message("pong")], registrations)

    exit_code = _run_repl(tmp_path, monkeypatch, "/whatever args\n/quit\n")

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "pong" in out
    assert registrations[0].get_pending_response_count() == 0  # a run happened
    session_files = list((tmp_path / "sessions").rglob("*.jsonl"))
    assert len(session_files) == 1
    # the line reached the model exactly as typed, slash and all
    assert "/whatever args" in session_files[0].read_text(encoding="utf-8")


def test_an_extension_command_shadows_a_same_named_template(tmp_path, monkeypatch, capsys):
    # the router itself: extension names sit between built-ins and templates
    assert karen_cli.parse_command("/hello x", {"hello"}, (), {"hello"}) == ("extension", "hello", "x")
    record = tmp_path / "record.jsonl"
    _write(tmp_path / ".karen" / "extensions" / "greeter.py", _recording_extension(record))
    _write(
        tmp_path / ".karen" / "prompts" / "hello.md",
        "---\ndescription: template hello\n---\n\nSay hello to $ARGUMENTS\n",
    )
    registrations = []
    _isolate(tmp_path, monkeypatch, [faux_assistant_message("pong")], registrations)

    exit_code = _run_repl(tmp_path, monkeypatch, "/templates\n/hello world\n/quit\n")

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "template hello" in out  # the template really was loaded...
    assert [entry["args"] for entry in _recorded(record)] == ["world"]  # ...yet the command won
    # a template invocation would have been sent to the model; nothing was
    assert registrations[0].get_pending_response_count() == 1


# ---------------------------------------------------------------------------
# extension tools reach the model
# ---------------------------------------------------------------------------


async def test_an_extension_tool_colliding_with_a_builtin_is_reported(tmp_path, monkeypatch):
    """A registration the built-in wins must be visible, not silently dropped.

    `.scratch/repro_ext_tool_collision.py` pinned the silent version: an
    extension registering `bash` simply never appeared in the tool set.
    """
    record = tmp_path / "record.jsonl"
    _write(tmp_path / ".karen" / "extensions" / "shadow.py",
           _tool_extension(record, tool_name="bash", output="shadow-output", extra_tool="uniqtool"))
    _isolate(tmp_path, monkeypatch, [])
    cli = karen_cli.KarenCli(cwd=str(tmp_path), model_id="faux-1", fresh=True, provider="faux")
    await cli._open_session(fresh=True)

    names = [tool.name for tool in cli.session.tools]
    assert names.count("bash") == 1  # the built-in kept its slot
    assert "uniqtool" in names  # the extension's other tool still landed
    messages = [error.message for error in cli.extension_runner.errors]
    assert any("'bash' was ignored" in message for message in messages), messages
    await cli.session.close()


async def test_extension_tools_reach_the_model_and_the_transcript(tmp_path, monkeypatch):
    record = tmp_path / "record.jsonl"
    _write(tmp_path / ".karen" / "extensions" / "toolbox.py", _tool_extension(record))
    registrations = []
    _isolate(
        tmp_path,
        monkeypatch,
        [faux_assistant_message([faux_tool_call("mytool", {"x": 1})]), faux_assistant_message("done")],
        registrations,
    )
    cli = karen_cli.KarenCli(cwd=str(tmp_path), model_id="faux-1", fresh=True, provider="faux")
    await cli._open_session(fresh=True)

    # the wrapped extension tool sits in the session's active tool set
    assert "mytool" in [tool.name for tool in cli.session.tools]

    await cli.session.prompt("use the tool")

    calls = _recorded(record)
    assert len(calls) == 1
    assert calls[0]["args"] == {"x": 1}  # the extension's own execute() ran
    assert registrations[0].get_pending_response_count() == 0  # both turns ran
    await cli.session.close()
    # and the tool's return value was persisted as the result the model saw
    session_files = list((tmp_path / "sessions").rglob("*.jsonl"))
    assert len(session_files) == 1
    assert "tool-output-marker" in session_files[0].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# session lifecycle events + runner sharing across `/new`
# ---------------------------------------------------------------------------


async def test_session_lifecycle_events_and_a_shared_runner_across_new(tmp_path, monkeypatch):
    record = tmp_path / "record.jsonl"
    _write(tmp_path / ".karen" / "extensions" / "lifecycle.py", _lifecycle_extension(record))
    _isolate(tmp_path, monkeypatch, [])
    cli = karen_cli.KarenCli(cwd=str(tmp_path), model_id="faux-1", fresh=True, provider="faux")

    await cli._open_session(fresh=True)
    runner = cli.extension_runner
    assert runner is not None

    await cli._open_session(fresh=True)  # the REPL's `/new`

    # the runner is re-bound to the fresh session, never reloaded
    assert cli.extension_runner is runner
    assert runner.host.session is cli.session

    await cli.session.close()

    assert [(entry["event"], entry["reason"]) for entry in _recorded(record)] == [
        ("session_start", "startup"),
        ("session_shutdown", "new"),
        ("session_start", "new"),
        ("session_shutdown", "quit"),
    ]


async def test_fork_pairs_a_shutdown_and_start_with_pi_reasons(tmp_path, monkeypatch):
    """The lifecycle pair around a session replacement carries pi's reasons.

    pi's `SessionShutdownEvent.reason` is `"quit" | "reload" | "new" | "resume"
    | "fork"`; a fork says `"fork"`, a switch to an existing session says
    `"resume"`. The reason the shutdown emits is the replacement's own, so a
    handler can pair the two events.
    """
    record = tmp_path / "record.jsonl"
    _write(tmp_path / ".karen" / "extensions" / "lifecycle.py", _lifecycle_extension(record))
    _isolate(tmp_path, monkeypatch, [])
    cli = karen_cli.KarenCli(cwd=str(tmp_path), model_id="faux-1", fresh=True, provider="faux")
    await cli._open_session(fresh=True)

    # a clone = fork with no entry: copies the whole branch and switches to it
    result = await cli.session.fork(None, position="at")
    assert result is not None  # the fork succeeded
    await cli.session.close()

    assert [(entry["event"], entry["reason"]) for entry in _recorded(record)] == [
        ("session_start", "startup"),
        ("session_shutdown", "fork"),
        ("session_start", "fork"),
        ("session_shutdown", "quit"),
    ]


async def test_a_cancelled_headless_run_still_closes_the_session(tmp_path, monkeypatch):
    """Ctrl-C cancels the main task, so `run_print`'s tail never executes.

    The session must still be closed: that is where the extension
    `session_shutdown` fires and the transcript is flushed. Regression pinned
    from `.scratch/repro_lifecycle_cleanup.py`.
    """
    record = tmp_path / "record.jsonl"
    _write(tmp_path / ".karen" / "extensions" / "lifecycle.py", _lifecycle_extension(record))
    _isolate(tmp_path, monkeypatch, [])
    cli = karen_cli.KarenCli(cwd=str(tmp_path), model_id="faux-1", fresh=True, provider="faux")

    started = asyncio.Event()

    async def hanging_prompt(self, input, images=None):
        started.set()
        await asyncio.Event().wait()  # an in-flight model turn, forever

    monkeypatch.setattr(Agent, "prompt", hanging_prompt)
    task = asyncio.create_task(cli.run_print(["hello"]))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cli.session.session is None  # the session really went down
    assert [(entry["event"], entry["reason"]) for entry in _recorded(record)] == [
        ("session_start", "startup"),
        ("session_shutdown", "quit"),
    ]


async def test_rpc_new_session_then_eof_closes_the_live_session(tmp_path, monkeypatch):
    """A `new_session` rebinds the RPC holder; EOF must close what it points at.

    Closing only the session captured at startup left the post-`new_session`
    session open and its `session_shutdown` unfired.
    """
    record = tmp_path / "record.jsonl"
    _write(tmp_path / ".karen" / "extensions" / "lifecycle.py", _lifecycle_extension(record))
    _isolate(tmp_path, monkeypatch, [])
    cli = karen_cli.KarenCli(cwd=str(tmp_path), model_id="faux-1", fresh=True, provider="faux")
    await cli._open_session(fresh=True)

    lines = iter(['{"type": "new_session"}'])

    async def input_lines():
        for line in lines:
            yield line

    written = []
    exit_code = await karen_rpc.run_rpc_mode(
        cli.session, input_iter=input_lines(), emit=written.append
    )

    assert exit_code == 0
    assert [(entry["event"], entry["reason"]) for entry in _recorded(record)] == [
        ("session_start", "startup"),
        ("session_shutdown", "new"),
        ("session_start", "new"),
        ("session_shutdown", "quit"),
    ]


# ---------------------------------------------------------------------------
# resources_discover: contributed prompt/skill dirs merge into the CLI's sets
# ---------------------------------------------------------------------------


async def test_resources_discover_merges_contributed_prompt_and_skill_dirs(tmp_path, monkeypatch):
    prompts_dir = tmp_path / "contrib" / "prompts"
    skills_dir = tmp_path / "contrib" / "skills"
    _write(prompts_dir / "extcmd.md", "---\ndescription: contributed template\n---\n\nDo $ARGUMENTS\n")
    _write(
        skills_dir / "extskill" / "SKILL.md",
        "---\nname: extskill\ndescription: Contributed skill\n---\n\nUse it.\n",
    )
    extension_file = tmp_path / "extras" / "contrib_ext.py"
    _write(extension_file, _resources_extension(prompts_dir, skills_dir))
    _isolate(tmp_path, monkeypatch, [])
    cli = karen_cli.KarenCli(
        cwd=str(tmp_path), model_id="faux-1", fresh=True, provider="faux",
        extension_paths=[str(extension_file)],
    )

    await cli._open_session(fresh=True)

    assert "extcmd" in {template.name for template in cli.templates}
    assert "extskill" in {skill.name for skill in cli.skills}
    await cli.session.close()


# ---------------------------------------------------------------------------
# RPC new_session keeps the extension runtime
# ---------------------------------------------------------------------------


async def test_rpc_new_session_keeps_extensions_without_duplicating_tools(tmp_path, monkeypatch):
    _write(tmp_path / ".karen" / "extensions" / "toolbox.py",
           _tool_extension(tmp_path / "record.jsonl"))
    _isolate(tmp_path, monkeypatch, [])
    cli = karen_cli.KarenCli(cwd=str(tmp_path), model_id="faux-1", fresh=True, provider="faux")
    await cli._open_session(fresh=True)
    runner = cli.extension_runner
    assert runner is not None

    holder = RpcSession(cli.session)
    new_session = await holder.new_session()

    # the runner is shared, not rebuilt, and its host follows the new session
    assert new_session.extensions is runner
    assert runner.host.session is new_session
    names = [tool.name for tool in new_session.tools]
    assert "mytool" in names
    assert len(names) == len(set(names))  # the wrapped tool was not re-added
    await new_session.close()


# ---------------------------------------------------------------------------
# TUI dispatch
# ---------------------------------------------------------------------------


class FakeController:
    """A `TerminalController` stand-in driven by a scripted key list."""

    def __init__(self, keys):
        self.keys = list(keys)
        self.frames = []
        self.entered = False
        self.left = False

    def enter(self):
        self.entered = True

    def leave(self):
        self.left = True

    def write(self, data):
        self.frames.append(data)

    def terminal_size(self):
        return (24, 80)

    def read_key(self):
        if not self.keys:
            return None  # EOF: ends the loop
        return self.keys.pop(0)


def _live(keys):
    controller = FakeController(keys)
    return LiveTui(controller=controller), controller


async def test_tui_dispatch_routes_extension_commands_and_notify_lands_on_the_app(capsys):
    live, _controller = _live(["/", "c", "m", "d", " ", "x", "enter"])
    handled = []

    async def handler(args, ctx):
        handled.append(args)
        ctx.notify("hi from cmd")

    record = Extension(path="fake", module_path="fake.py")
    record.commands["cmd"] = Command(name="cmd", handler=handler)
    runner = ExtensionRunner([record])

    # a bare KarenCli carrying only what the TUI input router reads
    cli = object.__new__(karen_cli.KarenCli)
    cli.session = None
    cli.templates = []
    cli.skills = []
    cli.tui = SimpleNamespace(app=live.app)
    cli.extension_runner = runner
    cli.extension_host = ExtensionHost(mode="tui", has_ui=True, notify=cli._extension_notify)

    async def prompt_handler(app, text):
        await cli._dispatch_tui_input(app, text)

    await live.run(prompt_handler)

    assert handled == ["x"]
    notices = [m.notice.text for m in live.transcript.messages if m.kind == "notice"]
    assert "hi from cmd" in notices
    # a notify inside the alt screen is a transcript notice, never a stdout print
    assert capsys.readouterr().out == ""

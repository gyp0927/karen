"""Unit tests for the extensions package itself: loader, runner, host, types
and events. Extensions under test are written as real .py files into tmp_path
directories and imported by the real loader; runner and host tests build the
records directly. No sessions, no shells, no network."""

import sys
from types import SimpleNamespace

import pytest

from karen_agent.types import AgentToolResult
from karen_ai import TextContent
from karen_coding_agent.bash_executor import BashResult
from karen_coding_agent.extensions import EVENT_NAMES
from karen_coding_agent.extensions import events as extension_events
from karen_coding_agent.extensions.host import (
    ExtensionHost,
    bash_result_from,
    make_agent_tool,
    tool_result_from,
)
from karen_coding_agent.extensions.loader import (
    ExtensionAPI,
    load_extensions,
    load_extensions_async,
)
from karen_coding_agent.extensions.runner import ExtensionRunner
from karen_coding_agent.extensions.types import (
    Command,
    Extension,
    ExtensionContext,
    ToolDefinition,
    ToolError,
    ToolResultDict,
    ToolResultEventResult,
    as_mapping,
    is_awaitable,
    maybe_await,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _factory_source(command_name):
    return (
        "def extension(pi):\n"
        f"    pi.register_command({command_name!r}, lambda args, ctx: None)\n"
    )


def _load(tmp_path, settings_paths=(), cli_paths=()):
    """Run the whole loader synchronously against empty conventional dirs."""
    return load_extensions(
        cwd=str(tmp_path / "proj"),
        agent_dir=str(tmp_path / "user"),
        settings_paths=list(settings_paths),
        cli_paths=list(cli_paths),
    )


def _extension(*, handlers=None, commands=None, path="ext.py"):
    record = Extension(path=path, module_path=path)
    for event, funcs in (handlers or {}).items():
        record.handlers.setdefault(event, []).extend(funcs)
    record.commands.update(commands or {})
    return record


# ---------------------------------------------------------------------------
# types + events
# ---------------------------------------------------------------------------


async def test_maybe_await_is_awaitable_and_as_mapping():
    class _Awaitable:
        def __init__(self, value):
            self.value = value

        def __await__(self):
            if False:
                yield
            return self.value

    assert is_awaitable(_Awaitable(3)) is True
    assert is_awaitable(3) is False
    assert await maybe_await(_Awaitable(3)) == 3
    assert await maybe_await(4) == 4
    assert as_mapping({"a": 1}) == {"a": 1}
    assert as_mapping(None) is None
    assert as_mapping("nope") is None
    assert as_mapping([1, 2]) is None


def test_event_names_match_every_payload_type():
    payload_classes = (
        extension_events.SessionStartEvent,
        extension_events.SessionInfoChangedEvent,
        extension_events.SessionShutdownEvent,
        extension_events.AgentStartEvent,
        extension_events.AgentEndEvent,
        extension_events.AgentErrorEvent,
        extension_events.TurnStartEvent,
        extension_events.TurnEndEvent,
        extension_events.MessageStartEvent,
        extension_events.MessageUpdateEvent,
        extension_events.MessageEndEvent,
        extension_events.ToolExecutionStartEvent,
        extension_events.ToolExecutionUpdateEvent,
        extension_events.ToolExecutionEndEvent,
        extension_events.ToolCallEvent,
        extension_events.ToolResultEvent,
        extension_events.UserBashEvent,
        extension_events.InputEvent,
        extension_events.ContextEvent,
        extension_events.ContextWithSystemEvent,
        extension_events.BeforeAgentStartEvent,
        extension_events.ResourcesDiscoverEvent,
    )
    assert {cls.type for cls in payload_classes} == set(EVENT_NAMES)
    assert len(EVENT_NAMES) == len(payload_classes)


# ---------------------------------------------------------------------------
# loader: discovery
# ---------------------------------------------------------------------------


def test_discovery_finds_direct_files_and_index_directories_only(tmp_path):
    ext_dir = tmp_path / "proj" / ".karen" / "extensions"
    _write(ext_dir / "alpha.py", _factory_source("alpha"))
    _write(ext_dir / "beta.py", _factory_source("beta"))
    _write(ext_dir / "pkg" / "index.py", _factory_source("pkg"))
    _write(ext_dir / "_private.py", _factory_source("private"))
    _write(ext_dir / ".hidden.py", _factory_source("hidden"))
    _write(ext_dir / "noindex" / "inner.py", _factory_source("inner"))
    _write(ext_dir / "outer" / "deep" / "index.py", _factory_source("deep"))
    _write(ext_dir / "notes.txt", "not python")

    loaded = _load(tmp_path)

    assert sorted(loaded.runner.commands()) == ["alpha", "beta", "pkg"]
    assert loaded.diagnostics == []


def test_discovery_takes_a_file_root_directly(tmp_path):
    single = tmp_path / "elsewhere" / "solo.py"
    _write(single, _factory_source("solo"))

    loaded = _load(tmp_path, settings_paths=[str(single)])

    assert list(loaded.runner.commands()) == ["solo"]
    assert loaded.extensions[0].source == "settings"
    assert loaded.diagnostics == []


def test_missing_conventional_dirs_are_silent_but_configured_paths_are_not(tmp_path):
    loaded = _load(tmp_path)
    assert loaded.extensions == []
    assert loaded.diagnostics == []

    loaded = _load(
        tmp_path,
        settings_paths=["no/such/place.py"],
        cli_paths=[str(tmp_path / "gone")],
    )
    assert loaded.extensions == []
    assert len(loaded.errors) == 2
    assert all("does not exist" in item.message for item in loaded.errors)
    assert {item.severity for item in loaded.diagnostics} == {"error"}


def test_the_same_file_arriving_via_two_roots_loads_once(tmp_path):
    shared = tmp_path / "shared.py"
    _write(shared, _factory_source("shared"))
    loaded = _load(tmp_path, settings_paths=[str(shared)], cli_paths=[str(shared)])
    assert len(loaded.extensions) == 1
    assert loaded.extensions[0].source == "settings"

    # a file found through a directory root is not reloaded as an explicit path
    proj_file = tmp_path / "proj" / ".karen" / "extensions" / "mine.py"
    _write(proj_file, _factory_source("mine"))
    loaded = _load(tmp_path, cli_paths=[str(proj_file)])
    assert len(loaded.extensions) == 1
    assert loaded.extensions[0].source == "project"


# ---------------------------------------------------------------------------
# loader: loading
# ---------------------------------------------------------------------------


def test_load_order_is_project_user_settings_cli(tmp_path):
    _write(tmp_path / "proj" / ".karen" / "extensions" / "p.py", _factory_source("from-project"))
    _write(tmp_path / "user" / "extensions" / "u.py", _factory_source("from-user"))
    settings_ext = tmp_path / "configured" / "s.py"
    cli_ext = tmp_path / "clidir" / "c.py"
    _write(settings_ext, _factory_source("from-settings"))
    _write(cli_ext, _factory_source("from-cli"))

    loaded = _load(tmp_path, settings_paths=[str(settings_ext)], cli_paths=[str(cli_ext)])

    assert [extension.source for extension in loaded.extensions] == [
        "project",
        "user",
        "settings",
        "cli",
    ]
    assert loaded.extensions == loaded.runner.extensions
    assert list(loaded.runner.commands()) == [
        "from-project",
        "from-user",
        "from-settings",
        "from-cli",
    ]


def test_load_extensions_runs_sync_and_async_factories(tmp_path):
    ext_dir = tmp_path / "proj" / ".karen" / "extensions"
    _write(ext_dir / "synced.py", _factory_source("synced"))
    _write(
        ext_dir / "asynced.py",
        "import asyncio\n\n"
        "async def extension(pi):\n"
        "    await asyncio.sleep(0)\n"
        "    pi.register_command('asynced', lambda args, ctx: None)\n",
    )

    loaded = _load(tmp_path)

    assert sorted(loaded.runner.commands()) == ["asynced", "synced"]
    assert loaded.diagnostics == []


async def test_load_extensions_async_awaits_async_factories(tmp_path):
    _write(
        tmp_path / "proj" / ".karen" / "extensions" / "asynced.py",
        "import asyncio\n\n"
        "async def extension(pi):\n"
        "    await asyncio.sleep(0)\n"
        "    pi.register_command('asynced', lambda args, ctx: None)\n",
    )

    loaded = await load_extensions_async(
        cwd=str(tmp_path / "proj"), agent_dir=str(tmp_path / "user")
    )

    assert list(loaded.runner.commands()) == ["asynced"]
    assert loaded.diagnostics == []


def test_a_raising_factory_is_a_diagnostic_and_never_fatal(tmp_path):
    ext_dir = tmp_path / "proj" / ".karen" / "extensions"
    _write(ext_dir / "abad.py", "def extension(pi):\n    raise RuntimeError('boom')\n")
    _write(ext_dir / "good.py", _factory_source("good"))

    loaded = _load(tmp_path)

    assert list(loaded.runner.commands()) == ["good"]
    assert len(loaded.errors) == 1
    assert loaded.errors[0].severity == "error"
    assert "boom" in loaded.errors[0].message


def test_an_async_factory_failure_is_a_diagnostic(tmp_path):
    _write(
        tmp_path / "proj" / ".karen" / "extensions" / "abad.py",
        "async def extension(pi):\n    raise ValueError('async boom')\n",
    )

    loaded = _load(tmp_path)

    assert loaded.extensions == []
    assert len(loaded.errors) == 1
    assert "async boom" in loaded.errors[0].message
    assert "async" in loaded.errors[0].message


def test_a_module_without_a_factory_is_a_diagnostic(tmp_path):
    _write(tmp_path / "proj" / ".karen" / "extensions" / "plain.py", "VALUE = 1\n")

    loaded = _load(tmp_path)

    assert loaded.extensions == []
    assert len(loaded.errors) == 1
    assert "factory" in loaded.errors[0].message


def test_a_slash_prefixed_command_name_is_a_diagnostic(tmp_path):
    _write(
        tmp_path / "proj" / ".karen" / "extensions" / "slash.py",
        "def extension(pi):\n    pi.register_command('/nope', lambda args, ctx: None)\n",
    )

    loaded = _load(tmp_path)

    assert loaded.extensions == []
    assert len(loaded.errors) == 1
    assert "leading slash" in loaded.errors[0].message
    assert "nope" not in loaded.runner.commands()


def test_a_directory_extension_imports_a_helper_beside_index(tmp_path):
    helper_name = "helper_beside_index_m10_test"
    pkg = tmp_path / "proj" / ".karen" / "extensions" / "pkgext"
    _write(pkg / f"{helper_name}.py", "VALUE = 'from-helper'\n")
    _write(
        pkg / "index.py",
        f"import {helper_name}\n\n"
        "def extension(pi):\n"
        f"    pi.register_command({helper_name}.VALUE, lambda args, ctx: None)\n",
    )

    try:
        loaded = _load(tmp_path)
        assert loaded.diagnostics == []
        assert list(loaded.runner.commands()) == ["from-helper"]
    finally:
        sys.modules.pop(helper_name, None)


def test_first_registration_wins_for_duplicate_tool_and_command_names(tmp_path):
    _write(
        tmp_path / "proj" / ".karen" / "extensions" / "first.py",
        "def extension(pi):\n"
        "    pi.register_tool(name='dup-tool', description='first', parameters={},"
        " execute=lambda *a: None)\n"
        "    pi.register_command('dup-cmd', lambda args, ctx: None, description='first')\n",
    )
    cli_ext = tmp_path / "cli" / "second.py"
    _write(
        cli_ext,
        "def extension(pi):\n"
        "    pi.register_tool(name='dup-tool', description='second', parameters={},"
        " execute=lambda *a: None)\n"
        "    pi.register_command('dup-cmd', lambda args, ctx: None, description='second')\n",
    )

    loaded = _load(tmp_path, cli_paths=[str(cli_ext)])

    tools = {tool.name: tool for tool in loaded.runner.tools()}
    assert tools["dup-tool"].description == "first"
    assert loaded.runner.commands()["dup-cmd"].description == "first"


# ---------------------------------------------------------------------------
# runner: dispatch and error isolation
# ---------------------------------------------------------------------------


async def test_emit_collects_non_none_results_and_isolates_a_failing_handler():
    seen = []
    sink = []

    def bad(event, ctx):
        raise RuntimeError("kaput")

    def good(event, ctx):
        seen.append(ctx is not None)
        return "value"

    def silent(event, ctx):
        return None

    record = _extension(handlers={"agent_start": [bad, good, silent]}, path="one.py")
    runner = ExtensionRunner([record], on_error=sink.append)

    results = await runner.emit(extension_events.AgentStartEvent())

    assert results == ["value"]
    assert seen == [True]  # a later handler still runs after the failure
    assert len(runner.errors) == 1
    error = runner.errors[0]
    assert error.event == "agent_start"
    assert error.extension_path == "one.py"
    assert "kaput" in error.message
    assert sink == runner.errors  # the on_error sink received the same record


async def test_the_context_is_built_only_when_a_handler_exists():
    builds = []

    def factory(signal=None):
        builds.append(signal)
        return ExtensionContext()

    runner = ExtensionRunner([], context_factory=factory)
    await runner.emit(extension_events.AgentStartEvent(), signal="sig")
    assert builds == []

    record = _extension(handlers={"agent_start": [lambda event, ctx: None]})
    runner = ExtensionRunner([record], context_factory=factory)
    await runner.emit(extension_events.AgentStartEvent(), signal="sig")
    assert builds == ["sig"]


# ---------------------------------------------------------------------------
# runner: transforming events
# ---------------------------------------------------------------------------


async def test_emit_tool_call_first_block_wins_and_mutation_is_observed():
    observed = []

    def mutating(event, ctx):
        event.input["extra"] = True

    def blocker(event, ctx):
        observed.append(dict(event.input))
        return {"block": True, "reason": "not today", "terminate": True}

    def never(event, ctx):
        observed.append("later")

    record = _extension(handlers={"tool_call": [mutating, blocker, never]})
    runner = ExtensionRunner([record])
    event = extension_events.ToolCallEvent(
        tool_call_id="1", tool_name="bash", input={"command": "ls"}
    )

    result = await runner.emit_tool_call(event)

    assert result.block is True
    assert result.reason == "not today"
    assert result.terminate is True
    assert observed == [{"command": "ls", "extra": True}]
    assert runner.errors == []


async def test_emit_tool_call_a_raising_handler_blocks_fail_closed():
    def bad(event, ctx):
        raise RuntimeError("kaput")

    def fine(event, ctx):
        return None

    record = _extension(handlers={"tool_call": [bad, fine]})
    runner = ExtensionRunner([record])

    result = await runner.emit_tool_call(extension_events.ToolCallEvent(tool_name="bash"))

    assert result.block is True
    assert result.reason == "a tool_call extension failed, so the call was not run"
    assert len(runner.errors) == 1


async def test_emit_tool_result_applies_field_wise_and_chains():
    seen = []

    def first(event, ctx):
        return {"content": ["edited"], "is_error": True}

    def second(event, ctx):
        seen.append((list(event.content), event.is_error))
        return {"details": {"k": 1}}

    record = _extension(handlers={"tool_result": [first, second]})
    runner = ExtensionRunner([record])
    event = extension_events.ToolResultEvent(
        tool_call_id="1", tool_name="bash", content=["original"], is_error=False
    )

    result = await runner.emit_tool_result(event)

    assert seen == [(["edited"], True)]  # the later handler observed the edit
    assert result.content == ["edited"]
    assert result.is_error is True
    assert result.details == {"k": 1}
    assert result.usage is None
    assert not hasattr(result, "terminate")  # pi's field set; a tool_result handler cannot change it


async def test_emit_tool_result_in_place_mutation_counts_and_no_change_is_empty():
    def mutating(event, ctx):
        event.content = ["mutated"]

    runner = ExtensionRunner([_extension(handlers={"tool_result": [mutating]})])
    result = await runner.emit_tool_result(
        extension_events.ToolResultEvent(tool_name="t", content=["orig"])
    )
    assert result.content == ["mutated"]
    assert result.details is None and result.is_error is None and result.usage is None

    def untouched(event, ctx):
        return None

    runner = ExtensionRunner([_extension(handlers={"tool_result": [untouched]})])
    result = await runner.emit_tool_result(
        extension_events.ToolResultEvent(tool_name="t", content=["orig"])
    )
    assert result == ToolResultEventResult()


async def test_emit_input_transforms_chain_and_handled_consumes():
    seen = []

    def upper(event, ctx):
        return {"action": "transform", "text": event.text.upper()}

    def suffix(event, ctx):
        seen.append(event.text)
        return {"action": "transform", "text": event.text + "!", "images": ["img"]}

    record = _extension(handlers={"input": [upper, suffix]})
    runner = ExtensionRunner([record])

    outcome = await runner.emit_input(extension_events.InputEvent(text="hello"))

    assert outcome.handled is False
    assert outcome.text == "HELLO!"
    assert outcome.images == ["img"]
    assert seen == ["HELLO"]  # the later handler saw the rewrite

    def consumer(event, ctx):
        return {"action": "handled"}

    runner = ExtensionRunner([_extension(handlers={"input": [consumer]})])
    outcome = await runner.emit_input(extension_events.InputEvent(text="hi"))
    assert outcome.handled is True
    assert outcome.text == "hi"


async def test_emit_input_continue_stops_the_chain():
    calls = []

    def stopper(event, ctx):
        calls.append("stopper")
        return {"action": "continue"}

    def later(event, ctx):
        calls.append("later")

    runner = ExtensionRunner([_extension(handlers={"input": [stopper, later]})])

    outcome = await runner.emit_input(extension_events.InputEvent(text="keep"))

    assert outcome.text == "keep"
    assert outcome.handled is False
    assert calls == ["stopper"]


async def test_emit_user_bash_a_result_answers_and_a_failure_blocks():
    answer = {"output": "answered", "exit_code": 0}

    def answering(event, ctx):
        return {"result": answer}

    runner = ExtensionRunner([_extension(handlers={"user_bash": [answering]})])
    outcome = await runner.emit_user_bash(extension_events.UserBashEvent(command="ls"))
    assert outcome.result is answer
    assert outcome.blocked is False

    def passing(event, ctx):
        return None

    runner = ExtensionRunner([_extension(handlers={"user_bash": [passing]})])
    outcome = await runner.emit_user_bash(extension_events.UserBashEvent(command="ls"))
    assert outcome.result is None
    assert outcome.blocked is False

    def bad(event, ctx):
        raise RuntimeError("kaput")

    runner = ExtensionRunner([_extension(handlers={"user_bash": [bad, passing]})])
    outcome = await runner.emit_user_bash(extension_events.UserBashEvent(command="ls"))
    assert outcome.blocked is True
    assert outcome.result is None
    assert outcome.reason == "a user_bash extension failed, so the command was not run"
    assert len(runner.errors) == 1


async def test_emit_message_end_same_role_replaces_and_role_change_is_refused():
    original = SimpleNamespace(role="assistant", content="a")
    same_role = SimpleNamespace(role="assistant", content="b")
    other_role = SimpleNamespace(role="user", content="c")

    def replacing(event, ctx):
        return {"message": same_role}

    def role_changing(event, ctx):
        return {"message": other_role}

    record = _extension(handlers={"message_end": [replacing, role_changing]}, path="msg.py")
    runner = ExtensionRunner([record])

    result = await runner.emit_message_end(extension_events.MessageEndEvent(message=original))

    assert result is same_role
    assert len(runner.errors) == 1
    assert runner.errors[0].event == "message_end"
    assert "role" in runner.errors[0].message

    runner = ExtensionRunner([])
    assert (
        await runner.emit_message_end(extension_events.MessageEndEvent(message=original))
        is original
    )


async def test_emit_before_agent_start_collects_and_chains_system_prompt():
    seen = []
    message_spec = {"content": "extra", "custom_type": "note"}

    def first(event, ctx):
        return {"message": message_spec, "system_prompt": "sys-one"}

    def second(event, ctx):
        seen.append(event.system_prompt)
        return {"system_prompt": "sys-two"}

    record = _extension(handlers={"before_agent_start": [first, second]})
    runner = ExtensionRunner([record])

    collected = await runner.emit_before_agent_start(
        extension_events.BeforeAgentStartEvent(prompt="hi", system_prompt="base")
    )

    assert collected == {"messages": [message_spec], "system_prompt": "sys-two"}
    assert seen == ["sys-one"]  # the later handler observed the replacement


async def test_emit_context_chains_replacements_and_a_failure_leaves_messages():
    seen = []

    def replacing(event, ctx):
        return {"messages": ["b"]}

    def bad(event, ctx):
        raise RuntimeError("kaput")

    def observing(event, ctx):
        seen.append(list(event.messages))
        return {"messages": event.messages + ["c"]}

    record = _extension(handlers={"context": [replacing, bad, observing]})
    runner = ExtensionRunner([record])

    result = await runner.emit_context(extension_events.ContextEvent(messages=["a"]))

    assert result == ["b", "c"]
    assert seen == [["b"]]  # the failed handler changed nothing
    assert len(runner.errors) == 1


async def test_emit_context_with_system_chains_replacements():
    def first(event, ctx):
        return {"messages": ["sys", "one"]}

    def second(event, ctx):
        return {"messages": event.messages + ["two"]}

    runner = ExtensionRunner([_extension(handlers={"context_with_system": [first, second]})])

    result = await runner.emit_context_with_system(
        extension_events.ContextWithSystemEvent(messages=["sys"])
    )

    assert result == ["sys", "one", "two"]


async def test_emit_resources_discover_merges_in_first_seen_order_without_duplicates():
    def first(event, ctx):
        return {"skill_paths": ["a", "b"], "prompt_paths": ["p"]}

    def second(event, ctx):
        return {"skill_paths": ["b", "c"], "prompt_paths": ["p", "q"], "theme_paths": ["t"]}

    runner = ExtensionRunner([_extension(handlers={"resources_discover": [first, second]})])

    merged = await runner.emit_resources_discover(
        extension_events.ResourcesDiscoverEvent(cwd=".", reason="startup")
    )

    assert merged == {
        "skill_paths": ["a", "b", "c"],
        "prompt_paths": ["p", "q"],
        "theme_paths": ["t"],
    }


# ---------------------------------------------------------------------------
# runner: commands and the bus
# ---------------------------------------------------------------------------


async def test_run_command_unknown_registered_and_raising():
    calls = []
    record = _extension(
        commands={"hi": Command(name="hi", handler=lambda args, ctx: calls.append((args, ctx)))}
    )
    runner = ExtensionRunner([record])

    assert await runner.run_command("nope", None, object()) is False
    assert await runner.run_command("hi", "world", "ctx") is True
    assert calls == [("world", "ctx")]

    def boom(args, ctx):
        raise ValueError("cmdfail")

    runner = ExtensionRunner(
        [_extension(commands={"bad": Command(name="bad", handler=boom)}, path="two.py")]
    )
    assert await runner.run_command("bad", None, None) is True
    assert len(runner.errors) == 1
    assert runner.errors[0].event == "command"
    assert runner.errors[0].extension_path == "<command:bad>"
    assert "cmdfail" in runner.errors[0].message


async def test_the_bus_isolates_failing_listeners_and_unsubscribe_works():
    seen = []
    runner = ExtensionRunner([])

    def bad(payload):
        raise RuntimeError("bus kaput")

    async def good(payload):
        seen.append(payload)

    off_bad = runner.on_bus("tick", bad)
    off_good = runner.on_bus("tick", good)

    await runner.emit_bus("tick", 1)
    assert seen == [1]
    assert len(runner.errors) == 1
    assert runner.errors[0].extension_path == "<bus:tick>"
    assert runner.errors[0].event == "tick"

    off_good()
    await runner.emit_bus("tick", 2)
    assert seen == [1]  # the unsubscribed listener no longer runs
    assert len(runner.errors) == 2  # the failing one is still subscribed

    off_bad()
    await runner.emit_bus("tick", 3)
    assert len(runner.errors) == 2  # now both are gone


async def test_the_extension_events_api_wraps_the_bus():
    record = _extension(path="api.py")
    runner = ExtensionRunner([record])
    api = ExtensionAPI(record, runner)
    seen = []

    unsubscribe = api.events.on("hello", lambda payload: seen.append(payload))
    await api.events.emit("hello", {"n": 1})
    assert seen == [{"n": 1}]

    unsubscribe()
    await api.events.emit("hello", {"n": 2})
    assert seen == [{"n": 1}]


# ---------------------------------------------------------------------------
# ExtensionAPI registration
# ---------------------------------------------------------------------------


def test_on_validates_event_names_and_unsubscribe_removes_the_handler():
    record = _extension()
    runner = ExtensionRunner([record])
    api = ExtensionAPI(record, runner)

    with pytest.raises(ValueError):
        api.on("not-an-event", lambda event, ctx: None)

    handler = lambda event, ctx: None
    unsubscribe = api.on("agent_start", handler)
    assert runner.handlers_for("agent_start") == [handler]
    assert runner.has_handlers("agent_start") is True

    unsubscribe()
    assert runner.handlers_for("agent_start") == []
    assert runner.has_handlers("agent_start") is False


def test_register_tool_and_command_validate_their_names():
    record = _extension()
    api = ExtensionAPI(record, ExtensionRunner([record]))

    with pytest.raises(ValueError):
        api.register_tool(name="", description="d", parameters={}, execute=lambda *a: None)
    api.register_tool(
        name="tool", description="d", parameters={}, execute=lambda *a: None, label="T"
    )
    assert record.tools["tool"].label == "T"

    with pytest.raises(ValueError):
        api.register_command("/bad", lambda args, ctx: None)
    with pytest.raises(ValueError):
        api.register_command("", lambda args, ctx: None)
    api.register_command("fine", lambda args, ctx: None, description="d")
    assert record.commands["fine"].description == "d"


def test_get_commands_spans_every_extension_and_get_settings_passthrough():
    first = _extension(commands={"a": Command(name="a", handler=lambda args, ctx: None)})
    second = _extension(
        commands={"b": Command(name="b", handler=lambda args, ctx: None)}, path="two.py"
    )
    runner = ExtensionRunner([first, second])
    settings = {"extensions": []}
    api = ExtensionAPI(second, runner, settings=settings)

    assert sorted(api.get_commands()) == ["a", "b"]
    assert api.get_settings() is settings


# ---------------------------------------------------------------------------
# host: result coercion
# ---------------------------------------------------------------------------


def test_bash_result_from_accepts_results_and_mappings():
    original = BashResult("out", 0, False, False, None)
    assert bash_result_from(original) is original

    snake = bash_result_from(
        {
            "output": "hi",
            "exit_code": 3,
            "cancelled": True,
            "truncated": True,
            "full_output_path": "p",
        }
    )
    assert (snake.output, snake.exit_code, snake.cancelled, snake.truncated) == (
        "hi",
        3,
        True,
        True,
    )
    assert snake.full_output_path == "p"

    camel = bash_result_from({"output": "hi", "exitCode": 7, "fullOutputPath": "q"})
    assert (camel.exit_code, camel.full_output_path) == (7, "q")

    defaults = bash_result_from({})
    assert (defaults.output, defaults.exit_code) == ("", None)
    assert (defaults.cancelled, defaults.truncated, defaults.full_output_path) == (
        False,
        False,
        None,
    )


def test_bash_result_from_rejects_anything_else():
    for value in ("text", 42, None, ["x"]):
        with pytest.raises(TypeError):
            bash_result_from(value)


def test_tool_result_from_normalizes_each_accepted_shape():
    passthrough = AgentToolResult(content=[TextContent(text="x")], details={"d": 1})
    assert tool_result_from(passthrough) is passthrough

    typed = tool_result_from(
        ToolResultDict(content=[TextContent(text="y")], details="det", terminate=True)
    )
    assert isinstance(typed, AgentToolResult)
    assert typed.content[0].text == "y"
    assert typed.details == "det"
    assert typed.terminate is True

    mapped = tool_result_from({"content": [TextContent(text="z")], "details": 5})
    assert isinstance(mapped, AgentToolResult)
    assert mapped.details == 5

    text = tool_result_from("plain")
    assert isinstance(text, AgentToolResult)
    assert isinstance(text.content[0], TextContent)
    assert text.content[0].text == "plain"

    empty = tool_result_from(None)
    assert isinstance(empty, AgentToolResult)
    assert empty.content[0].text == "(no output)"


# ---------------------------------------------------------------------------
# host: make_agent_tool
# ---------------------------------------------------------------------------


async def test_make_agent_tool_normalizes_sync_and_async_execute():
    calls = []

    def sync_execute(tool_call_id, args, signal, on_update):
        calls.append((tool_call_id, args, signal, on_update))
        return {"content": [TextContent(text="ok")], "details": {"n": 1}}

    tool = make_agent_tool(
        ToolDefinition(
            name="synced", description="d", parameters={"type": "object"}, execute=sync_execute
        )
    )
    assert tool.name == "synced"
    assert tool.label == "synced"  # the label defaults to the name
    assert tool.description == "d"

    result = await tool.execute("call-1", {"a": 1}, None, None)
    assert isinstance(result, AgentToolResult)
    assert result.details == {"n": 1}
    assert calls == [("call-1", {"a": 1}, None, None)]

    async def async_execute(tool_call_id, args, signal, on_update):
        return "from-async"

    tool = make_agent_tool(
        ToolDefinition(
            name="asynced", description="d", parameters={}, execute=async_execute, label="Async"
        )
    )
    assert tool.label == "Async"
    result = await tool.execute("call-2", {}, None, None)
    assert isinstance(result, AgentToolResult)
    assert result.content[0].text == "from-async"


async def test_make_agent_tool_turns_a_returned_tool_error_into_a_runtime_error():
    async def execute(tool_call_id, args, signal, on_update):
        return ToolError("nope")

    tool = make_agent_tool(ToolDefinition(name="t", description="d", parameters={}, execute=execute))
    with pytest.raises(RuntimeError, match="nope"):
        await tool.execute("c", {}, None, None)

    def raising_tool_error(tool_call_id, args, signal, on_update):
        raise ToolError("raised-nope")

    tool = make_agent_tool(
        ToolDefinition(name="t3", description="d", parameters={}, execute=raising_tool_error)
    )
    with pytest.raises(ToolError, match="raised-nope"):
        await tool.execute("c", {}, None, None)

    def raising(tool_call_id, args, signal, on_update):
        raise ValueError("raw")

    tool = make_agent_tool(
        ToolDefinition(name="t2", description="d", parameters={}, execute=raising)
    )
    with pytest.raises(ValueError, match="raw"):
        await tool.execute("c", {}, None, None)


# ---------------------------------------------------------------------------
# host: context and session controls
# ---------------------------------------------------------------------------


def test_make_context_reflects_the_face_and_the_live_idle_state():
    class _State:
        def __init__(self):
            self.is_streaming = False

    class _Agent:
        def __init__(self):
            self.state = _State()

    class _Session:
        def __init__(self):
            self.cwd = "/work"
            self.agent = _Agent()

    host = ExtensionHost(mode="tui", has_ui=True)
    session = _Session()
    host.bind(session)

    context = host.make_context(signal="sig")
    assert context.session is session
    assert context.mode == "tui"
    assert context.has_ui is True
    assert context.cwd == "/work"
    assert context.signal == "sig"
    assert context.is_idle() is True

    session.agent.state.is_streaming = True
    assert context.is_idle() is False  # read live, not captured at build time

    idle = ExtensionHost(mode="print", has_ui=False).make_context()
    assert idle.session is None
    assert idle.cwd == "."
    assert idle.is_idle() is True


def test_extension_api_session_controls_require_a_bound_session():
    record = _extension()
    api = ExtensionAPI(record, ExtensionRunner([record]))  # no host at all

    with pytest.raises(RuntimeError):
        api.append_entry("custom", {})
    with pytest.raises(RuntimeError):
        api.send_user_message("hi")
    with pytest.raises(RuntimeError):
        api.exec("echo hi")
    with pytest.raises(RuntimeError):
        api.set_model("provider/model")
    with pytest.raises(RuntimeError):
        api.set_session_name("x")
    with pytest.raises(RuntimeError):
        api.get_active_tools()

    unbound = ExtensionAPI(record, ExtensionRunner([record]), host=ExtensionHost())
    with pytest.raises(RuntimeError):
        unbound.append_entry("custom", {})


def test_notify_without_a_host_is_a_silent_noop_and_routes_with_one():
    record = _extension()
    api = ExtensionAPI(record, ExtensionRunner([record]))
    api.notify("hello")  # no host: nothing happens, nothing raises

    sent = []
    host = ExtensionHost(notify=lambda message, level="info": sent.append((message, level)))
    api = ExtensionAPI(record, ExtensionRunner([record]), host=host)
    api.notify("hello", level="warning")
    assert sent == [("hello", "warning")]


async def test_host_append_entry_delegates_to_the_session():
    class _Session:
        def __init__(self):
            self.calls = []

        async def _append_custom_entry(self, custom_type, data):
            self.calls.append((custom_type, data))
            return "entry-id"

    host = ExtensionHost()
    session = _Session()
    host.bind(session)
    record = _extension()
    api = ExtensionAPI(record, ExtensionRunner([record]), host=host)

    result = await api.append_entry("my-data", {"k": 1})

    assert result == "entry-id"
    assert session.calls == [("my-data", {"k": 1})]


async def test_host_set_model_resolves_provider_and_id():
    calls = {}

    class _Models:
        def get_model(self, provider, model_id):
            calls["lookup"] = (provider, model_id)
            if (provider, model_id) in (("prov", "m"), ("current", "m2")):
                return "MODEL"
            return None

    class _Session:
        model = SimpleNamespace(provider="current")
        models = _Models()

        def set_model(self, model):
            calls["set"] = model

    host = ExtensionHost()
    host.bind(_Session())

    assert await host.set_model("prov/m") is True
    assert calls["lookup"] == ("prov", "m")
    assert calls["set"] == "MODEL"

    assert await host.set_model("m2") is True  # a bare id uses the current provider
    assert calls["lookup"] == ("current", "m2")

    assert await host.set_model("nope/nada") is False

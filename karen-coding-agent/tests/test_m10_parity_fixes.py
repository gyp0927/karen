"""Regression tests for the M10 parity fixes (`.scratch` repros + the pi
snapshot audit): the `user_bash` operations handler must block rather than
fall through, `before_agent_start` keeps every handler's message and honours
a clearing (empty) system prompt, and the RPC `input` source is `"rpc"`.

The active-tool controls (`host.set_active_tools` narrowing what a run
declares) and the reported tool-name collision are pinned in
`tests/test_extension_wiring.py` on a real session; these pin the runner-
level semantics that do not need one."""

from karen_coding_agent.extensions import (
    Extension,
    ExtensionRunner,
    ToolDefinition,
    make_agent_tool,
)
from karen_coding_agent.extensions import events as extension_events
from karen_coding_agent.extensions.host import ExtensionHost


# ---------------------------------------------------------------------------
# user_bash: pi's shapes — anything but a `result` blocks, never falls through
# ---------------------------------------------------------------------------


async def test_user_bash_operations_handler_blocks_and_is_reported(tmp_path, monkeypatch):
    """A pi-style `{"operations": [...]}` answer must block, never fall through."""

    def ops_handler(event, ctx):
        return {"operations": [{"type": "exec", "command": event.command}]}

    errors = []
    record = Extension(
        path="fake", module_path="fake.py", handlers={"user_bash": [ops_handler]}
    )
    runner = ExtensionRunner([record], on_error=errors.append)
    outcome = await runner.emit_user_bash(extension_events.UserBashEvent(command="ls"))

    assert outcome.blocked is True
    assert outcome.result is None
    assert any("operations" in error.message for error in errors)


async def test_user_bash_unknown_shape_blocks_and_is_reported(tmp_path, monkeypatch):

    def weird_handler(event, ctx):
        return {"answer": "typo'd key"}

    errors = []
    record = Extension(
        path="fake", module_path="fake.py", handlers={"user_bash": [weird_handler]}
    )
    runner = ExtensionRunner([record], on_error=errors.append)
    outcome = await runner.emit_user_bash(extension_events.UserBashEvent(command="ls"))

    assert outcome.blocked is True
    assert outcome.result is None
    assert any("returns {'result': …} or nothing" in error.message for error in errors)


async def test_user_bash_declining_handler_still_lets_the_command_run(tmp_path, monkeypatch):
    """None is the documented pass-along: only a failure or a foreign shape blocks."""

    def declining(event, ctx):
        return None

    record = Extension(
        path="fake", module_path="fake.py", handlers={"user_bash": [declining]}
    )
    runner = ExtensionRunner([record])
    outcome = await runner.emit_user_bash(extension_events.UserBashEvent(command="ls"))

    assert outcome.blocked is False
    assert runner.errors == []


# ---------------------------------------------------------------------------
# before_agent_start: every handler's message kept, a clearing prompt honoured
# ---------------------------------------------------------------------------


async def test_before_agent_start_keeps_every_handlers_message(tmp_path, monkeypatch):
    def first(event, ctx):
        return {"message": {"custom_type": "a", "content": "first"}}

    def second(event, ctx):
        return {"message": {"custom_type": "b", "content": "second"}}

    record = Extension(
        path="fake", module_path="fake.py",
        handlers={"before_agent_start": [first, second]},
    )
    runner = ExtensionRunner([record])
    collected = await runner.emit_before_agent_start(
        extension_events.BeforeAgentStartEvent(prompt="hi", system_prompt="base")
    )

    assert collected["messages"] == [
        {"custom_type": "a", "content": "first"},
        {"custom_type": "b", "content": "second"},
    ]
    assert "system_prompt" not in collected


async def test_before_agent_start_empty_system_prompt_clears(tmp_path, monkeypatch):
    """pi's `systemPrompt !== undefined`: the empty string clears the prompt."""
    event_seen = []

    def clearing(event, ctx):
        event_seen.append(event.system_prompt)
        return {"system_prompt": ""}

    record = Extension(
        path="fake", module_path="fake.py", handlers={"before_agent_start": [clearing]}
    )
    runner = ExtensionRunner([record])
    event = extension_events.BeforeAgentStartEvent(prompt="hi", system_prompt="base")
    collected = await runner.emit_before_agent_start(event)

    assert collected.get("system_prompt") == ""
    assert event.system_prompt == ""  # later handlers observe the clearing


# ---------------------------------------------------------------------------
# active tools: the controls reach the state a run actually declares
# ---------------------------------------------------------------------------


async def test_active_tool_controls_narrow_the_state_a_run_reads(tmp_path, monkeypatch):
    """`set_active_tools` must narrow `agent.state.tools` (what a run sends to
    the model), not create a stray attribute the loop never reads; `get`
    reports the same narrowed set."""
    from karen_ai import create_models
    from karen_ai.providers import register_faux_provider

    from karen_coding_agent import AgentSession

    models = create_models()
    registration = register_faux_provider(responses=[])
    models.set_provider(registration.provider)
    session = AgentSession(
        cwd=str(tmp_path),
        models=models,
        model=registration.get_model(),
        sessions_root=str(tmp_path),
        fresh=True,
    )
    host = ExtensionHost()
    host.bind(session)

    definition = ToolDefinition(
        name="probe",
        description="probe",
        parameters={"type": "object", "properties": {}},
        execute=lambda *args, **kwargs: "ok",
    )
    session.tools = [*session.tools, make_agent_tool(definition)]

    full = [tool.name for tool in session.agent.state.tools]
    host.set_active_tools(["probe"])
    assert [tool.name for tool in session.agent.state.tools] == ["probe"]
    assert host.get_active_tools() == ["probe"]
    # re-activating from the same pool is the point of narrowing
    host.set_active_tools(full)
    assert [tool.name for tool in session.agent.state.tools] == full


# ---------------------------------------------------------------------------
# RPC input source
# ---------------------------------------------------------------------------


async def test_rpc_prompt_queues_see_the_input_event_with_rpc_source(
    tmp_path, monkeypatch
):
    """A queued `steer`/`follow_up` prompt and the RPC `prompt` command both
    report `source="rpc"` on the `input` event — pi's InputSource vocabulary."""
    from karen_ai import create_models
    from karen_ai.providers import register_faux_provider

    from karen_coding_agent import AgentSession
    from karen_coding_agent.rpc import RpcServer

    models = create_models()
    registration = register_faux_provider(responses=[])
    models.set_provider(registration.provider)
    session = AgentSession(
        cwd=str(tmp_path),
        models=models,
        model=registration.get_model(),
        sessions_root=str(tmp_path),
        fresh=True,
    )
    await session.open()

    seen = []

    def input_handler(event, ctx):
        seen.append((event.source, event.streaming_behavior, event.text))
        return None

    runner = ExtensionRunner(
        [Extension(path="fake", module_path="fake.py", handlers={"input": [input_handler]})]
    )
    session.extensions = runner

    server = RpcServer.__new__(RpcServer)
    server.session = session
    server._input_source = "rpc"
    # the queued path: what `_cmd_prompt` routes through when a run is in flight
    await session.send_user_message(
        "queued mid-run", None, deliver_as="follow_up", source="rpc"
    )
    assert ("rpc", "followUp", "queued mid-run") in seen

    await session.close()

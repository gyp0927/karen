"""End-to-end tests for the TUI's async driver (`LiveTui`) with a fake
terminal: no TTY is opened, but the real key loop, submit dispatch and
event ingestion run.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from karen_coding_agent import cli as karen_cli
from karen_coding_agent.tui.app import LiveTui


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


async def test_submit_dispatches_the_prompt_text():
    live, controller = _live(["h", "i", "enter"])
    seen = []

    async def handler(app, text):
        seen.append(text)

    code = await live.run(handler)
    assert code == 0
    assert seen == ["hi"]


async def test_user_message_lands_in_the_transcript_once():
    live, controller = _live(["h", "i", "enter"])

    async def handler(app, text):
        return None

    await live.run(handler)
    users = [m for m in live.transcript.messages if m.kind == "user"]
    assert len(users) == 1
    assert users[0].user.text == "hi"


async def test_the_editor_is_cleared_after_submit():
    live, controller = _live(["h", "i", "enter"])

    async def handler(app, text):
        return None

    await live.run(handler)
    assert live.app.editor_lines == [""]


async def test_agent_events_render_into_the_transcript():
    live, controller = _live([])

    async def handler(app, text):
        return None

    class _Am:
        type = "text_delta"
        delta = "Hello"

    class _Event:
        type = "message_update"
        assistant_message_event = _Am()

    live.app.start()
    live.app.ingest_agent_event(_Event(), None)
    live.app.stop()
    assistants = [m for m in live.transcript.messages if m.kind == "assistant"]
    assert assistants and assistants[0].assistant.text == "Hello"
    # frames were actually painted
    assert controller.frames


async def test_session_events_render_into_the_transcript():
    live, _ = _live([])
    live.app.start()
    live.app.ingest_session_event(
        {"type": "session_opened", "resumed": False, "reason": "new", "session_id": "abc"}
    )
    live.app.stop()
    notices = [m.notice.text for m in live.transcript.messages if m.kind == "notice"]
    assert any("abc" in text for text in notices)


async def test_quit_stops_the_loop_and_leaves_the_alt_screen():
    live, controller = _live(["ctrl_c"])
    code = await live.run(lambda app, text: None)
    assert code == 0
    assert controller.entered is True
    assert controller.left is True


async def test_eof_ends_the_loop():
    live, controller = _live([])
    code = await live.run(lambda app, text: None)
    assert code == 0
    assert controller.left is True


async def test_ctrl_j_builds_a_multiline_prompt_then_enter_sends_it():
    live, controller = _live(["a", "ctrl_j", "b", "ctrl_enter"])
    seen = []

    async def handler(app, text):
        seen.append(text)

    await live.run(handler)
    assert seen == ["a\nb"]


async def test_a_second_submit_while_a_run_is_in_flight_is_queued():
    live, controller = _live(["o", "n", "e", "enter", "t", "w", "o", "enter", "ctrl_c"])
    started = asyncio.Event()
    release = asyncio.Event()
    seen = []

    async def handler(app, text):
        seen.append(text)
        if text == "one":
            started.set()
            await release.wait()

    task = asyncio.create_task(live.run(handler))
    # give the loop a chance to submit the first prompt, start the run, and
    # queue the second submit
    for _ in range(200):
        if started.is_set() and controller.keys == []:
            break
        await asyncio.sleep(0.01)
    release.set()
    await asyncio.wait_for(task, 5)
    # the first prompt ran; the second was queued and sent after it finished
    assert seen == ["one", "two"]


async def test_prompt_handler_errors_do_not_kill_the_loop():
    live, controller = _live(["x", "enter", "y", "enter", "ctrl_c"])
    seen = []

    async def handler(app, text):
        seen.append(text)
        if text == "x":
            raise RuntimeError("boom")

    await asyncio.wait_for(live.run(handler), 5)
    assert seen == ["x", "y"]


# ---------------------------------------------------------------------------
# shell bypass (`!command`) and the command/prompt split
# ---------------------------------------------------------------------------


class FakeBashSession:
    """The slice of `AgentSession` the TUI's input router touches."""

    def __init__(self, output="", exit_code=0, cancelled=False, error=None, snapshot=None):
        self.output = output
        self.exit_code = exit_code
        self.cancelled = cancelled
        self.error = error
        #: Called after the output has been pushed, while the command is still
        #: "running" — sampling there is what proves the streams are live
        #: rather than the end-of-run snapshot being backfilled into the block.
        self.snapshot = snapshot
        self.mid_run = None
        self.commands = []
        self.aborted = False

    async def execute_bash(self, command, on_chunk=None, **kwargs):
        self.commands.append(command)
        if on_chunk is not None and self.output:
            on_chunk(self.output)
        if self.snapshot is not None:
            self.mid_run = self.snapshot()
        if self.error is not None:
            raise RuntimeError(self.error)
        return SimpleNamespace(
            output=self.output,
            exit_code=self.exit_code,
            cancelled=self.cancelled,
            truncated=False,
            full_output_path=None,
        )

    def abort_bash(self):
        self.aborted = True


def _cli_with(session, templates=(), skills=()):
    """A `KarenCli` carrying only what the input router reads.

    Built without `__init__`: opening a session needs a model, credentials and
    a session file, none of which this path touches.
    """
    cli = object.__new__(karen_cli.KarenCli)
    cli.session = session
    cli.templates = list(templates)
    cli.skills = list(skills)
    cli.tui = None
    return cli


def _bash_blocks(live):
    return [m for m in live.transcript.messages if m.kind == "bash"]


async def test_a_bang_line_runs_through_execute_bash():
    live, controller = _live(["!", "l", "s", "enter"])
    session = FakeBashSession(output="a.txt\nb.txt\n")
    cli = _cli_with(session)

    async def handler(app, text):
        await cli._dispatch_tui_input(app, text)

    await live.run(handler)
    assert session.commands == ["ls"]
    blocks = _bash_blocks(live)
    assert len(blocks) == 1
    assert blocks[0].bash.command == "ls"
    assert blocks[0].bash.output == "a.txt\nb.txt\n"
    assert blocks[0].bash.state == "done"
    # the bash block carries the command, so it is not echoed as a prompt too
    assert [m for m in live.transcript.messages if m.kind == "user"] == []


async def test_bash_output_streams_into_the_block_while_the_command_runs():
    """The live view, not just the end-of-run snapshot: if the `on_chunk`
    wiring is dropped, the block still fills in from `result.output` at the
    end, so only a sample taken mid-run can tell the two apart."""
    live, controller = _live(["!", "c", "a", "t", "enter"])

    def snapshot():
        blocks = _bash_blocks(live)
        if not blocks:
            return None
        return (blocks[0].bash.state, blocks[0].bash.output)

    session = FakeBashSession(output="part one\npart two\n", snapshot=snapshot)
    cli = _cli_with(session)

    async def handler(app, text):
        await cli._dispatch_tui_input(app, text)

    await live.run(handler)
    assert session.mid_run == ("running", "part one\npart two\n")


async def test_a_bang_line_with_a_nonzero_exit_is_an_error_block():
    live, controller = _live(["!", "f", "a", "l", "s", "e", "enter"])
    session = FakeBashSession(output="boom\n", exit_code=2)
    cli = _cli_with(session)

    async def handler(app, text):
        await cli._dispatch_tui_input(app, text)

    await live.run(handler)
    assert session.commands == ["false"]
    block = _bash_blocks(live)[0].bash
    assert block.state == "error"
    assert block.exit_code == 2


async def test_a_failing_bash_spawn_becomes_an_error_block():
    live, controller = _live(["!", "n", "o", "p", "e", "enter"])
    session = FakeBashSession(error="no such shell")
    cli = _cli_with(session)

    async def handler(app, text):
        await cli._dispatch_tui_input(app, text)

    await live.run(handler)
    block = _bash_blocks(live)[0].bash
    assert block.state == "error"
    assert "no such shell" in block.error_text


async def test_a_bare_bang_line_explains_the_usage():
    live, controller = _live(["!", "enter"])
    session = FakeBashSession()
    cli = _cli_with(session)

    async def handler(app, text):
        await cli._dispatch_tui_input(app, text)

    await live.run(handler)
    assert session.commands == []
    assert _bash_blocks(live) == []
    notices = [m.notice.text for m in live.transcript.messages if m.kind == "notice"]
    assert any("!<shell command>" in text for text in notices)


async def test_a_slash_command_line_is_not_left_pending():
    live, _ = _live([])
    live.app.start()
    cli = _cli_with(FakeBashSession())
    cli.tui = SimpleNamespace(app=live.app)
    live.app.notify_user_message("/help")

    await cli._dispatch_tui_input(live.app, "/help")

    live.app.stop()
    users = [m for m in live.transcript.messages if m.kind == "user"]
    assert len(users) == 1
    assert users[0].user.pending is False


class HangingBashSession:
    """A `!command` that never finishes on its own: only the teardown ends it."""

    def __init__(self):
        self.commands = []
        self.cancelled = False

    async def execute_bash(self, command, on_chunk=None, **kwargs):
        self.commands.append(command)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("the hanging command was never cancelled")

    def abort_bash(self):
        pass


async def test_ctrl_c_during_a_bang_line_leaves_no_streaming_cursor():
    """Ctrl+C tears the run down, so the block an answer was streaming into
    never gets its `agent_end`: nothing else would ever clear its cursor."""
    live, _ = _live(["!", "s", "l", "e", "e", "p", "enter", "ctrl_c"])
    session = HangingBashSession()
    cli = _cli_with(session)
    live.transcript.on_user_message("question")
    live.transcript.on_text_delta("partial answer")

    async def handler(app, text):
        await cli._dispatch_tui_input(app, text)

    await asyncio.wait_for(live.run(handler), 10)
    for _ in range(50):  # let the cancelled run unwind
        if session.cancelled:
            break
        await asyncio.sleep(0.01)

    assert session.commands == ["sleep"]
    assert session.cancelled is True
    assistants = [m for m in live.transcript.messages if m.kind == "assistant"]
    assert [m.assistant.streaming for m in assistants] == [False]

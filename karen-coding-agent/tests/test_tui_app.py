"""End-to-end tests for the TUI's async driver (`LiveTui`) with a fake
terminal: no TTY is opened, but the real key loop, submit dispatch and
event ingestion run.
"""

from __future__ import annotations

import asyncio

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

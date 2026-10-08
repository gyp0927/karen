"""TUI app loop: drives the transcript + layout off the AgentSession event
stream and a raw key reader.

The structure mirrors pi's interactive mode, but the rendering here is a
hand-rolled alt-screen viewport. `TuiApp` owns:

  * the `Transcript` (pure logic, fed by agent/session events)
  * the `Layout` (pure logic, produces ANSI lines)
  * the key source and screen sink (injected; the real ones wrap a
    `TerminalController`, tests wrap fakes)

`LiveTui` is the production wiring. Its run loop is **async**: key reads are
offloaded to a worker thread so the asyncio event loop stays free, and a
submitted prompt runs as a task, which keeps the TUI responsive (scrolling,
Ctrl-C to abort) while the model is streaming.
"""

from __future__ import annotations

import asyncio
import sys
from typing import List, Optional

from .layout import Layout, _LayoutInput
from .terminal import TerminalController
from .transcript import Transcript

# The default footer hint line. pi shows the active keybindings; we keep a
# short, stable hint and let the status line carry the live state.
_FOOTER = (
    "enter: send  ctrl+j: newline  up/down: scroll  ctrl+c: quit  !cmd: shell"
)


class TuiApp:
    """Interactive alt-screen front-end over `AgentSession`.

    Inject `key_reader`, `sink` and `size_provider` to test without a TTY. In
    production all three wrap a `TerminalController`.
    """

    def __init__(
        self,
        key_reader,
        sink,
        size_provider,
        controller: Optional[TerminalController] = None,
        width: int = 80,
        height: int = 24,
        color: bool = True,
    ) -> None:
        self._read_key = key_reader
        self._sink = sink
        self._size = size_provider
        self._controller = controller
        self.color = color
        self.transcript = Transcript()
        self.layout = Layout(width, height, color=color)
        self.status = ""
        self.editor_lines: List[str] = [""]
        self.editor_cursor = (0, 0)  # (line, col)
        self.pending_input = ""
        self.running = True
        #: False until `start()` enters the alt screen; events that arrive
        #: before then (the session-opened notice) update the model but must
        #: not paint over the normal screen.
        self.started = False
        self._status_provider = lambda: ""  # optional live status fn

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        if self._controller is not None:
            self._controller.enter()
        self.started = True
        self.redraw()

    def stop(self) -> None:
        self.running = False
        self.started = False
        if self._controller is not None:
            self._controller.leave()

    def set_status(self, text: str) -> None:
        self.status = text
        self.redraw()

    def set_status_provider(self, provider) -> None:
        self._status_provider = provider

    # -- event ingestion (called from the runner) ------------------------------

    def ingest_agent_event(self, event, session) -> None:
        """Mirror of `cli.py`'s `_on_agent_event`, redirected into the
        transcript instead of stdout."""
        event_type = getattr(event, "type", None)
        if event_type == "agent_start":
            self.transcript.on_agent_start()
        elif event_type == "message_update":
            am = event.assistant_message_event
            if am.type == "text_delta":
                self.transcript.on_text_delta(am.delta)
            elif am.type == "thinking_delta":
                self.transcript.on_thinking_delta(am.delta)
            elif am.type == "thinking_block_start":
                self.transcript.on_thinking_block_start(getattr(am, "thinking", "") or "")
            elif am.type == "thinking_block_end":
                self.transcript.on_thinking_block_end(getattr(am, "thinking", "") or "")
        elif event_type == "tool_execution_start":
            self.transcript.on_tool_execution_start(event.tool_call_id, event.tool_name, event.args)
        elif event_type == "tool_execution_update":
            self.transcript.on_tool_execution_update(event.tool_call_id, event.partial_result)
        elif event_type == "tool_execution_end":
            self.transcript.on_tool_execution_end(
                event.tool_call_id, event.tool_name, event.is_error, event.result
            )
        elif event_type == "agent_end":
            final = session.agent.state.messages[-1] if session is not None else None
            self.transcript.on_agent_end(final, will_retry=bool(getattr(event, "will_retry", None)))
        self.redraw()

    def ingest_session_event(self, event: dict) -> None:
        """Mirror of `cli.py`'s `_on_session_event` (dict events)."""
        self.transcript.on_session_event(event)
        self.redraw()

    def notify_user_message(self, text: str) -> None:
        """The REPL is about to run `text`: append it as a pending user line."""
        self.transcript.on_user_message(text)
        self.layout.follow = True
        self.redraw()

    def add_notice(self, text: str, error: bool = False) -> None:
        self.transcript.add_notice(text, error=error)
        self.redraw()

    # -- rendering --------------------------------------------------------------

    def redraw(self) -> None:
        if not self.started:
            return
        rows, cols = self._size()
        self.layout.resize(cols, rows)
        app = _LayoutInput()
        app.status = self.status or self._status_provider()
        app.editor_lines = tuple(self.editor_lines)
        app.footer = _FOOTER
        app.pending_input = self.pending_input if not any(self.editor_lines) else ""
        self._paint(self.layout.render(self.transcript, app))

    def _paint(self, lines: List[str]) -> None:
        # Repaint the full screen top-to-bottom; the alt screen keeps the
        # hardware cursor out of the way so no clearing is required.
        self._sink.write("\r\n".join(lines) + "\r\n")

    # -- input editing ------------------------------------------------------------

    def handle_key(self, key: str) -> str:
        """Apply one key to the editor; returns an action token for the runner.

        Actions: "noop" (redraw only), "submit" (send editor text), "quit",
        "clear". The runner owns the side effects; this only mutates
        editor/scroll state — in particular it never clears the editor text on
        submit, because the runner must read it first.
        """
        if key in ("quit", "ctrl_c"):
            return "quit"
        if key == "ctrl_l":
            return "clear"
        if key == "up":
            self.layout.scroll_up(3)
            return "noop"
        if key == "down":
            self.layout.scroll_down(3)
            return "noop"
        if key == "ctrl_d" and not any(self.editor_lines):
            return "quit"
        if key == "enter":
            # Plain enter submits a single-line prompt; on a multi-line editor
            # it inserts a newline instead.
            if len(self.editor_lines) == 1:
                return "submit" if self.editor_lines[0].strip() else "noop"
            self._insert_newline()
            return "noop"
        if key == "ctrl_j":
            # ctrl+J inserts a newline unconditionally: this is how a
            # single-line editor becomes multi-line.
            self._insert_newline()
            return "noop"
        if key == "ctrl_enter":
            # ctrl+enter always submits, even a multi-line buffer.
            return "submit" if self.editor_text else "noop"
        if key == "backspace":
            self._backspace()
            return "noop"
        if key == "ctrl_u":
            self._clear_line_before_cursor()
            return "noop"
        if key in ("ctrl_a", "left"):
            self._move_cursor_left()
            return "noop"
        if key in ("ctrl_e", "right"):
            self._move_cursor_right()
            return "noop"
        if key in ("ctrl_k", "ctrl_d"):
            self._delete_line_after_cursor()
            return "noop"
        if key == "tab":
            self._insert_text("    ")
            return "noop"
        if len(key) == 1:
            self._insert_text(key)
            return "noop"
        return "noop"

    # -- editor internals ----------------------------------------------------------

    def _cur_line(self) -> int:
        return min(self.editor_cursor[0], len(self.editor_lines) - 1)

    def _cur_col(self) -> int:
        line = self._cur_line()
        return min(self.editor_cursor[1], len(self.editor_lines[line]))

    def _insert_text(self, text: str) -> None:
        line_idx = self._cur_line()
        col = self._cur_col()
        current = self.editor_lines[line_idx]
        self.editor_lines[line_idx] = current[:col] + text + current[col:]
        self.editor_cursor = (line_idx, col + len(text))

    def _insert_newline(self) -> None:
        line_idx = self._cur_line()
        col = self._cur_col()
        current = self.editor_lines[line_idx]
        before = current[:col]
        after = current[col:]
        # Split the current line into a before-part and an after-part. Both
        # halves always exist (after may be empty), and the lines below shift
        # down by one row.
        new_lines = self.editor_lines[: line_idx + 1]
        new_lines[-1] = before
        new_lines.append(after)
        new_lines.extend(self.editor_lines[line_idx + 1 :])
        self.editor_lines = new_lines
        self.editor_cursor = (line_idx + 1, 0)

    def _backspace(self) -> None:
        line_idx = self._cur_line()
        col = self._cur_col()
        if col > 0:
            current = self.editor_lines[line_idx]
            self.editor_lines[line_idx] = current[: col - 1] + current[col:]
            self.editor_cursor = (line_idx, col - 1)
        elif line_idx > 0:
            prev = self.editor_lines[line_idx - 1]
            prev_len = len(prev)
            # Join this line's text onto the previous line and drop this row.
            self.editor_lines[line_idx - 1] = prev + self.editor_lines[line_idx]
            del self.editor_lines[line_idx]
            self.editor_cursor = (line_idx - 1, prev_len)

    def _clear_line_before_cursor(self) -> None:
        line_idx = self._cur_line()
        col = self._cur_col()
        self.editor_lines[line_idx] = self.editor_lines[line_idx][col:]
        self.editor_cursor = (line_idx, 0)

    def _delete_line_after_cursor(self) -> None:
        """Kill to end of line: keep the text before the cursor, drop the rest.
        On an earlier line the killed remainder is prepended to the next line
        (they share the row below)."""
        line_idx = self._cur_line()
        col = self._cur_col()
        current = self.editor_lines[line_idx]
        rest = current[col:]
        self.editor_lines[line_idx] = current[:col]
        if line_idx + 1 < len(self.editor_lines):
            self.editor_lines[line_idx + 1] = rest + self.editor_lines[line_idx + 1]
        self.editor_cursor = (line_idx, col)

    def _move_cursor_left(self) -> None:
        line_idx, col = self._cur_line(), self._cur_col()
        if col > 0:
            self.editor_cursor = (line_idx, col - 1)
        elif line_idx > 0:
            prev_len = len(self.editor_lines[line_idx - 1])
            self.editor_cursor = (line_idx - 1, prev_len)

    def _move_cursor_right(self) -> None:
        line_idx, col = self._cur_line(), self._cur_col()
        if col < len(self.editor_lines[line_idx]):
            self.editor_cursor = (line_idx, col + 1)
        elif line_idx + 1 < len(self.editor_lines):
            self.editor_cursor = (line_idx + 1, 0)

    # -- convenience for the runner -------------------------------------------------

    @property
    def editor_text(self) -> str:
        return "\n".join(self.editor_lines).strip()

    def clear_editor(self) -> None:
        self.editor_lines = [""]
        self.editor_cursor = (0, 0)
        self.pending_input = ""

    def take_pending_input(self) -> str:
        """Consume and return text queued while a run was in flight."""
        text = self.pending_input
        self.pending_input = ""
        return text


# ---------------------------------------------------------------------------
# Production wiring
# ---------------------------------------------------------------------------


class LiveTui:
    """Async driver: a `TuiApp` over a real terminal.

    The key loop runs in an executor thread (`asyncio.to_thread`) so it blocks
    without freezing the event loop; a submitted prompt is started as a task,
    which keeps the viewport responsive while the model streams.
    """

    def __init__(self, controller: Optional[TerminalController] = None,
                 in_stream=None, out_stream=None) -> None:
        in_stream = in_stream if in_stream is not None else sys.stdin
        out_stream = out_stream if out_stream is not None else sys.stdout
        self.controller = controller or TerminalController(in_stream, out_stream)
        self.app = TuiApp(
            key_reader=self.controller.read_key,
            sink=self.controller,
            size_provider=self.controller.terminal_size,
            controller=self.controller,
            color=_isatty(out_stream),
        )

    @property
    def transcript(self) -> Transcript:
        return self.app.transcript

    async def run(self, prompt_handler) -> int:
        """`await prompt_handler(app, text)` runs each submitted prompt.

        The key loop runs in an executor thread so a blocking read never
        freezes the event loop; each prompt is started as a task, which keeps
        the viewport responsive (scrolling, Ctrl-C) while the model streams.
        A prompt submitted while another is in flight is queued and sent
        automatically once the in-flight one completes.

        Returns 0 on a clean quit/EOF.
        """
        self.app.start()
        task: Optional[asyncio.Task] = None
        try:
            while self.app.running:
                if task is not None and task.done():
                    # The in-flight prompt finished: it may have raised or
                    # exited normally, so the loop must not wait on a
                    # dead task.
                    self._reap(task)
                    task = None
                text = self.app.take_pending_input()
                if text:
                    self.app.clear_editor()
                    self.app.notify_user_message(text)
                    task = asyncio.create_task(prompt_handler(self.app, text))
                    continue
                key = await asyncio.to_thread(self.app._read_key)
                if key is None:  # EOF on stdin
                    break
                action = self.app.handle_key(key)
                if action == "quit":
                    break
                if action == "clear":
                    self.app._sink.write("\x1b[2J\x1b[H")
                    continue
                if action != "submit":
                    self.app.redraw()
                    continue
                text = self.app.editor_text
                if not text:
                    self.app.redraw()
                    continue
                if task is not None and not task.done():
                    # A run is in flight: pi steers instead of starting a
                    # second one. Queue the text; it is sent as soon as the
                    # in-flight run completes.
                    self.app.pending_input = text
                    self.app.editor_lines = [""]
                    self.app.editor_cursor = (0, 0)
                    self.app.redraw()
                    continue
                self.app.clear_editor()
                self.app.notify_user_message(text)
                task = asyncio.create_task(prompt_handler(self.app, text))
        finally:
            if task is not None and not task.done():
                task.cancel()
            self.app.stop()
        return 0

    def _reap(self, task: "asyncio.Task") -> None:
        """Swallow the exception of a finished prompt task (already surfaced
        by the handler) so the loop does not crash on a dead task."""
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


def _isatty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


__all__ = ["TuiApp", "LiveTui"]

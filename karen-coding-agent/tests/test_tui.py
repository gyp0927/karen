"""Offline tests for the TUI's pure logic: transcript model, layout renderer,
the multi-line editor key handling, and the terminal key decoder (CSI/byte
tables — the pure parts, no TTY is opened)."""

from __future__ import annotations

import io
import os
import sys
from types import SimpleNamespace

import pytest

from karen_coding_agent.tui import Layout, TuiApp, Transcript
from karen_coding_agent.tui.terminal import TerminalController, _decode_csi
from karen_coding_agent.tui.transcript import _BASH_MAX_LINES, _BASH_MAX_LINE_CHARS


# -- event shims (the same shapes the agent/session layers emit) -------------


def _text_delta(delta: str):
    return SimpleNamespace(
        type="message_update",
        assistant_message_event=SimpleNamespace(type="text_delta", delta=delta),
    )


def _tool_start(name: str, args=None):
    return SimpleNamespace(type="tool_execution_start", tool_name=name, args=args)


def _tool_end(name: str, is_error=False, result=None):
    return SimpleNamespace(type="tool_execution_end", tool_name=name, is_error=is_error, result=result)


def _agent_end(final, will_retry=False):
    return SimpleNamespace(type="agent_end", messages=[], will_retry=will_retry, _final=final)


class _Sink:
    def __init__(self):
        self.frames = []

    def write(self, data):
        self.frames.append(data)


def _make_app(width=80, height=24, keys=()):
    key_queue = list(keys)
    sink = _Sink()

    def read_key():
        return key_queue.pop(0) if key_queue else None

    app = TuiApp(key_reader=read_key, sink=sink, size_provider=lambda: (height, width), color=False)
    return app, key_queue, sink


# ---------------------------------------------------------------------------
# transcript model
# ---------------------------------------------------------------------------


def test_user_message_is_appended_and_marked_pending():
    t = Transcript()
    msg = t.on_user_message("hello")
    assert t.messages[-1] is msg
    assert msg.user.pending is True
    t.mark_user_message_sent(msg)
    assert msg.user.pending is False


def test_streaming_text_accumulates_into_one_assistant_block():
    t = Transcript()
    t.on_user_message("hi")
    t.on_text_delta("He")
    t.on_text_delta("llo")
    t.finish_assistant()
    assistant = t.messages[-1]
    assert assistant.kind == "assistant"
    assert assistant.assistant.text == "Hello"
    assert assistant.assistant.streaming is False


def test_second_message_open_starts_a_fresh_assistant_block():
    t = Transcript()
    t.on_user_message("a")
    t.on_text_delta("x")
    t.finish_assistant()
    t.on_user_message("b")
    t.on_text_delta("y")
    assert [m.kind for m in t.messages] == ["user", "assistant", "user", "assistant"]
    assert t.messages[3].assistant.text == "y"


def test_error_stop_reason_is_stored_on_the_assistant_block():
    final = SimpleNamespace(stop_reason="error", error_message="boom")
    t = Transcript()
    t.on_user_message("hi")
    t.on_text_delta("partial")
    t.on_agent_end(final, will_retry=False)
    assert t.messages[-1].assistant.error == "boom"
    t.messages[-1].assistant.streaming is False


def test_a_retried_run_is_not_a_failure():
    t = Transcript()
    t.on_user_message("hi")
    t.on_text_delta("partial")
    t.on_agent_end(SimpleNamespace(stop_reason="error", error_message="x"), will_retry=True)
    # no error text should land: the retry continues the same stream
    assert t.messages[-1].assistant.error == ""


def test_a_retry_continues_the_same_assistant_block_across_notices():
    t = Transcript()
    t.on_user_message("hi")
    t.on_text_delta("partial")
    t.on_agent_end(SimpleNamespace(stop_reason="error", error_message="x"), will_retry=True)
    # the retry announcement is appended between the failed run and its retry
    t.on_session_event(
        {"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3, "delayMs": 500,
         "errorMessage": "x"}
    )
    t.on_agent_end(SimpleNamespace(stop_reason="error", error_message="x"), will_retry=True)
    t.on_text_delta("Hello")
    assistants = [m for m in t.messages if m.kind == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].assistant.text == "partialHello"


def test_a_retried_run_keeps_streaming():
    t = Transcript()
    t.on_user_message("hi")
    t.on_text_delta("partial")
    t.on_agent_end(SimpleNamespace(stop_reason="error", error_message="x"), will_retry=True)
    assert t.messages[-1].assistant.streaming is True


def test_agent_start_clears_pending():
    t = Transcript()
    msg = t.on_user_message("hi")
    assert msg.user.pending is True
    t.on_agent_start()
    assert msg.user.pending is False


def test_successful_run_clears_pending():
    t = Transcript()
    msg = t.on_user_message("hi")
    t.on_text_delta("ok")
    t.on_agent_end(SimpleNamespace(stop_reason="stop", error_message=None))
    assert msg.user.pending is False


def test_aborted_run_marks_the_user_line():
    t = Transcript()
    msg = t.on_user_message("hi")
    t.on_agent_start()
    t.on_agent_end(SimpleNamespace(stop_reason="aborted", error_message=None))
    assert msg.user.text == "hi (aborted)"


def test_error_before_any_delta_does_not_touch_the_previous_run():
    t = Transcript()
    t.on_user_message("a")
    t.on_text_delta("ok")
    t.on_agent_end(SimpleNamespace(stop_reason="stop", error_message=None))
    t.on_user_message("b")
    # run 2 fails before emitting any text
    t.on_agent_end(SimpleNamespace(stop_reason="error", error_message="boom"))
    first_assistant = [m for m in t.messages if m.kind == "assistant"][0]
    assert first_assistant.assistant.error == ""
    assert t.messages[-1].user.text == "b (error)"


def test_parallel_tool_calls_match_by_call_id():
    t = Transcript()
    t.on_tool_execution_start("A", "read", {"path": "a"})
    t.on_tool_execution_start("B", "read", {"path": "b"})
    # A finishes first, with an error
    t.on_tool_execution_end("A", "read", is_error=True,
                            result=SimpleNamespace(content=[SimpleNamespace(text="nope")]))
    tools = [m.tool for m in t.messages if m.kind == "tool"]
    assert tools[0].state == "error"
    assert tools[1].state == "running"
    t.on_tool_execution_end("B", "read", is_error=False, result=None)
    assert tools[1].state == "done"
    assert tools[0].state == "error"


def test_thinking_placeholder_is_replaced_by_real_text():
    t = Transcript()
    t.on_user_message("hi")
    t.on_thinking_block_end("let me think")
    assert t.messages[-1].assistant.text == "[thinking] let me think"
    t.on_text_delta("Answer")
    assert t.messages[-1].assistant.text == "Answer"


def test_tool_error_text_is_captured():
    t = Transcript()
    t.on_tool_execution_start("call-1", "read", {"path": "a.py"})
    content = [SimpleNamespace(text="no such file")]
    t.on_tool_execution_end("call-1", "read", is_error=True, result=SimpleNamespace(content=content))
    tool = t.messages[-1]
    assert tool.kind == "tool"
    assert tool.tool.state == "error"
    assert tool.tool.error_text == "no such file"


def test_tool_done_has_no_error_text():
    t = Transcript()
    t.on_tool_execution_start("call-2", "ls", None)
    t.on_tool_execution_end("call-2", "ls", is_error=False, result=None)
    assert t.messages[-1].tool.state == "done"
    assert t.messages[-1].tool.error_text == ""


def test_session_events_become_notices():
    t = Transcript()
    t.on_session_event({"type": "session_opened", "resumed": False, "reason": "new", "session_id": "abc"})
    t.on_session_event({"type": "compaction_end", "compacted": True, "tokens_before": 5000})
    texts = [m.notice.text for m in t.messages if m.kind == "notice"]
    assert any("resumed" in x or "new session" in x for x in texts) or "new session abc" in texts
    assert any("compacted" in x for x in texts)


def test_failed_compaction_is_flagged_as_error():
    t = Transcript()
    t.on_session_event({"type": "compaction_end", "compacted": False, "detail": "kaboom"})
    assert t.messages[-1].kind == "notice"
    assert t.messages[-1].notice.error is True


# ---------------------------------------------------------------------------
# shell bypass (`!command`)
# ---------------------------------------------------------------------------


def _bash_result(output="", exit_code=0, cancelled=False, truncated=False, full_output_path=None):
    return SimpleNamespace(
        output=output,
        exit_code=exit_code,
        cancelled=cancelled,
        truncated=truncated,
        full_output_path=full_output_path,
    )


def test_bash_chunks_stream_into_one_block():
    t = Transcript()
    msg = t.on_bash_start("echo hi")
    t.on_bash_chunk(msg, "hi")
    assert t.messages[-1].kind == "bash"
    assert t.messages[-1].bash.state == "running"
    t.on_bash_chunk(msg, "\n")
    t.on_bash_end(msg, _bash_result(output="hi\n"))
    bash = msg.bash
    assert bash.output == "hi\n"
    assert bash.state == "done"
    assert bash.exit_code == 0


def test_bash_partial_lines_survive_a_chunk_boundary():
    t = Transcript()
    msg = t.on_bash_start("printf")
    t.on_bash_chunk(msg, "par")
    t.on_bash_chunk(msg, "tial\nsecond")
    assert msg.bash.output == "partial\nsecond"


def test_bash_output_without_a_trailing_newline_is_kept_as_is():
    t = Transcript()
    msg = t.on_bash_start("printf")
    t.on_bash_chunk(msg, "no newline")
    assert msg.bash.output == "no newline"


def test_bash_nonzero_exit_is_an_error_state():
    t = Transcript()
    msg = t.on_bash_start("false")
    t.on_bash_end(msg, _bash_result(exit_code=1))
    assert msg.bash.state == "error"
    assert msg.bash.exit_code == 1


def test_bash_cancelled_run_is_its_own_state():
    t = Transcript()
    msg = t.on_bash_start("sleep 100")
    t.on_bash_end(msg, _bash_result(exit_code=None, cancelled=True))
    assert msg.bash.state == "cancelled"


def test_bash_spawn_failure_is_recorded_as_an_error():
    t = Transcript()
    msg = t.on_bash_start("nope")
    t.on_bash_end(msg, error="[Errno 2] no such file")
    assert msg.bash.state == "error"
    assert "no such file" in msg.bash.error_text


def test_bash_truncation_reports_the_spill_file():
    t = Transcript()
    msg = t.on_bash_start("yes")
    t.on_bash_end(msg, _bash_result(output="x\n", truncated=True, full_output_path="C:/tmp/full.txt"))
    assert msg.bash.truncated is True
    assert msg.bash.full_output_path == "C:/tmp/full.txt"


def test_bash_output_is_bounded_to_a_tail():
    t = Transcript()
    msg = t.on_bash_start("yes")
    for index in range(1000):
        t.on_bash_chunk(msg, f"line {index}\n")
    bash = msg.bash
    assert len(bash.lines) == _BASH_MAX_LINES + 1  # the 400 kept + the open one
    assert bash.dropped_lines == 1000 - _BASH_MAX_LINES
    # the window holds a contiguous tail up to the last written line
    assert bash.output.endswith("line 999\n")
    assert f"line {1000 - _BASH_MAX_LINES}\n" in bash.output
    assert f"line {999 - _BASH_MAX_LINES}\n" not in bash.output


def test_a_single_unbroken_line_keeps_only_its_tail():
    t = Transcript()
    msg = t.on_bash_start("base64 -w0 big.bin")
    t.on_bash_chunk(msg, "START" + "a" * 100_000)
    t.on_bash_chunk(msg, "END")
    line = msg.bash.lines[-1]
    assert len(line) == _BASH_MAX_LINE_CHARS + 1  # the tail plus the "…" marker
    assert line.startswith("…")
    assert line.endswith("END")
    assert "START" not in line


def test_a_long_complete_line_is_capped_too():
    t = Transcript()
    msg = t.on_bash_start("cat bundle.js")
    t.on_bash_chunk(msg, ("z" * 5000) + "\ntail\n")
    assert len(msg.bash.lines[0]) == _BASH_MAX_LINE_CHARS + 1
    assert msg.bash.lines[0].startswith("…")
    assert msg.bash.lines[1] == "tail"


def test_a_capped_line_does_not_accumulate_markers():
    t = Transcript()
    msg = t.on_bash_start("yes")
    for _ in range(20):
        t.on_bash_chunk(msg, "q" * 20_000)
    assert len(msg.bash.lines[-1]) == _BASH_MAX_LINE_CHARS + 1
    assert msg.bash.lines[-1].count("…") == 1


def test_a_bash_block_does_not_disturb_the_assistant_run():
    t = Transcript()
    t.on_user_message("hi")
    t.on_text_delta("partial")
    t.on_bash_start("ls")
    t.on_text_delta("more")
    assistants = [m for m in t.messages if m.kind == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].assistant.text == "partialmore"


def test_a_slash_command_line_is_not_left_pending():
    t = Transcript()
    msg = t.on_user_message("/tree")
    t.resolve_pending()
    assert msg.user.pending is False


def _streaming_run(t: Transcript, text: str = "part one") -> None:
    """A run that has started and is still producing text."""
    t.on_user_message("hi")
    t.on_agent_start()
    t.on_text_delta(text)


def test_a_slash_command_does_not_split_the_run_it_interrupts():
    """A command line submitted mid-answer is not a run of its own, so it must
    not cost the answer its block: a second block would split the text in two
    and leave the first with a cursor that no `agent_end` will ever clear."""
    t = Transcript()
    _streaming_run(t)
    t.on_user_message("/help")
    t.resolve_pending()
    t.on_text_delta(" part two")
    assistants = [m for m in t.messages if m.kind == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].assistant.text == "part one part two"
    t.finish_assistant()
    assert [m.assistant.streaming for m in assistants] == [False]


def test_a_bang_line_does_not_split_the_run_it_interrupts():
    t = Transcript()
    _streaming_run(t)
    t.on_user_message("!ls")
    t.drop_pending_user("!ls")
    t.on_bash_start("ls")
    t.on_text_delta(" part two")
    assistants = [m for m in t.messages if m.kind == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].assistant.text == "part one part two"
    t.finish_assistant()
    assert [m.assistant.streaming for m in assistants] == [False]


def test_a_prompt_really_does_start_a_new_assistant_block():
    """The other side of the same coin: a second *prompt* still opens a block
    of its own, and a command line after it must not hand the previous run's
    block back over it."""
    t = Transcript()
    _streaming_run(t)
    t.finish_assistant()
    t.on_user_message("!ls")
    t.drop_pending_user("!ls")
    t.on_user_message("second prompt")
    t.on_agent_start()
    t.on_text_delta("second answer")
    t.on_user_message("/help")
    t.resolve_pending()
    t.on_text_delta(" continues")
    assistants = [m for m in t.messages if m.kind == "assistant"]
    assert [m.assistant.text for m in assistants] == ["part one", "second answer continues"]


def test_drop_pending_user_removes_the_echoed_bash_line():
    t = Transcript()
    t.on_user_message("!ls")
    t.drop_pending_user("!ls")
    assert t.messages == []
    # a later prompt still becomes pending normally
    t.on_user_message("hi")
    assert t.messages[-1].user.pending is True


def test_drop_pending_user_ignores_a_different_line():
    t = Transcript()
    t.on_user_message("!ls")
    t.drop_pending_user("/tree")
    assert len(t.messages) == 1


# ---------------------------------------------------------------------------
# layout renderer
# ---------------------------------------------------------------------------


def _lines(t: Transcript, layout: Layout, app=None) -> list:
    from karen_coding_agent.tui.layout import _LayoutInput

    app = app or _LayoutInput()
    app.status = "model: test | ready"
    app.editor_lines = tuple(app.editor_lines or ("",))
    app.footer = "hint"
    return layout.render(t, app)


def test_render_returns_exactly_height_lines():
    t = Transcript()
    layout = Layout(60, 12, color=False)
    out = _lines(t, layout)
    assert len(out) == 12


def test_render_returns_exactly_height_lines_with_a_tall_dock():
    """A dock taller than the screen must still yield exactly height rows."""
    t = Transcript()
    t.on_user_message("hi")
    layout = Layout(40, 6, color=False)
    from karen_coding_agent.tui.layout import _LayoutInput

    app = _LayoutInput()
    app.status = "status"
    app.editor_lines = tuple("abcde")
    app.footer = "footer"
    out = layout.render(t, app)
    assert len(out) == 6
    # the footer is the last row and must survive a tall dock
    assert "footer" in out[-1]


def test_pending_marker_never_overflows_the_width():
    t = Transcript()
    t.on_user_message("x" * 60)
    layout = Layout(40, 10, color=False)
    out = _lines(t, layout)
    for row in out:
        assert len(row) <= 40, row


def test_a_long_editor_row_is_truncated_to_the_width():
    t = Transcript()
    layout = Layout(30, 10, color=False)
    from karen_coding_agent.tui.layout import _LayoutInput

    app = _LayoutInput()
    app.editor_lines = ("y" * 200,)
    app.footer = "f"
    out = layout.render(t, app)
    assert len(out) == 10
    for row in out:
        assert len(row) <= 30, row


def test_a_notice_wraps_to_multiple_rows_without_truncation():
    t = Transcript()
    t.add_notice("word " * 30)
    layout = Layout(40, 12, color=False)
    out = _lines(t, layout)
    joined = "\n".join(out)
    # the tail of the long notice must be present, not silently dropped
    assert joined.count("word") > 10


def test_a_resize_does_not_teleport_a_scrolled_viewport():
    t = Transcript()
    for i in range(200):
        t.on_user_message(f"line {i}")
    layout = Layout(60, 10, color=False)
    _lines(t, layout)
    layout.scroll_up(50)
    before = layout.top
    layout.resize(60, 30)  # no transcript knowledge: must not clamp to tiny
    _lines(t, layout)
    assert layout.top >= min(before, 1)


def test_follow_sticks_to_the_latest_line():
    t = Transcript()
    layout = Layout(60, 10, color=False)
    for i in range(50):
        t.on_user_message(f"line {i}")
    out = _lines(t, layout)
    # following -> the last user line must be visible
    joined = "\n".join(out)
    assert "line 49" in joined


def test_scroll_up_stops_following_and_keeps_offset():
    t = Transcript()
    layout = Layout(60, 10, color=False)
    for i in range(40):
        t.on_user_message(f"line {i}")
    _lines(t, layout)  # follow to bottom
    layout.scroll_up(5)
    out = _lines(t, layout)
    joined = "\n".join(out)
    assert "line 49" not in joined
    assert layout.follow is False


def test_scroll_down_relocks_follow_at_bottom():
    t = Transcript()
    layout = Layout(60, 10, color=False)
    for i in range(40):
        t.on_user_message(f"line {i}")
    layout.scroll_up(20)
    layout.scroll_down(100)
    assert layout.follow is True


def test_user_line_is_wrapped_and_prefixed():
    t = Transcript()
    t.on_user_message("a" * 120)
    layout = Layout(40, 10, color=False)
    out = _lines(t, layout)
    joined = "\n".join(out)
    assert "❯" in joined
    # no single visible user row exceeds the width
    for row in out:
        assert len(row) <= 40, row


def test_pending_user_line_shows_a_pending_marker():
    t = Transcript()
    msg = t.on_user_message("waiting")
    layout = Layout(40, 10, color=False)
    out = _lines(t, layout)
    assert "(pending)" in "\n".join(out)


def test_no_color_emits_no_sgr_when_disabled():
    t = Transcript()
    t.on_user_message("hi")
    layout = Layout(40, 10, color=False)
    out = _lines(t, layout)
    joined = "\n".join(out)
    assert "\x1b[" not in joined


def test_a_bash_block_renders_command_output_and_exit_code():
    t = Transcript()
    msg = t.on_bash_start("false")
    t.on_bash_chunk(msg, "boom\n")
    t.on_bash_end(msg, _bash_result(output="boom\n", exit_code=1))
    layout = Layout(50, 10, color=False)
    joined = "\n".join(_lines(t, layout))
    assert "[bash <-] false" in joined
    assert "boom" in joined
    assert "exit 1" in joined


def test_a_running_bash_block_uses_the_running_label():
    t = Transcript()
    t.on_bash_start("sleep 5")
    layout = Layout(40, 8, color=False)
    joined = "\n".join(_lines(t, layout))
    assert "[bash ->] sleep 5" in joined
    assert "exit" not in joined


def test_a_silent_bash_block_renders_no_output_rows():
    t = Transcript()
    msg = t.on_bash_start("true")
    t.on_bash_end(msg, _bash_result(exit_code=0))
    layout = Layout(50, 8, color=False)
    assert layout.transcript_lines(t) == ["[bash <-] true"]


def test_a_trailing_newline_does_not_add_a_phantom_blank_row():
    t = Transcript()
    msg = t.on_bash_start("ls")
    t.on_bash_chunk(msg, "a.txt\nb.txt\n")
    t.on_bash_end(msg, _bash_result(output="a.txt\nb.txt\n"))
    layout = Layout(50, 8, color=False)
    assert layout.transcript_lines(t) == [
        "[bash <-] ls",
        "    a.txt",
        "    b.txt",
    ]


def test_a_successful_bash_block_prints_no_exit_code():
    t = Transcript()
    msg = t.on_bash_start("true")
    t.on_bash_end(msg, _bash_result(exit_code=0))
    layout = Layout(50, 8, color=False)
    joined = "\n".join(_lines(t, layout))
    assert "exit 0" not in joined


def test_a_long_bash_command_wraps_instead_of_overflowing():
    t = Transcript()
    t.on_bash_start("x" * 200)
    layout = Layout(30, 14, color=False)
    for row in _lines(t, layout):
        assert len(row) <= 30, row


def test_long_bash_output_rows_never_exceed_the_width():
    t = Transcript()
    msg = t.on_bash_start("cat log")
    t.on_bash_chunk(msg, ("y" * 300) + "\n")
    layout = Layout(40, 12, color=False)
    for row in _lines(t, layout):
        assert len(row) <= 40, row


def test_a_cancelled_bash_block_says_so():
    t = Transcript()
    msg = t.on_bash_start("sleep 100")
    t.on_bash_end(msg, _bash_result(exit_code=None, cancelled=True))
    layout = Layout(50, 8, color=False)
    joined = "\n".join(_lines(t, layout))
    assert "cancelled" in joined


def test_dropped_bash_lines_are_announced():
    t = Transcript()
    msg = t.on_bash_start("yes")
    for index in range(_BASH_MAX_LINES + 5):
        t.on_bash_chunk(msg, f"line {index}\n")
    layout = Layout(50, 10, color=False)
    # the block is taller than the viewport by construction, so assert on the
    # rendered transcript rather than on the scrolled window
    rows = layout.transcript_lines(t)
    assert any("5 earlier line(s) dropped" in row for row in rows)
    # the header and the tail of the output are both rendered
    assert rows[0].startswith("[bash ->] yes")
    assert any("line 404" in row for row in rows)


def test_a_truncated_bash_block_points_at_the_full_output():
    t = Transcript()
    msg = t.on_bash_start("yes")
    t.on_bash_end(msg, _bash_result(truncated=True, full_output_path="C:/tmp/full.txt"))
    layout = Layout(60, 8, color=False)
    joined = "\n".join(_lines(t, layout))
    assert "output truncated" in joined
    assert "C:/tmp/full.txt" in joined


# ---------------------------------------------------------------------------
# multi-line editor
# ---------------------------------------------------------------------------


def test_enter_submits_a_single_line():
    app, q, sink = _make_app()
    app.handle_key("h")
    app.handle_key("i")
    assert app.handle_key("enter") == "submit"
    # submit no longer clears the editor — the runner clears after reading
    assert app.editor_text == "hi"


def test_blank_enter_is_a_noop():
    app, q, sink = _make_app()
    assert app.handle_key("enter") == "noop"


def test_ctrl_enter_always_submits_even_single_line():
    app, q, sink = _make_app()
    app.handle_key("h")
    assert app.handle_key("ctrl_enter") == "submit"
    assert app.editor_text == "h"


def test_ctrl_j_inserts_the_first_newline():
    app, q, sink = _make_app()
    app.handle_key("a")
    assert app.handle_key("ctrl_j") == "noop"
    app.handle_key("b")
    assert app.editor_lines == ["a", "b"]
    assert app.editor_cursor == (1, 1)


def test_multiline_enter_inserts_a_newline():
    app, q, sink = _make_app()
    app.editor_lines = ["a", "b"]
    app.editor_cursor = (0, 1)
    assert app.handle_key("enter") == "noop"
    assert app.editor_lines == ["a", "", "b"]
    assert app.editor_cursor == (1, 0)


def test_multiline_text_is_joined_by_newlines():
    app, q, sink = _make_app()
    app.editor_lines = ["a", "b"]
    app.editor_cursor = (1, 1)
    assert app.editor_text == "a\nb"


def test_ctrl_k_kills_to_end_of_last_line():
    app, q, sink = _make_app()
    app.handle_key("h")
    app.handle_key("e")
    app.editor_cursor = (0, 2)
    assert app.handle_key("ctrl_k") == "noop"
    assert app.editor_lines == ["he"]


def test_ctrl_k_on_merges_remainder_into_next_line():
    app, q, sink = _make_app()
    app.editor_lines = ["hello", "world"]
    app.editor_cursor = (0, 2)
    assert app.handle_key("ctrl_k") == "noop"
    # kill "llo", fold it onto the next line; the next line survives
    assert app.editor_lines == ["he", "llo" + "world"]


def test_backspace_at_line_start_joins_previous_line():
    app, q, sink = _make_app()
    app.editor_lines = ["a", "b"]
    app.editor_cursor = (1, 1)
    app.handle_key("backspace")  # deletes 'b'
    assert app.handle_key("backspace") == "noop"
    assert app.editor_lines == ["a"]
    assert app.editor_cursor == (0, 1)


def test_ctrl_a_moves_to_line_start_and_wraps_with_left():
    app, q, sink = _make_app()
    app.handle_key("x")
    app.handle_key("left")
    assert app.editor_cursor == (0, 0)


def test_ctrl_u_clears_to_line_start():
    app, q, sink = _make_app()
    app.handle_key("abc")
    app.handle_key("ctrl_u")
    assert app.editor_lines[0] == ""
    assert app.editor_cursor == (0, 0)


def test_ctrl_d_on_empty_editor_quits():
    app, q, sink = _make_app()
    assert app.handle_key("ctrl_d") == "quit"


def test_ctrl_c_quits():
    app, q, sink = _make_app()
    assert app.handle_key("ctrl_c") == "quit"


def test_cursor_moving_across_lines_right():
    app, q, sink = _make_app()
    app.editor_lines = ["a", "b"]
    app.editor_cursor = (0, 1)
    app.handle_key("right")
    assert app.editor_cursor == (1, 0)


# ---------------------------------------------------------------------------
# terminal key decoding (pure decoder, no TTY)
# ---------------------------------------------------------------------------


def _byte_table():
    return TerminalController(sys.stdin, sys.stdout)._decode_byte


def test_byte_tab_ctrl_u_ctrl_o_are_distinct():
    table = _byte_table()
    assert table(9) == "tab"
    assert table(21) == "ctrl_u"
    assert table(15) == "ctrl_o"
    assert table(4) == "ctrl_d"


def test_byte_enter_and_backspace():
    table = _byte_table()
    assert table(13) == "enter"
    assert table(10) == "enter"
    assert table(127) == "backspace"
    assert table(3) == "ctrl_c"


def test_byte_ctrl_letter_maps_to_ctrl_x():
    table = _byte_table()
    assert table(1) == "ctrl_a"
    assert table(11) == "ctrl_k"
    assert table(12) == "ctrl_l"


def test_csi_plain_arrows():
    assert _decode_csi("A") == "up"
    assert _decode_csi("B") == "down"
    assert _decode_csi("C") == "right"
    assert _decode_csi("D") == "left"
    assert _decode_csi("H") == "home"
    assert _decode_csi("F") == "end"


def test_csi_modified_arrows():
    assert _decode_csi("1;5C") == "ctrl_right"
    assert _decode_csi("1;2A") == "shift_up"
    assert _decode_csi("1;3D") == "alt_left"


def test_csi_tilde_keys():
    assert _decode_csi("3~") == "delete"
    assert _decode_csi("5~") == "pageup"
    assert _decode_csi("6~") == "pagedown"
    assert _decode_csi("15~") == "f5"
    assert _decode_csi("21~") == "f10"
    assert _decode_csi("24~") == "f12"


def test_csi_unknown_and_empty():
    assert _decode_csi("") == "escape"
    assert _decode_csi("M") == "mouse"
    assert _decode_csi("ZZZ") == "unknown"


# ---------------------------------------------------------------------------
# the TTY gate (what decides TUI vs plain REPL)
# ---------------------------------------------------------------------------


class _Tty(io.StringIO):
    def isatty(self):
        return True


class _Pipe(io.StringIO):
    def isatty(self):
        return False


def test_supports_tty_is_false_when_a_stream_is_a_pipe(monkeypatch):
    import karen_coding_agent.tui.terminal as terminal

    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(sys, "stdout", _Pipe())
    assert terminal.supports_tty() is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console semantics")
def test_supports_tty_rejects_a_non_console_on_windows(monkeypatch):
    """On Windows `isatty()` is not enough — NUL is a character device — so the
    gate must consult the console check as well."""
    import karen_coding_agent.tui.terminal as terminal

    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(sys, "stdout", _Tty())
    monkeypatch.setattr(terminal, "_has_windows_console", lambda handle: False)
    assert terminal.supports_tty() is False
    monkeypatch.setattr(terminal, "_has_windows_console", lambda handle: True)
    assert terminal.supports_tty() is True


@pytest.mark.skipif(sys.platform != "win32", reason="Windows device semantics")
def test_the_nul_device_is_not_a_console():
    """The quirk the gate exists for: `karen < NUL > NUL` reports isatty() on
    both ends (so a console-less launch would enter the alt screen and park in
    `msvcrt.getwch()`, which has no EOF path) while GetConsoleMode says no."""
    import msvcrt

    from karen_coding_agent.tui.terminal import _is_console_handle

    descriptor = os.open("NUL", os.O_RDWR)
    try:
        assert os.isatty(descriptor) is True
        assert _is_console_handle(msvcrt.get_osfhandle(descriptor)) is False
    finally:
        os.close(descriptor)

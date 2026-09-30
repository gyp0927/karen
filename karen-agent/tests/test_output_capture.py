"""Bounded shell-output capture (pi's output-capture.ts behaviors, snapshot variant)."""

import asyncio

import pytest

from karen_agent.utils.output_capture import OutputCapture, sanitize_shell_output


def test_sanitize_shell_output_strips_control_chars():
    assert sanitize_shell_output("a\x00b\x07c\x1bd") == "abcd"
    # \x0d (\r) is inside pi's stripped \x0b-\x1f range; \t and \n survive.
    assert sanitize_shell_output("keep\nnew\ttabs\r") == "keep\nnew\ttabs"
    assert sanitize_shell_output("￹weird￻") == "weird"


async def test_capture_collects_and_snapshots():
    capture = OutputCapture()
    capture.push("hello ")
    capture.push(b"world\n")
    capture.finish()
    view = capture.snapshot()
    assert view.text == "hello world\n"  # untruncated content passes through verbatim
    assert view.truncation.truncated is False
    assert view.truncation.total_lines == 1
    assert view.truncation.total_bytes == 12
    assert view.spill_path is None


async def test_capture_incremental_utf8_decode():
    capture = OutputCapture()
    capture.push(b"\xc3")  # first half of é
    capture.push(b"\xa9")
    capture.finish()
    assert capture.snapshot().text == "é"


async def test_capture_truncated_flag_and_tail_snapshot():
    capture = OutputCapture(max_lines=3, max_bytes=10_000)
    for i in range(10):
        capture.push(f"line{i}\n")
    assert capture.truncated is True
    view = capture.snapshot()
    assert view.truncation.truncated is True
    assert view.truncation.truncated_by == "lines"
    assert view.truncation.total_lines == 10
    assert view.text == "line7\nline8\nline9"


async def test_capture_head_retention():
    capture = OutputCapture(max_lines=2, max_bytes=10_000, retain="head")
    for i in range(5):
        capture.push(f"line{i}\n")
    assert capture.snapshot().text == "line0\nline1"


async def test_capture_spill_path_in_view():
    capture = OutputCapture()
    capture.push("x")
    capture.set_spill_path("/tmp/spill.log")
    assert capture.snapshot().spill_path == "/tmp/spill.log"


async def test_capture_updates_first_immediate_then_coalesced():
    views = []
    capture = OutputCapture(on_update=views.append)
    capture.push("a")  # idle → immediate
    assert len(views) == 1
    capture.push("b")  # within min interval → deferred
    capture.push("c")
    assert len(views) == 1
    await asyncio.sleep(0.25)  # trailing timer publishes latest
    assert len(views) == 2
    assert views[-1].text == "abc"
    capture.dispose()


async def test_capture_flush_forces_pending_update():
    views = []
    capture = OutputCapture(on_update=views.append)
    capture.push("a")
    capture.push("b")
    capture.flush()
    assert len(views) == 2
    assert views[-1].text == "ab"
    capture.dispose()


async def test_capture_validates_limits():
    with pytest.raises(TypeError, match="Output maxBytes must be a positive finite number"):
        OutputCapture(max_bytes=0)
    with pytest.raises(TypeError, match="Output maxLines must be a positive integer"):
        OutputCapture(max_lines=0)


async def test_capture_sanitizes_view_text():
    capture = OutputCapture()
    capture.push(b"ok\x00\xff bad\n")
    assert capture.snapshot().text == "ok\ufffd bad\n"  # control stripped, invalid byte replaced


async def test_capture_last_line_bytes_reported_when_partial():
    capture = OutputCapture(max_bytes=10, max_lines=100)
    capture.push(b"x" * 60)
    capture.finish()
    view = capture.snapshot()
    assert view.truncation.last_line_partial is True
    assert view.last_line_bytes == 60

"""Truncation utilities (pi's truncate.ts behaviors)."""

from karen_agent.utils.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncate_head,
    truncate_line,
    truncate_tail,
    utf8_byte_length,
)


def test_constants_match_pi():
    assert DEFAULT_MAX_LINES == 2000
    assert DEFAULT_MAX_BYTES == 50 * 1024


def test_utf8_byte_length_counts_multibyte():
    assert utf8_byte_length("hello") == 5
    assert utf8_byte_length("héllo") == 6
    assert utf8_byte_length("日本語") == 9
    assert utf8_byte_length("🎉") == 4


def test_format_size():
    assert format_size(0) == "0B"
    assert format_size(512) == "512B"
    assert format_size(1024) == "1.0KB"
    assert format_size(50 * 1024) == "50.0KB"
    assert format_size(60000) == "58.6KB"
    assert format_size(3 * 1024 * 1024) == "3.0MB"


def test_head_no_truncation_passthrough():
    result = truncate_head("a\nb\nc\n")
    assert result.truncated is False
    assert result.truncated_by is None
    assert result.content == "a\nb\nc\n"
    assert result.total_lines == 3
    assert result.output_lines == 3
    assert result.total_bytes == result.output_bytes == 6
    assert result.last_line_partial is False
    assert result.first_line_exceeds_limit is False


def test_head_truncates_by_lines():
    result = truncate_head("l1\nl2\nl3\nl4", max_lines=2)
    assert result.truncated is True
    assert result.truncated_by == "lines"
    assert result.content == "l1\nl2"
    assert result.total_lines == 4
    assert result.output_lines == 2


def test_head_truncates_by_bytes_without_partial_lines():
    # 10 lines of 10 bytes each; byte limit cuts mid-file.
    content = "\n".join(f"line{i:05d}" for i in range(10))  # 10 bytes per line
    result = truncate_head(content, max_bytes=35)
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.content == "line00000\nline00001\nline00002"
    assert result.output_lines == 3


def test_head_multibyte_counts_bytes_not_chars():
    # "éé" is 2 chars but 4 bytes; with max_bytes=3 it does not fit after a 1-byte line.
    result = truncate_head("a\néé", max_bytes=3)
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.content == "a"


def test_head_first_line_exceeds_limit():
    result = truncate_head("x" * 100 + "\nsecond", max_bytes=50)
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.content == ""
    assert result.output_lines == 0
    assert result.first_line_exceeds_limit is True


def test_tail_no_truncation_passthrough():
    result = truncate_tail("a\nb")
    assert result.truncated is False
    assert result.content == "a\nb"
    assert result.total_lines == 2


def test_tail_truncates_by_lines_keeping_end():
    result = truncate_tail("l1\nl2\nl3\nl4", max_lines=2)
    assert result.truncated is True
    assert result.truncated_by == "lines"
    assert result.content == "l3\nl4"
    assert result.output_lines == 2


def test_tail_truncates_by_bytes():
    content = "\n".join(f"line{i:05d}" for i in range(10))
    result = truncate_tail(content, max_bytes=35)
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.content == "line00007\nline00008\nline00009"


def test_tail_partial_last_line_when_it_exceeds_limit():
    result = truncate_tail("x" * 100, max_bytes=10)
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.content == "x" * 10
    assert result.last_line_partial is True
    assert result.output_lines == 1
    assert result.output_bytes == 10


def test_tail_partial_line_is_utf8_safe():
    # é is 2 bytes: "a"*8 + "éé" is 12 bytes; the last 3 bytes start mid-codepoint.
    result = truncate_tail("a" * 8 + "éé", max_bytes=3)
    assert result.last_line_partial is True
    assert result.content == "é"  # mid-codepoint start skipped, not replacement garbage
    assert result.output_bytes == 2


def test_truncate_line():
    assert truncate_line("short") == ("short", False)
    text, was = truncate_line("x" * 600)
    assert was is True
    assert text == "x" * 500 + "... [truncated]"


def test_empty_content():
    for fn in (truncate_head, truncate_tail):
        result = fn("")
        assert result.truncated is False
        assert result.total_lines == 0
        assert result.content == ""

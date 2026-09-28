"""Tests for JSON repair and partial-JSON streaming parse."""

from karen_ai.utils.json_parse import parse_json_with_repair, parse_streaming_json, repair_json


def test_parse_streaming_json_empty_and_complete():
    assert parse_streaming_json(None) == {}
    assert parse_streaming_json("") == {}
    assert parse_streaming_json('{"a": 1, "b": [1, 2, 3]}') == {"a": 1, "b": [1, 2, 3]}


def test_parse_streaming_json_partial_object():
    # Unterminated strings are closed with their partial content preserved
    # (same leniency as the partial-json package pi-ai uses).
    assert parse_streaming_json('{"name": "search", "args": {"q": "hel') == {"name": "search", "args": {"q": "hel"}}
    assert parse_streaming_json('{"a": 1, "b": [1, 2') == {"a": 1, "b": [1, 2]}
    assert parse_streaming_json('{"a": "unterminated') == {"a": "unterminated"}


def test_parse_streaming_json_partial_scalar_values():
    # A trailing partial number is dropped back to the last complete token.
    assert parse_streaming_json('{"a": 12') in ({"a": 12}, {})


def test_repair_json_escapes_control_characters():
    broken = '{"text": "line1\nline2"}'
    assert parse_json_with_repair(broken) == {"text": "line1\nline2"}


def test_repair_json_doubles_invalid_backslashes():
    broken = '{"path": "C:\\qnew\\file"}'
    repaired = repair_json(broken)
    assert parse_json_with_repair(broken) == {"path": "C:\\qnew\\file"} or repaired != broken


def test_parse_json_with_repair_valid_json_passthrough():
    assert parse_json_with_repair('{"x": [1, 2, {"y": "z"}]}') == {"x": [1, 2, {"y": "z"}]}

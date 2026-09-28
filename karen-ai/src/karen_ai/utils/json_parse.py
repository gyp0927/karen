"""JSON repair and partial-JSON parsing for streaming tool-call arguments.

Mirrors pi-ai's utils/json-parse.ts: `repair_json` fixes malformed string
literals (raw control chars, invalid escapes); `parse_streaming_json` always
returns a valid object even for incomplete JSON.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

_VALID_JSON_ESCAPES = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}


def _is_control_character(char: str) -> bool:
    return 0x00 <= ord(char) <= 0x1F


def _escape_control_character(char: str) -> str:
    return {
        "\b": "\\b",
        "\f": "\\f",
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
    }.get(char, f"\\u{ord(char):04x}")


def repair_json(text: str) -> str:
    """Repair malformed JSON string literals by escaping raw control characters
    and doubling backslashes before invalid escape characters."""
    repaired: list[str] = []
    in_string = False
    index = 0
    n = len(text)

    while index < n:
        char = text[index]

        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue

        if char == '"':
            repaired.append(char)
            in_string = False
            index += 1
            continue

        if char == "\\":
            next_char = text[index + 1] if index + 1 < n else None
            if next_char is None:
                repaired.append("\\\\")
                index += 1
                continue
            if next_char == "u":
                unicode_digits = text[index + 2 : index + 6]
                if len(unicode_digits) == 4 and all(c in "0123456789abcdefABCDEF" for c in unicode_digits):
                    repaired.append(f"\\u{unicode_digits}")
                    index += 6
                    continue
            if next_char in _VALID_JSON_ESCAPES:
                repaired.append(f"\\{next_char}")
                index += 2
                continue
            repaired.append("\\\\")
            index += 1
            continue

        repaired.append(_escape_control_character(char) if _is_control_character(char) else char)
        index += 1

    return "".join(repaired)


def parse_json_with_repair(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_json(text)
        if repaired != text:
            return json.loads(repaired)
        raise


def _parse_partial_json(text: str) -> Any:
    """Leniently parse incomplete JSON.

    Scans once, recording a checkpoint (position + container-stack snapshot)
    after every complete token. Candidates are tried newest-first: text up to
    a checkpoint plus the closers for the stack at that checkpoint. If the scan
    ends inside a string, closing the string first is tried before anything else.
    """
    stack: list[str] = []  # '{' or '['
    in_string = False
    escaped = False
    checkpoints: list[tuple[int, list[str]]] = []  # (index_after_token, stack copy)

    closers = {"{": "}", "[": "]"}

    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                checkpoints.append((i + 1, list(stack)))
        else:
            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
                checkpoints.append((i + 1, list(stack)))
            elif char in "}]":
                if stack:
                    stack.pop()
                checkpoints.append((i + 1, list(stack)))
            elif not char.isspace() and char not in ",:":
                # Scalar characters (digits, true/false/null letters) are tentative:
                # a later character can extend the same token.
                checkpoints.append((i + 1, list(stack)))
        i += 1

    def try_parse(candidate: str):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return _FAILED

    # If the scan ended mid-string, try closing the string then the containers.
    if in_string:
        result = try_parse(text + '"' + "".join(closers[c] for c in reversed(stack)))
        if result is not _FAILED:
            return result

    for index, stack_snapshot in reversed(checkpoints):
        head = text[:index].rstrip()
        while head.endswith(","):
            head = head[:-1].rstrip()
        if not head:
            continue
        result = try_parse(head + "".join(closers[c] for c in reversed(stack_snapshot)))
        if result is not _FAILED:
            return result
    return {}


_FAILED = object()


def parse_streaming_json(partial_json: Optional[str]) -> Dict[str, Any]:
    """Parse potentially incomplete JSON during streaming.

    Always returns a valid object, even if the JSON is incomplete.
    """
    if not partial_json or partial_json.strip() == "":
        return {}

    try:
        result = parse_json_with_repair(partial_json)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        pass

    result = _parse_partial_json(partial_json)
    if isinstance(result, dict) and result:
        return result

    result = _parse_partial_json(repair_json(partial_json))
    return result if isinstance(result, dict) else {}

"""JSONL transaction framing and atomic file publication (pi's `jsonl/io.ts`).

Layout: one header line, then one line per commit. A commit with a single write
is stored as a bare object; multiple writes as an array. Appends are plain file
appends; snapshot rewrites (fork, torn-tail repair) go through
`publish_file_atomically` (write to `<path>.tmp`, then rename over the target).
"""

from __future__ import annotations

import json
import os
from typing import Any, Awaitable, Callable, Iterable, Iterator, List, Tuple

from ..commit import write_seq
from ..types import (
    EntryWrite,
    ListAppendWrite,
    ListDeleteWrite,
    UsageRow,
    UsageWrite,
    ValueDeleteWrite,
    ValueSetWrite,
    Write,
)
from .codec import entry_from_json, entry_to_json, to_jsonable
from .types import JsonlStorageHeader


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def split_complete_lines(content: str) -> Tuple[List[str], bool]:
    """Split into complete lines; `torn` when the file does not end with a newline."""
    if content.endswith("\n"):
        return content[:-1].split("\n"), False
    last_newline = content.rfind("\n")
    if last_newline == -1:
        return [], True
    return content[:last_newline].split("\n"), True


def iter_file_lines(path: str) -> Iterator[Tuple[str, bool]]:
    """Yield (text, terminated) per line; the final line may be unterminated."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.endswith("\n"):
                yield line[:-1], True
            else:
                yield line, False


def read_jsonl_header_line(line: Tuple[str, bool], path: str):
    """Parse a (text, terminated) first line into a parsed session header."""
    from .codec import parse_jsonl_session_header

    text, terminated = line
    if not terminated or text == "":
        raise ValueError(f"Invalid JSONL storage {path}: missing header")
    try:
        return parse_jsonl_session_header(text)
    except ValueError as error:
        raise ValueError(f"Invalid JSONL storage {path}: invalid header") from error


def _require_safe_integer(value: Any, field: str, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"Invalid JSONL {field}")


def parse_committed_write(value: Any) -> Write:
    if not isinstance(value, dict):
        raise ValueError("Invalid JSONL transaction write")
    _require_safe_integer(value.get("seq"), "write seq", 1)
    kind = value.get("kind")
    if kind == "entry":
        _require_safe_integer(value.get("timestamp"), "entry timestamp", 0)
        data = {key: item for key, item in value.items() if key != "kind"}
        return EntryWrite(entry=entry_from_json(data))
    if kind == "usage":
        data = {key: item for key, item in value.items() if key != "kind"}
        return UsageWrite(row=UsageRow.model_validate(data))
    if kind == "value":
        op = value.get("op")
        if op == "set":
            return ValueSetWrite.model_validate(value)
        if op == "delete":
            return ValueDeleteWrite.model_validate(value)
        raise ValueError(f"Invalid JSONL value operation: {op}")
    if kind == "list":
        op = value.get("op")
        if op == "append":
            return ListAppendWrite.model_validate(value)
        if op == "delete":
            return ListDeleteWrite.model_validate(value)
        raise ValueError(f"Invalid JSONL list operation: {op}")
    raise ValueError(f"Invalid JSONL write kind: {kind}")


def parse_jsonl_transaction(line: str) -> List[Write]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError("Invalid JSONL transaction: not valid JSON") from error
    items = value if isinstance(value, list) else [value]
    return [parse_committed_write(item) for item in items]


def write_to_json(write: Write) -> dict:
    """Flatten a committed write to its JSONL object form (`kind` folded in)."""
    if isinstance(write, EntryWrite):
        return {"kind": "entry", **entry_to_json(write.entry)}
    if isinstance(write, UsageWrite):
        return {"kind": "usage", **write.row.model_dump(mode="json", by_alias=True, exclude_none=True)}
    data = write.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(write, (ValueSetWrite, ListAppendWrite)):
        data["value"] = to_jsonable(write.value)
    return data


def serialize_jsonl_transaction(writes: List[Write]) -> str:
    if len(writes) == 1:
        return _dumps(write_to_json(writes[0]))
    return _dumps([write_to_json(write) for write in writes])


def header_to_json(header: JsonlStorageHeader) -> str:
    return _dumps(header.model_dump(mode="json", by_alias=True, exclude_none=True))


async def publish_file_atomically(
    destination_path: str,
    write_content: Callable[[Callable[[str], Awaitable[None]]], Awaitable[None]],
) -> None:
    """Publish only after the callback succeeds; it must await each append before returning."""
    temp_path = f"{destination_path}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8", newline="") as handle:

            async def append(content: str) -> None:
                handle.write(content)

            await write_content(append)
        os.replace(temp_path, destination_path)
    except BaseException:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


async def publish_jsonl(
    destination_path: str,
    header: JsonlStorageHeader,
    write_transactions: Callable[[Callable[[List[Write]], Awaitable[None]]], Awaitable[None]],
) -> None:
    """Stream a header and complete transactions through the shared atomic publisher."""

    async def body(append: Callable[[str], Awaitable[None]]) -> None:
        await append(header_to_json(header) + "\n")

        async def append_transaction(writes: List[Write]) -> None:
            await append(serialize_jsonl_transaction(writes) + "\n")

        await write_transactions(append_transaction)

    await publish_file_atomically(destination_path, body)


def write_sequence(write: Write) -> int:
    """Re-export of commit.write_seq for fork boundary checks."""
    return write_seq(write)

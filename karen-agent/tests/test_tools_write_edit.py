"""The write and edit tools (pi's write.ts / edit.ts behaviors)."""

import asyncio

import pytest
from karen_ai import AbortController

from karen_agent.tools import create_edit_tool, create_write_tool, prepare_edit_arguments
from karen_agent.tools.file_mutation_queue import with_file_mutation_queue


async def _run(tool, params, signal=None):
    return await tool.execute("call-1", params, signal, None)


# --- write -------------------------------------------------------------------


async def test_write_creates_file_and_parents(tmp_path):
    result = await _run(create_write_tool(str(tmp_path)), {"path": "sub/dir/f.txt", "content": "hello"})
    assert result.content[0].text == "Successfully wrote to sub/dir/f.txt"
    assert (tmp_path / "sub/dir/f.txt").read_bytes() == b"hello"


async def test_write_overwrites_without_newline_translation(tmp_path):
    target = tmp_path / "f.txt"
    target.write_bytes(b"old")
    await _run(create_write_tool(str(tmp_path)), {"path": "f.txt", "content": "a\nb\n"})
    assert target.read_bytes() == b"a\nb\n"  # no \r\n translation on Windows


async def test_write_aborted(tmp_path):
    controller = AbortController()
    controller.abort()
    with pytest.raises(RuntimeError, match="^Operation aborted$"):
        await _run(
            create_write_tool(str(tmp_path)), {"path": "f.txt", "content": "x"}, signal=controller.signal
        )
    assert not (tmp_path / "f.txt").exists()


# --- edit --------------------------------------------------------------------


async def test_edit_replaces_unique_text(tmp_path):
    target = tmp_path / "f.txt"
    target.write_bytes(b"hello world\ngoodbye\n")
    result = await _run(
        create_edit_tool(str(tmp_path)),
        {"path": "f.txt", "edits": [{"oldText": "world", "newText": "there"}]},
    )
    assert result.content[0].text == "Successfully replaced 1 block(s) in f.txt."
    assert target.read_bytes() == b"hello there\ngoodbye\n"
    assert result.details["firstChangedLine"] == 1
    assert "Index: f.txt" in result.details["patch"]
    assert "-1 hello world" in result.details["diff"]


async def test_edit_multiple_blocks(tmp_path):
    target = tmp_path / "f.txt"
    target.write_bytes(b"a=1\nb=2\nc=3\n")
    result = await _run(
        create_edit_tool(str(tmp_path)),
        {"path": "f.txt", "edits": [{"oldText": "a=1", "newText": "a=10"}, {"oldText": "c=3", "newText": "c=30"}]},
    )
    assert result.content[0].text == "Successfully replaced 2 block(s) in f.txt."
    assert target.read_bytes() == b"a=10\nb=2\nc=30\n"


async def test_edit_preserves_crlf_and_bom(tmp_path):
    target = tmp_path / "f.txt"
    target.write_bytes("﻿line1\r\nold\r\nline3\r\n".encode("utf-8"))
    await _run(
        create_edit_tool(str(tmp_path)), {"path": "f.txt", "edits": [{"oldText": "old", "newText": "new"}]}
    )
    assert target.read_bytes() == "﻿line1\r\nnew\r\nline3\r\n".encode("utf-8")


async def test_edit_fuzzy_matches_smart_quotes(tmp_path):
    target = tmp_path / "f.txt"
    target.write_bytes('say “hi”\nkeep “y”\n'.encode("utf-8"))
    await _run(
        create_edit_tool(str(tmp_path)),
        {"path": "f.txt", "edits": [{"oldText": 'say "hi"', "newText": 'say "bye"'}]},
    )
    assert target.read_bytes() == 'say "bye"\nkeep “y”\n'.encode("utf-8")


async def test_edit_empty_edits_invalid(tmp_path):
    (tmp_path / "f.txt").write_bytes(b"x")
    with pytest.raises(ValueError) as excinfo:
        await _run(create_edit_tool(str(tmp_path)), {"path": "f.txt", "edits": []})
    assert str(excinfo.value) == "Edit tool input is invalid. edits must contain at least one replacement."


async def test_edit_directory_path(tmp_path):
    (tmp_path / "sub").mkdir()
    with pytest.raises(RuntimeError) as excinfo:
        await _run(
            create_edit_tool(str(tmp_path)),
            {"path": "sub", "edits": [{"oldText": "a", "newText": "b"}]},
        )
    assert str(excinfo.value) == "Could not edit file: sub. Path is not a file."


async def test_edit_missing_file_error_code(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        await _run(
            create_edit_tool(str(tmp_path)),
            {"path": "missing.txt", "edits": [{"oldText": "a", "newText": "b"}]},
        )
    assert str(excinfo.value) == "Could not edit file: missing.txt. Error code: ENOENT."


async def test_edit_duplicate_and_not_found(tmp_path):
    (tmp_path / "f.txt").write_bytes(b"x\nx\n")
    with pytest.raises(ValueError, match="Found 2 occurrences of the text"):
        await _run(create_edit_tool(str(tmp_path)), {"path": "f.txt", "edits": [{"oldText": "x", "newText": "y"}]})
    with pytest.raises(ValueError, match="Could not find the exact text"):
        await _run(create_edit_tool(str(tmp_path)), {"path": "f.txt", "edits": [{"oldText": "zzz", "newText": "y"}]})


# --- prepare_edit_arguments ----------------------------------------------------


def test_prepare_arguments_parses_json_string_edits():
    out = prepare_edit_arguments({"path": "f", "edits": '[{"oldText": "a", "newText": "b"}]'})
    assert out["edits"] == [{"oldText": "a", "newText": "b"}]


def test_prepare_arguments_wraps_single_edit_object():
    out = prepare_edit_arguments({"path": "f", "edits": {"oldText": "a", "newText": "b"}})
    assert out["edits"] == [{"oldText": "a", "newText": "b"}]


def test_prepare_arguments_legacy_top_level_old_new_text():
    out = prepare_edit_arguments({"path": "f", "oldText": "a", "newText": "b"})
    assert out == {"path": "f", "edits": [{"oldText": "a", "newText": "b"}]}


def test_prepare_arguments_legacy_merges_with_existing_edits():
    out = prepare_edit_arguments(
        {"path": "f", "edits": [{"oldText": "x", "newText": "y"}], "oldText": "a", "newText": "b"}
    )
    assert out["edits"] == [{"oldText": "x", "newText": "y"}, {"oldText": "a", "newText": "b"}]
    assert "oldText" not in out


def test_prepare_arguments_leaves_non_dicts_and_bad_json_alone():
    assert prepare_edit_arguments("nope") == "nope"
    out = prepare_edit_arguments({"path": "f", "edits": "[not json"})
    assert out["edits"] == "[not json"


# --- file mutation queue -------------------------------------------------------


async def test_mutation_queue_serializes_same_path(tmp_path):
    running = False

    async def mutation():
        nonlocal running
        assert not running  # would trip if two mutations overlapped
        running = True
        await asyncio.sleep(0.01)
        running = False

    await asyncio.gather(*(with_file_mutation_queue(str(tmp_path / "f.txt"), mutation) for _ in range(8)))


async def test_mutation_queue_allows_different_paths_in_parallel(tmp_path):
    started = asyncio.Event()
    other_started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(0.05)

    async def fast():
        other_started.set()

    async def run_both():
        await asyncio.gather(
            with_file_mutation_queue(str(tmp_path / "a.txt"), slow),
            with_file_mutation_queue(str(tmp_path / "b.txt"), fast),
        )

    await asyncio.wait_for(run_both(), timeout=5)
    assert started.is_set() and other_started.is_set()

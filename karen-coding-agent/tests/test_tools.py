"""Tests for the application-level tools: globs, the ignore-aware walker,
find/grep/ls against tmp trees, powershell (Windows), and the AgentSession
integration (default tool set + a faux-provider tool call through the loop).
"""

import os
import sys

import pytest

from karen_ai.providers import faux_assistant_message, faux_model, faux_tool_call, register_faux_provider
from karen_ai import create_models
from karen_coding_agent import AgentSession
from karen_coding_agent.tools import create_default_tools, create_find_tool, create_grep_tool, create_ls_tool
from karen_coding_agent.tools.globs import compile_find_matcher, compile_grep_glob_matcher, translate_glob
from karen_coding_agent.tools.walk import walk


# ---------------------------------------------------------------------------
# globs
# ---------------------------------------------------------------------------


def test_translate_glob_star_stays_within_segment():
    regex = translate_glob("*.py")
    assert regex.fullmatch("a.py")
    assert not regex.fullmatch("src/a.py")
    assert not regex.fullmatch("a.pyc")


def test_translate_glob_double_star_crosses_segments():
    assert translate_glob("**/x").fullmatch("x")
    assert translate_glob("**/x").fullmatch("a/b/x")
    assert translate_glob("a/**").fullmatch("a/")
    assert translate_glob("a/**").fullmatch("a/b/c")
    assert not translate_glob("a/**").fullmatch("ab/c")


def test_translate_glob_question_mark_and_classes():
    assert translate_glob("?").fullmatch("a")
    assert not translate_glob("?").fullmatch("/")
    assert translate_glob("[abc]x").fullmatch("bx")
    assert not translate_glob("[abc]x").fullmatch("dx")
    assert translate_glob("[!abc]x").fullmatch("dx")
    assert not translate_glob("[!abc]x").fullmatch("ax")


def test_translate_glob_escapes_regex_chars():
    assert translate_glob("file.txt").fullmatch("file.txt")
    assert not translate_glob("file.txt").fullmatch("filextxt")
    # unterminated class is a literal "["
    assert translate_glob("a[").fullmatch("a[")


def test_find_matcher_basename_when_no_slash():
    matcher = compile_find_matcher("*.py")
    assert matcher("src/a.py", False)
    assert matcher("a.py", False)
    assert not matcher("src/a.pyc", False)


def test_find_matcher_full_path_gets_auto_prefix():
    matcher = compile_find_matcher("src/*.py")
    assert matcher("src/a.py", False)
    assert matcher("deep/nested/src/a.py", False)  # implicit **/ prefix
    assert not matcher("src/lib/a.py", False)


def test_find_matcher_leading_slash_anchors_at_root():
    matcher = compile_find_matcher("/src/*.py")
    assert matcher("src/a.py", False)
    assert not matcher("deep/src/a.py", False)


def test_find_matcher_double_star_matches_everything():
    matcher = compile_find_matcher("**")
    assert matcher("anything/at/all.txt", False)
    assert matcher("dir", True)


def test_grep_glob_anchored_when_slash_present():
    assert compile_grep_glob_matcher("*.py")("src/a.py")
    assert compile_grep_glob_matcher("src/*.py")("src/a.py")
    assert not compile_grep_glob_matcher("src/*.py")("deep/src/a.py")


# ---------------------------------------------------------------------------
# fixture tree + walker
# ---------------------------------------------------------------------------


def _build_tree(root):
    """The shared fixture tree (ignore rules exercise every walker branch)."""
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("hello git\n")
    (root / ".gitignore").write_text("build/\n*.log\n!keep.log\nsecret.txt\nprecedence.txt\n")
    (root / ".ignore").write_text("gone.txt\n!precedence.txt\n")
    (root / ".hidden.txt").write_text("secret\n")
    (root / "README.md").write_text("readme\n")
    (root / "bin.bin").write_bytes(b"hello bin\x00rest hello\n")
    (root / "keep.log").write_text("hello keep\n")
    (root / "debug.log").write_text("hello debug\n")
    (root / "gone.txt").write_text("hello gone\n")
    (root / "precedence.txt").write_text("hello precedence\n")
    (root / "long.txt").write_text("x" * 600 + " hello\n")
    (root / "empty.txt").write_text("")
    (root / "build").mkdir()
    (root / "build" / "out.o").write_text("hello build\n")
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("def main():\n    print('hello world')\n")
    (root / "src" / "lib").mkdir()
    (root / "src" / "lib" / "util.py").write_text("def util():\n    return 'hello'\n")
    (root / "src" / "lib" / "data.txt").write_text("alpha\nbeta\ngamma\n")
    (root / "sub").mkdir()
    (root / "sub" / ".gitignore").write_text("!secret.txt\n")
    (root / "sub" / "nested.txt").write_text("hello nested\n")
    (root / "sub" / "secret.txt").write_text("hello secret\n")


@pytest.fixture
def tree(tmp_path):
    _build_tree(tmp_path)
    return tmp_path


def _rel_paths(root, **kwargs):
    return [entry.rel_path for entry in walk(root, **kwargs)]


def test_walk_prunes_git_and_ignored(tree):
    paths = _rel_paths(tree)
    assert ".git" not in paths
    assert not any(path.startswith(".git/") for path in paths)
    assert "build" not in paths  # dir pattern prunes the whole subtree
    assert "debug.log" not in paths
    assert "gone.txt" not in paths  # .ignore-only rule


def test_walk_negations_and_precedence(tree):
    paths = _rel_paths(tree)
    assert "keep.log" in paths  # negation inside one .gitignore
    assert "precedence.txt" in paths  # .ignore outranks .gitignore in the same dir
    assert "sub/secret.txt" in paths  # deepest .gitignore wins over the root one


def test_walk_includes_hidden_and_is_deterministic(tree):
    paths = _rel_paths(tree)
    assert ".hidden.txt" in paths
    assert ".gitignore" in paths
    assert paths == _rel_paths(tree)


def test_walk_include_dirs(tree):
    paths = _rel_paths(tree, include_dirs=True)
    assert "src" in paths
    assert "src/lib" in paths
    assert "build" not in paths  # still pruned


def test_walk_does_not_descend_symlinks(tree):
    link = tree / "link_dir"
    try:
        os.symlink(tree / "src", link, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation requires privileges on this machine")
    entries = {entry.rel_path: entry for entry in walk(tree)}
    assert "link_dir" in entries
    assert not entries["link_dir"].is_dir
    assert "link_dir/main.py" not in entries


# ---------------------------------------------------------------------------
# find
# ---------------------------------------------------------------------------


async def _run(tool, params):
    return await tool.execute("call-1", params, None, None)


async def test_find_basename_pattern(tree):
    result = await _run(create_find_tool(str(tree)), {"pattern": "*.py"})
    assert result.content[0].text == "src/lib/util.py\nsrc/main.py"
    assert result.details is None


async def test_find_full_path_pattern(tree):
    result = await _run(create_find_tool(str(tree)), {"pattern": "src/*.py"})
    assert result.content[0].text == "src/main.py"


async def test_find_respects_gitignore(tree):
    result = await _run(create_find_tool(str(tree)), {"pattern": "**"})
    text = result.content[0].text
    assert ".git/" not in text  # the .git directory itself is pruned
    assert "build/" not in text
    assert "debug.log" not in text
    assert "keep.log" in text
    assert "src/" in text  # directories carry a trailing slash


async def test_find_limit_notice_and_details(tree):
    result = await _run(create_find_tool(str(tree)), {"pattern": "**", "limit": 2})
    text = result.content[0].text
    assert len(text.splitlines()[:2]) == 2
    assert "[2 results limit reached. Use limit=4 for more, or refine pattern]" in text
    assert result.details["resultLimitReached"] == 2


async def test_find_no_match(tree):
    result = await _run(create_find_tool(str(tree)), {"pattern": "*.rs"})
    assert result.content[0].text == "No files found matching pattern"
    assert result.details is None


async def test_find_path_not_found(tree):
    with pytest.raises(RuntimeError, match="Path not found"):
        await _run(create_find_tool(str(tree)), {"pattern": "**", "path": "nope"})


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


async def test_grep_basic_match_format(tree):
    result = await _run(create_grep_tool(str(tree)), {"pattern": "hello world"})
    assert result.content[0].text == "src/main.py:2:     print('hello world')"
    assert result.details is None


async def test_grep_ignore_case_and_literal(tree):
    result = await _run(create_grep_tool(str(tree)), {"pattern": "HELLO WORLD", "ignoreCase": True})
    assert "src/main.py:2:" in result.content[0].text
    result = await _run(create_grep_tool(str(tree)), {"pattern": "print('hello", "literal": True})
    assert "src/main.py:2:" in result.content[0].text


async def test_grep_context_lines(tree):
    result = await _run(create_grep_tool(str(tree)), {"pattern": "beta", "context": 1})
    assert result.content[0].text == (
        "src/lib/data.txt-1- alpha\nsrc/lib/data.txt:2: beta\nsrc/lib/data.txt-3- gamma"
    )


async def test_grep_glob_filter(tree):
    result = await _run(create_grep_tool(str(tree)), {"pattern": "hello", "glob": "*.py"})
    text = result.content[0].text
    assert "src/main.py:2:" in text
    assert "src/lib/util.py:2:" in text
    assert "nested.txt" not in text
    assert "bin.bin" not in text


async def test_grep_limit_notice_and_details(tree):
    result = await _run(create_grep_tool(str(tree)), {"pattern": "hello", "limit": 2})
    text = result.content[0].text
    assert text == (
        "bin.bin:1: hello bin\nkeep.log:1: hello keep\n\n"
        "[2 matches limit reached. Use limit=4 for more, or refine pattern]"
    )
    assert result.details["matchLimitReached"] == 2


async def test_grep_binary_search_stops_at_nul(tree):
    result = await _run(create_grep_tool(str(tree)), {"pattern": "hello bin"})
    assert result.content[0].text == "bin.bin:1: hello bin"
    result = await _run(create_grep_tool(str(tree)), {"pattern": "rest hello"})
    assert result.content[0].text == "No matches found"  # past the NUL byte


async def test_grep_long_lines_truncated(tree):
    result = await _run(create_grep_tool(str(tree)), {"pattern": "hello", "glob": "long.txt"})
    line = result.content[0].text.splitlines()[0]
    assert line.startswith("long.txt:1: " + "x" * 100)
    assert line.endswith("... [truncated]")
    assert "[Some lines truncated to 500 chars. Use read tool to see full lines]" in result.content[0].text
    assert result.details["linesTruncated"] is True


async def test_grep_explicit_file_uses_basename(tree):
    result = await _run(create_grep_tool(str(tree)), {"pattern": "readme", "path": "README.md"})
    assert result.content[0].text == "README.md:1: readme"


async def test_grep_errors(tree):
    with pytest.raises(RuntimeError, match="Path not found"):
        await _run(create_grep_tool(str(tree)), {"pattern": "x", "path": "nope"})
    with pytest.raises(RuntimeError, match="Invalid regular expression"):
        await _run(create_grep_tool(str(tree)), {"pattern": "("})


async def test_grep_empty_file_yields_no_phantom_match(tree):
    result = await _run(create_grep_tool(str(tree)), {"pattern": "^", "glob": "empty.txt"})
    assert result.content[0].text == "No matches found"


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


async def test_ls_sorted_with_dir_suffixes(tree):
    result = await _run(create_ls_tool(str(tree)), {})
    assert result.content[0].text == (
        ".git/\n.gitignore\n.hidden.txt\n.ignore\nbin.bin\nbuild/\ndebug.log\nempty.txt\n"
        "gone.txt\nkeep.log\nlong.txt\nprecedence.txt\nREADME.md\nsrc/\nsub/"
    )
    assert result.details is None


async def test_ls_empty_directory(tree):
    (tree / "blank").mkdir()
    result = await _run(create_ls_tool(str(tree)), {"path": "blank"})
    assert result.content[0].text == "(empty directory)"


async def test_ls_limit_notice_and_details(tree):
    result = await _run(create_ls_tool(str(tree)), {"limit": 3})
    text = result.content[0].text
    assert text.startswith(".git/\n.gitignore\n.hidden.txt")
    assert "[3 entries limit reached. Use limit=6 for more]" in text
    assert result.details["entryLimitReached"] == 3


async def test_ls_errors(tree):
    with pytest.raises(RuntimeError, match="Path not found"):
        await _run(create_ls_tool(str(tree)), {"path": "nope"})
    with pytest.raises(RuntimeError, match="Not a directory"):
        await _run(create_ls_tool(str(tree)), {"path": "README.md"})


# ---------------------------------------------------------------------------
# powershell (Windows only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="the powershell tool is Windows-only")
async def test_powershell_tool_runs_command(tmp_path):
    from karen_coding_agent.tools.powershell import (
        POWERSHELL_DESCRIPTION,
        create_powershell_tool,
    )

    tool = create_powershell_tool(str(tmp_path))
    assert tool.name == "powershell"
    assert tool.description == POWERSHELL_DESCRIPTION
    assert tool.parameters["properties"]["command"]["description"] == "PowerShell command to execute"
    result = await _run(tool, {"command": "Write-Output hello"})
    assert "hello" in result.content[0].text


@pytest.mark.skipif(sys.platform != "win32", reason="the powershell tool is Windows-only")
async def test_powershell_tool_utf8_output(tmp_path):
    from karen_coding_agent.tools.powershell import create_powershell_tool

    tool = create_powershell_tool(str(tmp_path))
    # "€" is not in the legacy ANSI codepages — it only survives if the
    # UTF8_OUTPUT_PREFIX actually switched the console output encoding.
    result = await _run(tool, {"command": "Write-Output '€ héllo 中文'"})
    assert "€ héllo 中文" in result.content[0].text


# ---------------------------------------------------------------------------
# AgentSession integration
# ---------------------------------------------------------------------------


EXPECTED_TOOL_ORDER_WINDOWS = ["read", "bash", "powershell", "edit", "write", "grep", "find", "ls"]
EXPECTED_TOOL_ORDER_POSIX = ["read", "bash", "edit", "write", "grep", "find", "ls"]


async def test_default_tools_match_pi_set_and_order(tmp_path):
    models = create_models()
    registration = register_faux_provider(models=[faux_model()], responses=[])
    models.set_provider(registration.provider)
    session = AgentSession(
        cwd=str(tmp_path),
        models=models,
        model=registration.get_model(),
        sessions_root=str(tmp_path / "sessions"),
        fresh=True,
    )
    await session.open()
    expected = EXPECTED_TOOL_ORDER_WINDOWS if sys.platform == "win32" else EXPECTED_TOOL_ORDER_POSIX
    assert [tool.name for tool in session.tools] == expected
    assert "find/grep/ls" in session.system_prompt_text
    await session.close()


async def test_find_tool_call_through_the_loop(tree):
    models = create_models()
    registration = register_faux_provider(
        models=[faux_model()],
        responses=[
            faux_assistant_message([faux_tool_call("find", {"pattern": "*.py"})]),
            faux_assistant_message("done"),
        ],
    )
    models.set_provider(registration.provider)
    session = AgentSession(
        cwd=str(tree),
        models=models,
        model=registration.get_model(),
        sessions_root=str(tree / "sessions"),
        fresh=True,
    )
    await session.open()

    await session.prompt("find the python files")

    roles = [message.role for message in session.agent.state.messages]
    assert roles == ["system", "user", "assistant", "toolResult", "assistant"]
    tool_result = session.agent.state.messages[3]
    assert "src/main.py" in tool_result.content[0].text
    assert session.agent.state.messages[4].content[0].text == "done"
    await session.close()


def test_create_default_tools_omits_powershell_off_windows(tree):
    names = [tool.name for tool in create_default_tools(str(tree))]
    expected = EXPECTED_TOOL_ORDER_WINDOWS if sys.platform == "win32" else EXPECTED_TOOL_ORDER_POSIX
    assert names == expected

"""Prompt template loading and substitution (pi's prompt-templates.ts)."""

import pytest

from karen_agent.prompt_templates import (
    SourcedPath,
    format_prompt_template_invocation,
    load_prompt_templates,
    load_sourced_prompt_templates,
    parse_command_args,
    substitute_args,
)
from karen_agent.resources import PromptTemplate


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_from_directory_non_recursive_sorted(tmp_path):
    _write(tmp_path / "b.md", "second")
    _write(tmp_path / "a.md", "first")
    _write(tmp_path / "skip.txt", "not markdown")
    _write(tmp_path / "nested" / "inner.md", "nested template")  # subdir not loaded
    result = load_prompt_templates(str(tmp_path))
    assert result.diagnostics == []
    assert [t.name for t in result.prompt_templates] == ["a", "b"]
    assert result.prompt_templates[0].content == "first"


def test_load_single_file_and_skip(tmp_path):
    _write(tmp_path / "one.md", "content here")
    result = load_prompt_templates(str(tmp_path / "one.md"))
    assert [t.name for t in result.prompt_templates] == ["one"]
    # non-markdown file and missing path are skipped silently
    _write(tmp_path / "two.txt", "nope")
    assert load_prompt_templates(str(tmp_path / "two.txt")).prompt_templates == []
    missing = load_prompt_templates(str(tmp_path / "missing.md"))
    assert missing.prompt_templates == []
    assert missing.diagnostics == []


def test_frontmatter_description_and_body(tmp_path):
    _write(
        tmp_path / "t.md",
        "---\ndescription: Does a thing\nargument-hint: <file>\n---\n\nBody line one\n\nBody line two\n",
    )
    [template] = load_prompt_templates(str(tmp_path)).prompt_templates
    assert template.description == "Does a thing"
    assert template.content == "Body line one\n\nBody line two"


def test_description_falls_back_to_first_line(tmp_path):
    _write(tmp_path / "t.md", "Short description line\n\nrest")
    [template] = load_prompt_templates(str(tmp_path)).prompt_templates
    assert template.description == "Short description line"

    long_line = "x" * 70
    _write(tmp_path / "long.md", long_line)
    [long_template] = load_prompt_templates(str(tmp_path / "long.md")).prompt_templates
    assert long_template.description == "x" * 60 + "..."


def test_crlf_normalized(tmp_path):
    _write(tmp_path / "t.md", "---\r\ndescription: crlf\r\n---\r\nbody")
    [template] = load_prompt_templates(str(tmp_path)).prompt_templates
    assert template.description == "crlf"
    assert template.content == "body"


def test_no_frontmatter_and_unterminated(tmp_path):
    _write(tmp_path / "plain.md", "just content")
    [plain] = load_prompt_templates(str(tmp_path / "plain.md")).prompt_templates
    assert plain.description == "just content"
    assert plain.content == "just content"

    _write(tmp_path / "open.md", "---\ndescription: never closed")
    [open_template] = load_prompt_templates(str(tmp_path / "open.md")).prompt_templates
    # unterminated frontmatter: whole file is body
    assert open_template.content == "---\ndescription: never closed"


def test_invalid_frontmatter_is_a_parse_diagnostic(tmp_path):
    _write(tmp_path / "bad.md", "---\nkey: [unclosed\n---\nbody")
    result = load_prompt_templates(str(tmp_path))
    assert result.prompt_templates == []
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].type == "warning"
    assert result.diagnostics[0].code == "parse_failed"
    assert result.diagnostics[0].path.endswith("bad.md")


def test_name_strips_md_case_insensitively(tmp_path):
    _write(tmp_path / "UPPER.md", "x")
    [template] = load_prompt_templates(str(tmp_path)).prompt_templates
    assert template.name == "UPPER"
    # .MD files are not matched by the (case-sensitive) file filter
    _write(tmp_path / "caps.MD", "x")
    names = [t.name for t in load_prompt_templates(str(tmp_path)).prompt_templates]
    assert "caps" not in names


def test_load_sourced_prompt_templates(tmp_path):
    _write(tmp_path / "a.md", "from A")
    _write(tmp_path / "bad.md", "---\nkey: [unclosed\n---\nx")
    result = load_sourced_prompt_templates(
        [SourcedPath(path=str(tmp_path), source="project")],
    )
    assert [(t.prompt_template.name, t.source) for t in result.prompt_templates] == [("a", "project")]
    assert [(d.code, d.source) for d in result.diagnostics] == [("parse_failed", "project")]

    mapped = load_sourced_prompt_templates(
        [SourcedPath(path=str(tmp_path), source="project")],
        map_prompt_template=lambda template, source: PromptTemplate(
            name=f"{source}/{template.name}", description=template.description, content=template.content
        ),
    )
    assert mapped.prompt_templates[0].prompt_template.name == "project/a"


def test_parse_command_args():
    assert parse_command_args("") == []
    assert parse_command_args("a b  c") == ["a", "b", "c"]
    assert parse_command_args("a\tb") == ["a", "b"]
    assert parse_command_args('say "hello world" again') == ["say", "hello world", "again"]
    assert parse_command_args("ab 'c d' e") == ["ab", "c d", "e"]
    assert parse_command_args("it's 'quoted'") == ["its quoted"]  # apostrophe opens a quote
    assert parse_command_args('"unclosed') == ["unclosed"]
    assert parse_command_args('a "b c" d') == ["a", "b c", "d"]


def test_substitute_args():
    assert substitute_args("$1 $2 $3", ["a", "b"]) == "a b "
    assert substitute_args("$0 $1", ["a"]) == " a"  # $0 -> index -1 -> ""
    assert substitute_args("$@", ["a", "b"]) == "a b"
    assert substitute_args("$ARGUMENTS!", ["a", "b"]) == "a b!"
    assert substitute_args("${@:2}", ["a", "b", "c"]) == "b c"
    assert substitute_args("${@:1:2}", ["a", "b", "c"]) == "a b"
    assert substitute_args("${@:0}", ["a"]) == "a"  # clamped to start 0
    assert substitute_args("no args", ["a"]) == "no args"
    assert substitute_args("$1", []) == ""


def test_format_prompt_template_invocation():
    template = PromptTemplate(name="t", content="Review $1 with $2")
    assert format_prompt_template_invocation(template, ["file.py", "care"]) == "Review file.py with care"
    assert format_prompt_template_invocation(template) == "Review  with "

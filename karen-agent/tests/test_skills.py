"""Tests for `karen_agent.skills` (pi's `harness/skills.ts`)."""

from pathlib import Path

from karen_agent import (
    SourcedPath,
    Skill,
    format_skill_invocation,
    load_skills,
    load_sourced_skills,
)
from karen_agent.skills import _dirname_env_path, _relative_env_path


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _skill_md(body: str = "Do the thing.", name: str = "alpha", description: str = "Alpha skill") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n{body}"


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------


def test_dirname_env_path():
    assert _dirname_env_path("/skills/a.md") == "/skills"
    assert _dirname_env_path("C:\\skills\\a.md") == "C:\\skills"
    assert _dirname_env_path("C:/skills/a.md") == "C:/skills"
    # A drive-rooted absolute path resolves to the drive root.
    assert _dirname_env_path("C:/skills") == "C:/"
    assert _dirname_env_path("C:/") == "/"
    assert _dirname_env_path("a.md") == "/"
    assert _dirname_env_path("/") == "/"


def test_relative_env_path():
    assert _relative_env_path("/root", "/root") == ""
    assert _relative_env_path("/root", "/root/a/b.md") == "a/b.md"
    assert _relative_env_path("C:\\root", "C:\\root\\a\\b.md") == "a/b.md"
    assert _relative_env_path("/root/", "/root/sub/") == "sub"
    assert _relative_env_path("/other", "/elsewhere/b.md") == "elsewhere/b.md"


# ---------------------------------------------------------------------------
# format_skill_invocation
# ---------------------------------------------------------------------------


def test_format_skill_invocation():
    skill = Skill(
        name="deploy",
        description="Deploy things",
        content="Run the deploy script.",
        file_path="/skills/deploy/SKILL.md",
        disable_model_invocation=False,
    )
    assert format_skill_invocation(skill) == (
        '<skill name="deploy" location="/skills/deploy/SKILL.md">\n'
        "References are relative to /skills/deploy.\n\n"
        "Run the deploy script.\n</skill>"
    )
    assert format_skill_invocation(skill, "Also notify the team.") == (
        '<skill name="deploy" location="/skills/deploy/SKILL.md">\n'
        "References are relative to /skills/deploy.\n\n"
        "Run the deploy script.\n</skill>\n\nAlso notify the team."
    )


def test_format_skill_invocation_windows_path():
    skill = Skill(name="a", description="d", content="c", file_path="C:\\skills\\a\\SKILL.md")
    assert "References are relative to C:\\skills\\a." in format_skill_invocation(skill)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_missing_directory_is_silent(tmp_path):
    result = load_skills(tmp_path / "nope")
    assert result.skills == []
    assert result.diagnostics == []


def test_file_input_is_skipped(tmp_path):
    path = _write(tmp_path / "file.md", "hello")
    result = load_skills(path)
    assert result.skills == [] and result.diagnostics == []


def test_loads_declared_skill_md(tmp_path):
    _write(tmp_path / "alpha" / "SKILL.md", _skill_md("Body text.\n"))
    result = load_skills(tmp_path)
    assert result.diagnostics == []
    assert len(result.skills) == 1
    skill = result.skills[0]
    assert skill.name == "alpha"
    assert skill.description == "Alpha skill"
    assert skill.content == "Body text."
    assert skill.file_path == str(tmp_path / "alpha" / "SKILL.md")
    assert skill.disable_model_invocation is False


def test_skill_md_short_circuits_sibling_files(tmp_path):
    _write(tmp_path / "alpha" / "SKILL.md", _skill_md())
    _write(tmp_path / "alpha" / "extra.md", "---\ndescription: Extra\n---\nExtra body")
    result = load_skills(tmp_path)
    assert [s.name for s in result.skills] == ["alpha"]


def test_nested_skills_are_recursive(tmp_path):
    _write(tmp_path / "a" / "SKILL.md", _skill_md(name="a", description="A"))
    _write(tmp_path / "b" / "c" / "SKILL.md", _skill_md(name="c", description="C"))
    _write(tmp_path / "b" / "c" / "root.md", "---\ndescription: Nested root md\n---\nbody")
    result = load_skills(tmp_path)
    assert sorted(s.name for s in result.skills) == ["a", "c"]


def test_root_md_files_with_frontmatter_are_loaded(tmp_path):
    root = tmp_path / "skills"
    _write(root / "notes.md", "---\ndescription: Root notes\n---\nNotes body")
    result = load_skills(root)
    assert result.diagnostics == []
    assert [s.name for s in result.skills] == ["skills"]  # no frontmatter name -> parent directory name
    assert result.skills[0].description == "Root notes"
    assert result.skills[0].content == "Notes body"


def test_root_md_without_description_is_dropped_silently(tmp_path):
    _write(tmp_path / "notes.md", "---\ntitle: x\n---\nbody")
    result = load_skills(tmp_path)
    assert result.skills == []
    assert result.diagnostics == []


def test_declared_skill_requires_description_and_reports(tmp_path):
    _write(tmp_path / "alpha" / "SKILL.md", "---\nname: alpha\n---\nBody")
    result = load_skills(tmp_path)
    assert result.skills == []
    assert [(d.code, d.message) for d in result.diagnostics] == [("invalid_metadata", "description is required")]


def test_name_validation_diagnostics(tmp_path):
    _write(tmp_path / "myskill" / "SKILL.md", _skill_md(name="Bad_Name"))
    result = load_skills(tmp_path)
    assert len(result.skills) == 1  # invalid metadata warns but still loads
    messages = [d.message for d in result.diagnostics]
    assert all(d.code == "invalid_metadata" for d in result.diagnostics)
    assert 'name "Bad_Name" does not match parent directory "myskill"' in messages
    assert "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)" in messages


def test_name_rules_for_length_hyphens(tmp_path):
    long_root = tmp_path / "long"
    long_root.mkdir()
    long_name = "a" * 65
    _write(long_root / long_name / "SKILL.md", _skill_md(name=long_name))
    result = load_skills(long_root)
    assert "name exceeds 64 characters (65)" in [d.message for d in result.diagnostics]

    hyphen_root = tmp_path / "hyphen"
    hyphen_root.mkdir()
    _write(hyphen_root / "x-y" / "SKILL.md", _skill_md(name="x-y"))
    assert load_skills(hyphen_root).diagnostics == []


def test_description_length_diagnostic(tmp_path):
    _write(tmp_path / "alpha" / "SKILL.md", _skill_md(description="d" * 1025))
    result = load_skills(tmp_path)
    assert [d.message for d in result.diagnostics] == [
        "description exceeds 1024 characters (1025)"
    ]


def test_invalid_yaml_in_declared_skill_is_parse_failed(tmp_path):
    _write(tmp_path / "alpha" / "SKILL.md", "---\nkey: [unclosed\n---\nBody")
    result = load_skills(tmp_path)
    assert result.skills == []
    assert [d.code for d in result.diagnostics] == ["parse_failed"]


def test_invalid_yaml_in_root_md_is_silent(tmp_path):
    _write(tmp_path / "notes.md", "---\nkey: [unclosed\n---\nBody")
    result = load_skills(tmp_path)
    assert result.skills == [] and result.diagnostics == []


def test_disable_model_invocation(tmp_path):
    _write(
        tmp_path / "alpha" / "SKILL.md",
        "---\nname: alpha\ndescription: A\ndisable-model-invocation: true\n---\nBody",
    )
    result = load_skills(tmp_path)
    assert result.skills[0].disable_model_invocation is True


def test_crlf_frontmatter(tmp_path):
    path = tmp_path / "alpha" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"---\r\nname: alpha\r\ndescription: CRLF skill\r\n---\r\nBody line\r\n")
    result = load_skills(tmp_path)
    assert result.skills[0].description == "CRLF skill"
    assert result.skills[0].content == "Body line"


def test_node_modules_and_dot_directories_skipped(tmp_path):
    _write(tmp_path / "node_modules" / "pkg" / "SKILL.md", _skill_md(name="pkg"))
    _write(tmp_path / ".hidden" / "SKILL.md", _skill_md(name="hidden"))
    _write(tmp_path / "visible" / "SKILL.md", _skill_md(name="visible"))
    result = load_skills(tmp_path)
    assert [s.name for s in result.skills] == ["visible"]


# ---------------------------------------------------------------------------
# ignore files
# ---------------------------------------------------------------------------


def test_gitignore_excludes_skills_and_negation_restores(tmp_path):
    _write(tmp_path / ".gitignore", "ignored/\n*.tmp.md\n!keep.md\n# a comment\n")
    _write(tmp_path / "ignored" / "SKILL.md", _skill_md(name="ignored"))
    _write(tmp_path / "notes.tmp.md", "---\ndescription: Temp\n---\nbody")
    _write(tmp_path / ".hidden.md", "---\ndescription: Dot\n---\nbody")
    _write(tmp_path / "keep.md", "---\nname: keep\ndescription: Keep\n---\nbody")
    result = load_skills(tmp_path)
    assert [s.name for s in result.skills] == ["keep"]


def test_nested_ignore_file_patterns_are_prefixed(tmp_path):
    _write(tmp_path / "sub" / ".ignore", "hidden/\n")
    _write(tmp_path / "sub" / "hidden" / "SKILL.md", _skill_md(name="hidden"))
    _write(tmp_path / "sub" / "visible" / "SKILL.md", _skill_md(name="visible"))
    # Same directory name outside `sub/` is unaffected by the nested ignore file.
    _write(tmp_path / "hidden" / "SKILL.md", _skill_md(name="hidden-root"))
    result = load_skills(tmp_path)
    assert sorted(s.name for s in result.skills) == ["hidden-root", "visible"]


def test_nested_loose_md_files_are_not_loaded(tmp_path):
    # Only root-level .md files carry frontmatter skills; nested dirs load SKILL.md only.
    sub = tmp_path / "sub"
    _write(sub / "notes.md", "---\nname: notes\ndescription: N\n---\nbody")
    result = load_skills(tmp_path)
    assert result.skills == [] and result.diagnostics == []


def test_fdignore_is_honored(tmp_path):
    _write(tmp_path / ".fdignore", "secret/\n")
    _write(tmp_path / "secret" / "SKILL.md", _skill_md(name="secret"))
    _write(tmp_path / "ok" / "SKILL.md", _skill_md(name="ok"))
    result = load_skills(tmp_path)
    assert [s.name for s in result.skills] == ["ok"]


# ---------------------------------------------------------------------------
# sourced loading
# ---------------------------------------------------------------------------


def test_load_sourced_skills_tags_results(tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    _write(first / "alpha" / "SKILL.md", _skill_md(name="alpha", description="A"))
    _write(second / "beta" / "SKILL.md", "---\nname: beta\n---\nBody")  # missing description -> diagnostic

    result = load_sourced_skills(
        [SourcedPath(path=str(first), source="project"), SourcedPath(path=str(second), source="user")]
    )
    assert [(s.skill.name, s.source) for s in result.skills] == [("alpha", "project")]
    assert [(d.code, d.source) for d in result.diagnostics] == [("invalid_metadata", "user")]


def test_load_sourced_skills_applies_mapping(tmp_path):
    _write(tmp_path / "alpha" / "SKILL.md", _skill_md(name="alpha", description="A"))
    result = load_sourced_skills(
        [SourcedPath(path=str(tmp_path), source={"origin": "app"})],
        map_skill=lambda skill, source: {"name": skill.name, "origin": source["origin"]},
    )
    assert result.skills[0].skill == {"name": "alpha", "origin": "app"}

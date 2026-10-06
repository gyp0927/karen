"""Tests for system-prompt assembly (`prompt.py`) and resource discovery
(`resources.py`) — pi's `core/system-prompt.ts` + the context-file half of
`core/resource-loader.ts`, driven entirely on tmp trees."""

import os
from pathlib import Path

import pytest

from karen_agent import Skill

from karen_coding_agent.prompt import (
    TOOL_GUIDELINES,
    build_system_prompt,
    build_system_prompt_sections,
    render_system_prompt_sections,
)
from karen_coding_agent.resources import (
    CONTEXT_FILE_CANDIDATES,
    ContextFile,
    default_skill_dirs,
    discover_append_system_prompt_file,
    discover_system_prompt_file,
    load_project_context_files,
    load_project_skills,
)


def _skill(name="greet", description="Greet people"):
    return Skill(
        name=name,
        description=description,
        content=f"# {name}\nDo the thing.",
        file_path=f"/skills/{name}/SKILL.md",
    )


def _own(files, root: Path):
    """Entries under `root` — the ancestor walk may add real files above tmp."""
    return [file for file in files if file.path.startswith(str(root))]


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def test_sections_shape_and_order():
    sections = build_system_prompt_sections(cwd="/work/proj", selected_tools=["read", "bash"])
    assert list(sections) == ["preamble", "tools", "rules", "cwd"]
    # the preamble is untagged; every other section is wrapped in its own tag
    assert not sections["preamble"].startswith("<")
    assert sections["tools"].startswith("<tools>\n") and sections["tools"].endswith("\n</tools>")
    assert sections["cwd"] == "<cwd>\n/work/proj\n</cwd>"


def test_tools_section_lists_selected_tools_in_order():
    sections = build_system_prompt_sections(cwd="/w", selected_tools=["write", "read", "mystery"])
    tools = sections["tools"]
    assert "- write: Create or overwrite files" in tools
    assert "- read: Read file contents" in tools
    assert "mystery" not in tools  # no snippet -> not advertised
    assert tools.index("- write") < tools.index("- read")
    assert "you may have access to other custom tools" in tools


def test_tools_section_is_none_when_nothing_has_a_snippet():
    sections = build_system_prompt_sections(cwd="/w", selected_tools=["mystery"], tool_snippets={})
    assert "(none)" in sections["tools"]


def test_default_selected_tools_are_pis_subset():
    sections = build_system_prompt_sections(cwd="/w")
    assert "- read:" in sections["tools"] and "- edit:" in sections["tools"]


def test_rules_fallback_only_without_search_tools():
    shell_only = build_system_prompt_sections(cwd="/w", selected_tools=["bash"])
    assert "- Use bash for file operations like ls, rg, find" in shell_only["rules"]

    powershell_only = build_system_prompt_sections(cwd="/w", selected_tools=["powershell"])
    assert "- Use PowerShell for file operations like listing, searching, and finding files" in powershell_only["rules"]

    both = build_system_prompt_sections(cwd="/w", selected_tools=["bash", "powershell"])
    assert "- Use bash or PowerShell for file operations like listing, searching, and finding files" in both["rules"]

    with_search = build_system_prompt_sections(cwd="/w", selected_tools=["bash", "grep"])
    assert "Use bash for file operations" not in with_search["rules"]


def test_rules_include_tool_guidelines_and_closing_rules():
    sections = build_system_prompt_sections(cwd="/w", selected_tools=["read", "write"])
    rules = sections["rules"]
    for rule in TOOL_GUIDELINES["read"] + TOOL_GUIDELINES["write"]:
        assert f"- {rule}" in rules
    assert rules.endswith("- Be concise in your responses\n- Show file paths clearly when working with files\n</rules>")


def test_rules_are_deduplicated():
    sections = build_system_prompt_sections(
        cwd="/w",
        selected_tools=["read"],
        prompt_guidelines=["Be concise in your responses", TOOL_GUIDELINES["read"][0], "  ", "Custom rule"],
    )
    rules = sections["rules"]
    assert rules.count("Be concise in your responses") == 1
    assert rules.count(TOOL_GUIDELINES["read"][0]) == 1
    assert "- Custom rule" in rules


def test_custom_prompt_replaces_preamble_and_drops_tools_and_rules():
    sections = build_system_prompt_sections(
        cwd="/w", selected_tools=["read"], custom_prompt="You are a pirate."
    )
    assert sections["preamble"] == "You are a pirate."
    assert "tools" not in sections and "rules" not in sections
    assert sections["cwd"] == "<cwd>\n/w\n</cwd>"


def test_append_system_prompt_becomes_addendum_before_context():
    sections = build_system_prompt_sections(
        cwd="/w",
        selected_tools=["read"],
        append_system_prompt="Always write tests.",
        context_files=[ContextFile(path="/w/AGENTS.md", content="Be nice")],
        skills=[_skill()],
    )
    assert list(sections) == ["preamble", "tools", "rules", "addendum", "project_context", "skills", "cwd"]
    assert sections["addendum"] == "<addendum>\nAlways write tests.\n</addendum>"


def test_project_context_matches_pi_shape():
    sections = build_system_prompt_sections(
        cwd="/w",
        selected_tools=["read"],
        context_files=[
            ContextFile(path="/AGENTS.md", content="global rules"),
            ContextFile(path="/w/AGENTS.md", content="project rules"),
        ],
    )
    assert sections["project_context"] == (
        "<project_context>\n"
        "Project-specific instructions and guidelines:\n\n"
        '<project_instructions path="/AGENTS.md">\nglobal rules\n</project_instructions>\n\n'
        '<project_instructions path="/w/AGENTS.md">\nproject rules\n</project_instructions>\n'
        "</project_context>"
    )


def test_skills_section_needs_a_file_reading_tool():
    with_skills = build_system_prompt_sections(cwd="/w", selected_tools=["read"], skills=[_skill()])
    assert "<available_skills>" in with_skills["skills"]
    assert "<name>greet</name>" in with_skills["skills"]

    without_read = build_system_prompt_sections(cwd="/w", selected_tools=["edit"], skills=[_skill()])
    assert "skills" not in without_read

    no_skills = build_system_prompt_sections(cwd="/w", selected_tools=["read"], skills=[])
    assert "skills" not in no_skills


def test_skills_with_disable_model_invocation_are_hidden():
    hidden = _skill(name="secret")
    hidden.disable_model_invocation = True
    sections = build_system_prompt_sections(cwd="/w", selected_tools=["read"], skills=[hidden])
    assert "skills" not in sections


def test_cwd_uses_forward_slashes():
    sections = build_system_prompt_sections(cwd=r"C:\work\proj", selected_tools=["read"])
    assert sections["cwd"] == "<cwd>\nC:/work/proj\n</cwd>"


def test_custom_sections_are_appended_and_validated():
    sections = build_system_prompt_sections(
        cwd="/w", selected_tools=["read"], sections={"extra": "<extra>\nhi\n</extra>"}
    )
    assert sections["extra"] == "<extra>\n<extra>\nhi\n</extra>\n</extra>"

    with pytest.raises(ValueError):
        build_system_prompt_sections(cwd="/w", sections={"preamble": "nope"})
    with pytest.raises(ValueError):
        build_system_prompt_sections(cwd="/w", sections={"Bad Name": "nope"})


def test_render_matches_build_system_prompt():
    options = dict(
        cwd="/w",
        selected_tools=["read", "bash"],
        append_system_prompt="extra",
        context_files=[ContextFile(path="/w/AGENTS.md", content="rules")],
        skills=[_skill()],
    )
    rendered = render_system_prompt_sections(build_system_prompt_sections(**options))
    assert build_system_prompt(**options) == rendered
    # preamble first, then one blank line, then the tagged sections
    lines = rendered.splitlines()
    assert lines[0].startswith("You are an expert coding assistant")
    assert lines[1] == ""
    assert lines[2] == "<tools>"
    assert rendered.endswith("</cwd>")


# ---------------------------------------------------------------------------
# context files
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_context_file_candidate_order(tmp_path):
    project = tmp_path / "proj"
    _write(project / "CLAUDE.md", "claude")
    _write(project / "AGENTS.md", "agents")
    _write(project / "AGENTS.override.md", "override")

    files = _own(load_project_context_files(project, tmp_path / "agent"), project)
    assert [Path(f.path).name for f in files] == ["AGENTS.override.md"]
    assert files[0].content == "override"

    os.remove(project / "AGENTS.override.md")
    files = _own(load_project_context_files(project, tmp_path / "agent"), project)
    assert [Path(f.path).name for f in files] == ["AGENTS.md"]

    os.remove(project / "AGENTS.md")
    files = _own(load_project_context_files(project, tmp_path / "agent"), project)
    assert [Path(f.path).name for f in files] == ["CLAUDE.md"]

    assert CONTEXT_FILE_CANDIDATES[0] == "AGENTS.override.md" and "CLAUDE.MD" in CONTEXT_FILE_CANDIDATES


def test_context_files_global_first_then_ancestors_outermost_first(tmp_path):
    agent = tmp_path / "agent"
    root = tmp_path / "root"
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    _write(agent / "AGENTS.md", "global")
    _write(root / "AGENTS.md", "root")
    _write(nested / "AGENTS.md", "nested")

    files = _own(load_project_context_files(nested, agent), tmp_path)
    assert [(Path(f.path).name, f.content) for f in files] == [
        ("AGENTS.md", "global"),
        ("AGENTS.md", "root"),
        ("AGENTS.md", "nested"),
    ]
    assert Path(files[0].path).parent == agent  # agent dir first
    assert Path(files[-1].path).parent == nested  # cwd last


def test_context_files_strip_bom_and_are_deduped(tmp_path):
    agent = tmp_path / "agent"
    work = tmp_path / "work"
    _write(work / "AGENTS.md", "\ufeffwith bom")
    files = _own(load_project_context_files(work, agent), work)
    assert files[-1].content == "with bom"

    # the agent dir is also the cwd: the same file must not load twice
    agent.mkdir(parents=True, exist_ok=True)
    _write(agent / "AGENTS.md", "shared")
    files = _own(load_project_context_files(agent, agent), agent)
    assert [f.content for f in files] == ["shared"]


def test_context_file_skips_directories(tmp_path):
    work = tmp_path / "work"
    (work / "AGENTS.md").mkdir(parents=True)  # a directory named like a context file
    _write(work / "CLAUDE.md", "fallback wins")
    files = _own(load_project_context_files(work, tmp_path / "agent"), work)
    assert [Path(f.path).name for f in files] == ["CLAUDE.md"]


# ---------------------------------------------------------------------------
# SYSTEM.md / APPEND_SYSTEM.md / skills
# ---------------------------------------------------------------------------


def test_system_prompt_file_project_wins_then_global(tmp_path):
    cwd = tmp_path / "proj"
    agent = tmp_path / "agent"
    _write(agent / "SYSTEM.md", "global prompt")
    assert discover_system_prompt_file(cwd, agent) == str(agent / "SYSTEM.md")

    _write(cwd / ".karen" / "SYSTEM.md", "project prompt")
    assert discover_system_prompt_file(cwd, agent) == str(cwd / ".karen" / "SYSTEM.md")

    assert discover_system_prompt_file(tmp_path / "empty", tmp_path / "nowhere") is None


def test_append_system_prompt_file_project_wins_then_global(tmp_path):
    cwd = tmp_path / "proj"
    agent = tmp_path / "agent"
    _write(agent / "APPEND_SYSTEM.md", "global append")
    assert discover_append_system_prompt_file(cwd, agent) == str(agent / "APPEND_SYSTEM.md")

    _write(cwd / ".karen" / "APPEND_SYSTEM.md", "project append")
    assert discover_append_system_prompt_file(cwd, agent) == str(cwd / ".karen" / "APPEND_SYSTEM.md")

    assert discover_append_system_prompt_file(tmp_path / "empty", tmp_path / "nowhere") is None


def test_skill_dirs_and_loading(tmp_path):
    cwd = tmp_path / "proj"
    agent = tmp_path / "agent"
    assert default_skill_dirs(cwd, agent) == [
        str(cwd / ".karen" / "skills"),
        str(agent / "skills"),
    ]

    _write(cwd / ".karen" / "skills" / "greet" / "SKILL.md", "---\nname: greet\ndescription: Say hi\n---\n\nHi!")
    _write(agent / "skills" / "notes" / "SKILL.md", "---\nname: notes\ndescription: Take notes\n---\n\nNote it.")

    result = load_project_skills(cwd, agent)
    assert sorted(skill.name for skill in result.skills) == ["greet", "notes"]
    assert result.diagnostics == []


def test_missing_skill_dirs_are_skipped(tmp_path):
    result = load_project_skills(tmp_path / "proj", tmp_path / "agent")
    assert result.skills == [] and result.diagnostics == []

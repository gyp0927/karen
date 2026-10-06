"""System-prompt assembly (pi coding-agent's `core/system-prompt.ts`).

The prompt is an ordered set of named sections: `.preamble` is untagged text,
every other section is wrapped in an XML tag of the same name so the model can
relate later updates to it. `AgentSession` puts these sections on the session's
system message (`SystemMessage.sections`) and karen-ai renders them at request
time; `build_system_prompt` renders the same text for callers that want a
plain string.

Deviations from pi: the `docs` section (pi's own documentation paths) has no
karen equivalent and is omitted; `forceSystemPrompt` (set by a pi extension
hook) is not ported; and karen's shell tools expose no session environment
variables, so the `PI_*` guideline those tools contribute is dropped.
"""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Sequence

from karen_ai import SystemMessage
from karen_ai.utils.text import get_system_message_text
from karen_agent import Skill, format_skills_for_system_prompt

from .resources import ContextFile

__all__ = [
    "PREAMBLE_TEMPLATE",
    "SECTION_NAME_PATTERN",
    "TOOL_SNIPPETS",
    "TOOL_GUIDELINES",
    "build_system_prompt_sections",
    "build_system_prompt",
    "render_system_prompt_sections",
]

#: Section names must be tag-safe; `preamble` is reserved for the untagged text.
SECTION_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")

PREAMBLE_TEMPLATE = (
    "You are an expert coding assistant operating inside karen, a coding agent harness. "
    "You help users by reading files, executing commands, editing code, and writing new files."
)

#: One-line tool descriptions (pi's `promptSnippet`s) for the `tools` section.
TOOL_SNIPPETS: Dict[str, str] = {
    "read": "Read file contents",
    "bash": "Execute bash commands (ls, grep, find, etc.)",
    "powershell": "Execute PowerShell commands",
    "edit": "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
    "write": "Create or overwrite files",
    "grep": "Search file contents for patterns (respects .gitignore)",
    "find": "Find files by glob pattern (respects .gitignore)",
    "ls": "List directory contents",
}

#: Extra rules contributed by individual tools (pi's `promptGuidelines`s).
TOOL_GUIDELINES: Dict[str, List[str]] = {
    "read": ["Use read to examine files instead of cat or sed."],
    "edit": [
        "Use edit for precise changes (edits[].oldText must match exactly)",
        "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
        "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
        "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
    ],
    "write": ["Use write only for new files or complete rewrites."],
}

#: Tools in pi's default prompt selection when the caller names none.
DEFAULT_SELECTED_TOOLS = ("read", "bash", "edit", "write")


def _build_rules(
    selected_tools: Sequence[str],
    tool_guidelines: Mapping[str, Sequence[str]],
    prompt_guidelines: Sequence[str],
) -> str:
    """Assemble the `rules` bullets: fallback file-op rule, tool rules, extra
    rules, then the always-present closing rules — deduplicated, order kept."""
    rules: List[str] = []
    seen = set()

    def add(rule: str) -> None:
        normalized = rule.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        rules.append(normalized)

    has_bash = "bash" in selected_tools
    has_powershell = "powershell" in selected_tools
    has_search_tools = any(name in selected_tools for name in ("grep", "find", "ls"))

    if (has_bash or has_powershell) and not has_search_tools:
        if has_bash and has_powershell:
            add("Use bash or PowerShell for file operations like listing, searching, and finding files")
        elif has_powershell:
            add("Use PowerShell for file operations like listing, searching, and finding files")
        else:
            add("Use bash for file operations like ls, rg, find")

    for name in selected_tools:
        for rule in tool_guidelines.get(name, ()):
            add(rule)
    for rule in prompt_guidelines:
        add(rule)
    add("Be concise in your responses")
    add("Show file paths clearly when working with files")
    return "\n".join(f"- {rule}" for rule in rules)


def _render_project_context(context_files: Sequence[ContextFile]) -> str:
    parts = ["Project-specific instructions and guidelines:"]
    for file in context_files:
        parts.append(f'<project_instructions path="{file.path}">\n{file.content}\n</project_instructions>')
    return "\n\n".join(parts)


def build_system_prompt_sections(
    *,
    cwd: str,
    selected_tools: Optional[Sequence[str]] = None,
    custom_prompt: Optional[str] = None,
    append_system_prompt: str = "",
    sections: Optional[Mapping[str, str]] = None,
    context_files: Sequence[ContextFile] = (),
    skills: Sequence[Skill] = (),
    tool_snippets: Optional[Mapping[str, str]] = None,
    tool_guidelines: Optional[Mapping[str, Sequence[str]]] = None,
    prompt_guidelines: Sequence[str] = (),
) -> Dict[str, str]:
    """Build the ordered sections of the structured system prompt.

    `custom_prompt` (karen's `SYSTEM.md`) replaces the default preamble;
    `append_system_prompt` (`APPEND_SYSTEM.md`) becomes the `addendum` section,
    which sits after project context and before skills and cwd, like pi.
    """
    tool_names = list(selected_tools) if selected_tools is not None else list(DEFAULT_SELECTED_TOOLS)
    snippets = dict(TOOL_SNIPPETS if tool_snippets is None else tool_snippets)
    guidelines = dict(TOOL_GUIDELINES if tool_guidelines is None else tool_guidelines)
    custom_sections = dict(sections or {})

    for name in custom_sections:
        if name == "preamble" or not SECTION_NAME_PATTERN.match(name):
            raise ValueError(f"Invalid system prompt section name: {name}")

    if custom_prompt:
        preamble = custom_prompt
    else:
        visible_tools = [name for name in tool_names if snippets.get(name)]
        tools = (
            "\n".join(f"- {name}: {snippets[name]}" for name in visible_tools)
            if visible_tools
            else "(none)"
        )
        preamble = PREAMBLE_TEMPLATE

    prompt_sections: Dict[str, str] = {"preamble": preamble}
    if not custom_prompt:
        prompt_sections["tools"] = (
            f"{tools}\n\nIn addition to the tools above, you may have access to "
            "other custom tools depending on the project."
        )
        prompt_sections["rules"] = _build_rules(tool_names, guidelines, prompt_guidelines)

    if append_system_prompt:
        prompt_sections["addendum"] = append_system_prompt
    if context_files:
        prompt_sections["project_context"] = _render_project_context(context_files)
    # pi requires a file-reading tool before advertising skills; karen's read
    # tool is what loads them.
    if "read" in tool_names or "bash" in tool_names:
        skills_prompt = format_skills_for_system_prompt(list(skills)).strip()
        if skills_prompt:
            prompt_sections["skills"] = skills_prompt
    prompt_sections["cwd"] = cwd.replace("\\", "/")
    for name, content in custom_sections.items():
        if content:
            prompt_sections[name] = content

    rendered: Dict[str, str] = {"preamble": prompt_sections["preamble"]}
    for name, content in prompt_sections.items():
        if name != "preamble":
            rendered[name] = f"<{name}>\n{content}\n</{name}>"
    return rendered


def render_system_prompt_sections(sections: Mapping[str, str]) -> str:
    """Render sections exactly as the transcript's system message replays them."""
    return get_system_message_text(
        SystemMessage(content="", sections=dict(sections), timestamp=0)
    )


def build_system_prompt(**options) -> str:
    """Build the system prompt text (see `build_system_prompt_sections`)."""
    return render_system_prompt_sections(build_system_prompt_sections(**options))

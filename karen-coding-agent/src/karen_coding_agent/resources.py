"""Project resources: context files, custom system prompts, and skills — the
subset of pi coding-agent's `core/resource-loader.ts` that karen consumes.

On-disk layout (pi uses `.pi` and `~/.pi/agent`; karen uses `.karen` and
`~/.karen`):

    ~/.karen/AGENTS.md           global project context (always first)
    <cwd>/.karen/AGENTS.md       project context; also loaded from every
                                 ancestor directory, outermost first
    ~/.karen/SYSTEM.md           custom system prompt (replaces the preamble)
    ~/.karen/APPEND_SYSTEM.md    text appended before context, skills and cwd
    ~/.karen/skills/             global skills (SKILL.md, like karen-agent)
    <cwd>/.karen/skills/         project skills

Context-file candidates per directory, in pi's order: `AGENTS.override.md`,
`AGENTS.md`, `AGENTS.MD`, `CLAUDE.md`, `CLAUDE.MD` — the first one that exists
wins, files are read as UTF-8 with a leading BOM stripped.

Deviations from pi: project `SYSTEM.md`/`APPEND_SYSTEM.md` are gated behind
pi's project-trust prompt and the git-worktree "shadowed context file" rule
skips a duplicated worktree copy — karen has neither, so a project file always
applies and both copies would load. Instruction strings are not supported for
`SYSTEM.md` (pi's `resolvePromptInput` reads a file *or* uses the literal text).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

from karen_agent import LoadSkillsResult, load_skills

__all__ = [
    "DEFAULT_AGENT_DIR",
    "CONTEXT_FILE_CANDIDATES",
    "ContextFile",
    "load_project_context_files",
    "discover_system_prompt_file",
    "discover_append_system_prompt_file",
    "default_skill_dirs",
    "load_project_skills",
    "read_text_file",
]

#: karen's config/home directory — pi's `getAgentDir()` (`~/.pi/agent`).
DEFAULT_AGENT_DIR = Path.home() / ".karen"

#: First match per directory wins, in this order (pi's `loadContextFileFromDir`).
CONTEXT_FILE_CANDIDATES = ("AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD")

SYSTEM_PROMPT_FILE = "SYSTEM.md"
APPEND_SYSTEM_PROMPT_FILE = "APPEND_SYSTEM.md"
SKILLS_DIR_NAME = "skills"


@dataclass(frozen=True)
class ContextFile:
    """A loaded project-instructions file (pi's `{ path, content }`)."""

    path: str
    content: str


def read_text_file(path: Union[str, Path]) -> str:
    """Read a UTF-8 text file, stripping a leading BOM (pi's `stripBom`)."""
    with open(path, "r", encoding="utf-8-sig") as handle:
        return handle.read()


def _load_context_file_from_dir(directory: str) -> Optional[ContextFile]:
    for filename in CONTEXT_FILE_CANDIDATES:
        candidate = os.path.join(directory, filename)
        if not os.path.isfile(candidate):
            continue
        try:
            return ContextFile(path=candidate, content=read_text_file(candidate))
        except OSError:
            continue
    return None


def load_project_context_files(
    cwd: Union[str, Path],
    agent_dir: Union[str, Path] = DEFAULT_AGENT_DIR,
) -> List[ContextFile]:
    """Load context files: the agent dir's file first, then every ancestor of
    `cwd` outermost-first, ending with `cwd` itself.

    Paths are deduplicated by real path, so a symlinked or reused file is
    never applied twice.
    """
    resolved_cwd = os.path.abspath(str(cwd))
    resolved_agent_dir = os.path.abspath(str(agent_dir))

    context_files: List[ContextFile] = []
    seen: set = set()

    global_context = _load_context_file_from_dir(resolved_agent_dir)
    if global_context is not None:
        context_files.append(global_context)
        seen.add(os.path.realpath(global_context.path))

    ancestors: List[ContextFile] = []
    current_dir = resolved_cwd
    while True:
        context_file = _load_context_file_from_dir(current_dir)
        if context_file is not None:
            real = os.path.realpath(context_file.path)
            if real not in seen:
                ancestors.insert(0, context_file)
                seen.add(real)
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    context_files.extend(ancestors)
    return context_files


def _discover_prompt_file(cwd: Union[str, Path], agent_dir: Union[str, Path], filename: str) -> Optional[str]:
    project_path = os.path.join(os.path.abspath(str(cwd)), ".karen", filename)
    if os.path.isfile(project_path):
        return project_path
    global_path = os.path.join(os.path.abspath(str(agent_dir)), filename)
    if os.path.isfile(global_path):
        return global_path
    return None


def discover_system_prompt_file(
    cwd: Union[str, Path], agent_dir: Union[str, Path] = DEFAULT_AGENT_DIR
) -> Optional[str]:
    """The `SYSTEM.md` that replaces the default preamble, if any (project wins)."""
    return _discover_prompt_file(cwd, agent_dir, SYSTEM_PROMPT_FILE)


def discover_append_system_prompt_file(
    cwd: Union[str, Path], agent_dir: Union[str, Path] = DEFAULT_AGENT_DIR
) -> Optional[str]:
    """The `APPEND_SYSTEM.md` appended to the prompt, if any (project wins)."""
    return _discover_prompt_file(cwd, agent_dir, APPEND_SYSTEM_PROMPT_FILE)


def default_skill_dirs(
    cwd: Union[str, Path], agent_dir: Union[str, Path] = DEFAULT_AGENT_DIR
) -> List[str]:
    """Skill directories, project first (pi's `.pi/skills` + `~/.pi/agent/skills`)."""
    return [
        os.path.join(os.path.abspath(str(cwd)), ".karen", SKILLS_DIR_NAME),
        os.path.join(os.path.abspath(str(agent_dir)), SKILLS_DIR_NAME),
    ]


def load_project_skills(
    cwd: Union[str, Path],
    agent_dir: Union[str, Path] = DEFAULT_AGENT_DIR,
    extra_dirs: Sequence[str] = (),
) -> LoadSkillsResult:
    """Load project + global skills (plus `extra_dirs`), with diagnostics."""
    dirs = [*default_skill_dirs(cwd, agent_dir), *extra_dirs]
    return load_skills(dirs)

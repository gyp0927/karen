"""Skill loading (pi's `harness/skills.ts`).

Loads `SKILL.md` files recursively plus direct root `.md` files with skill
frontmatter, honoring `.gitignore` / `.ignore` / `.fdignore` files. Missing
input directories are skipped silently; every other problem is a diagnostic.

karen deviations from pi: synchronous `pathlib` I/O instead of `ExecutionEnv`,
PyYAML instead of the `yaml` npm package (diagnostic messages come from the
parser and differ in wording), pathspec's `GitIgnoreSpec` instead of the
`ignore` npm package (same gitignore semantics), no chord `Context` parameter, and
`os.path.join` is infallible so pi's join-failure diagnostic branch does not
exist.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import yaml
from karen_ai.types import KarenBase
from pathspec import GitIgnoreSpec
from pydantic import ConfigDict

from .prompt_templates import SourcedPath
from .resources import Skill

__all__ = [
    "MAX_NAME_LENGTH",
    "MAX_DESCRIPTION_LENGTH",
    "IGNORE_FILE_NAMES",
    "SkillDiagnosticCode",
    "SkillDiagnostic",
    "LoadSkillsResult",
    "SourcedSkill",
    "SourcedSkillDiagnostic",
    "LoadSourcedSkillsResult",
    "format_skill_invocation",
    "load_skills",
    "load_sourced_skills",
]

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
IGNORE_FILE_NAMES = [".gitignore", ".ignore", ".fdignore"]

SkillDiagnosticCode = Literal["file_info_failed", "list_failed", "read_failed", "parse_failed", "invalid_metadata"]


class SkillDiagnostic(KarenBase):
    """Warning produced while loading skills."""

    #: Diagnostic severity. Currently only warnings are emitted.
    type: Literal["warning"] = "warning"
    #: Stable diagnostic code.
    code: SkillDiagnosticCode
    #: Human-readable diagnostic message.
    message: str
    #: Path associated with the diagnostic.
    path: str


class LoadSkillsResult(KarenBase):
    skills: List[Skill]
    diagnostics: List[SkillDiagnostic]


class SourcedSkill(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    skill: Any  # Skill or the mapped TSkill
    source: Any = None


class SourcedSkillDiagnostic(SkillDiagnostic):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Any = None


class LoadSourcedSkillsResult(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    skills: List[SourcedSkill]
    diagnostics: List[SourcedSkillDiagnostic]


def format_skill_invocation(skill: Skill, additional_instructions: Optional[str] = None) -> str:
    """Format a skill invocation prompt, optionally appending additional user instructions."""
    skill_block = (
        f'<skill name="{skill.name}" location="{skill.file_path}">\n'
        f"References are relative to {_dirname_env_path(skill.file_path)}.\n\n"
        f"{skill.content}\n</skill>"
    )
    return f"{skill_block}\n\n{additional_instructions}" if additional_instructions else skill_block


def load_skills(dirs: Union[str, Path, List[Union[str, Path]]]) -> LoadSkillsResult:
    """Load skills from one or more directories.

    Traverses directories recursively, loads `SKILL.md` files, loads direct root `.md` files with skill
    frontmatter, honors ignore files, and returns diagnostics for invalid declared skill files. Missing input
    directories are skipped.
    """
    skills: List[Skill] = []
    diagnostics: List[SkillDiagnostic] = []
    dir_list = [dirs] if isinstance(dirs, (str, Path)) else list(dirs)
    for dir_path in dir_list:
        dir_str = str(dir_path)
        try:
            os.lstat(dir_str)
        except FileNotFoundError:
            continue
        except OSError as error:
            diagnostics.append(SkillDiagnostic(code="file_info_failed", message=str(error), path=dir_str))
            continue
        if _resolve_kind(dir_str, diagnostics) != "directory":
            continue
        found, found_diagnostics = _load_skills_from_dir(dir_str, True, _IgnoreMatcher(), dir_str)
        skills.extend(found)
        diagnostics.extend(found_diagnostics)
    return LoadSkillsResult(skills=skills, diagnostics=diagnostics)


def load_sourced_skills(
    inputs: List[SourcedPath],
    map_skill: Optional[Callable[[Skill, Any], Any]] = None,
) -> LoadSourcedSkillsResult:
    """Load skills from source-tagged directories.

    Source values are preserved exactly and attached to every loaded skill and diagnostic. karen-agent does not
    interpret source values; applications define their own provenance shape.
    """
    skills: List[SourcedSkill] = []
    diagnostics: List[SourcedSkillDiagnostic] = []
    for source_input in inputs:
        result = load_skills(source_input.path)
        for skill in result.skills:
            skills.append(
                SourcedSkill(
                    skill=map_skill(skill, source_input.source) if map_skill is not None else skill,
                    source=source_input.source,
                )
            )
        for diagnostic in result.diagnostics:
            diagnostics.append(
                SourcedSkillDiagnostic(
                    code=diagnostic.code,
                    message=diagnostic.message,
                    path=diagnostic.path,
                    source=source_input.source,
                )
            )
    return LoadSourcedSkillsResult(skills=skills, diagnostics=diagnostics)


class _IgnoreMatcher:
    """Accumulates root-relative ignore patterns (pi uses npm `ignore`; karen uses pathspec's GitIgnoreSpec)."""

    def __init__(self) -> None:
        self._patterns: List[str] = []
        self._spec = GitIgnoreSpec.from_lines([])

    def add(self, patterns: List[str]) -> None:
        if not patterns:
            return
        self._patterns.extend(patterns)
        self._spec = GitIgnoreSpec.from_lines(self._patterns)

    def ignores(self, rel_path: str) -> bool:
        return self._spec.match_file(rel_path)


def _load_skills_from_dir(
    dir_path: str,
    include_root_files: bool,
    ignore_matcher: _IgnoreMatcher,
    root_dir: str,
) -> Tuple[List[Skill], List[SkillDiagnostic]]:
    skills: List[Skill] = []
    diagnostics: List[SkillDiagnostic] = []

    try:
        os.lstat(dir_path)
    except FileNotFoundError:
        return skills, diagnostics
    except OSError as error:
        diagnostics.append(SkillDiagnostic(code="file_info_failed", message=str(error), path=dir_path))
        return skills, diagnostics
    if _resolve_kind(dir_path, diagnostics) != "directory":
        return skills, diagnostics

    _add_ignore_rules(dir_path, ignore_matcher, root_dir, diagnostics)

    try:
        entries = list(os.scandir(dir_path))
    except OSError as error:
        diagnostics.append(SkillDiagnostic(code="list_failed", message=str(error), path=dir_path))
        return skills, diagnostics

    parent_dir_name = Path(dir_path).name

    for entry in entries:
        if entry.name != "SKILL.md":
            continue
        full_path = entry.path
        if _resolve_kind(full_path, diagnostics) != "file":
            continue
        rel_path = _relative_env_path(root_dir, full_path)
        if ignore_matcher.ignores(rel_path):
            continue
        skill, file_diagnostics = _load_skill_from_file(full_path, parent_dir_name)
        if skill is not None:
            skills.append(skill)
        diagnostics.extend(file_diagnostics)
        return skills, diagnostics

    for entry in sorted(entries, key=lambda e: e.name):
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        full_path = entry.path
        kind = _resolve_kind(full_path, diagnostics)
        if kind is None:
            continue

        rel_path = _relative_env_path(root_dir, full_path)
        ignore_path = f"{rel_path}/" if kind == "directory" else rel_path
        if ignore_matcher.ignores(ignore_path):
            continue

        if kind == "directory":
            found, found_diagnostics = _load_skills_from_dir(full_path, False, ignore_matcher, root_dir)
            skills.extend(found)
            diagnostics.extend(found_diagnostics)
            continue

        if not include_root_files or not entry.name.endswith(".md"):
            continue
        skill, file_diagnostics = _load_skill_from_file(full_path, parent_dir_name)
        if skill is not None:
            skills.append(skill)
        diagnostics.extend(file_diagnostics)

    return skills, diagnostics


def _add_ignore_rules(
    dir_path: str,
    matcher: _IgnoreMatcher,
    root_dir: str,
    diagnostics: List[SkillDiagnostic],
) -> None:
    relative_dir = _relative_env_path(root_dir, dir_path)
    prefix = f"{relative_dir}/" if relative_dir else ""

    for filename in IGNORE_FILE_NAMES:
        ignore_path = os.path.join(dir_path, filename)
        try:
            info = os.lstat(ignore_path)
        except FileNotFoundError:
            continue
        except OSError as error:
            diagnostics.append(SkillDiagnostic(code="file_info_failed", message=str(error), path=ignore_path))
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        try:
            content = Path(ignore_path).read_text(encoding="utf-8")
        except OSError as error:
            diagnostics.append(SkillDiagnostic(code="read_failed", message=str(error), path=ignore_path))
            continue
        patterns = [
            prefixed
            for line in re.split(r"\r?\n", content)
            if (prefixed := _prefix_ignore_pattern(line, prefix)) is not None
        ]
        matcher.add(patterns)


def _prefix_ignore_pattern(line: str, prefix: str) -> Optional[str]:
    trimmed = line.strip()
    if not trimmed:
        return None
    if trimmed.startswith("#") and not trimmed.startswith("\\#"):
        return None

    pattern = line
    negated = False
    if pattern.startswith("!"):
        negated = True
        pattern = pattern[1:]
    elif pattern.startswith("\\!"):
        pattern = pattern[1:]
    if pattern.startswith("/"):
        pattern = pattern[1:]
    prefixed = f"{prefix}{pattern}" if prefix else pattern
    return f"!{prefixed}" if negated else prefixed


def _load_skill_from_file(file_path: str, parent_dir_name: str) -> Tuple[Optional[Skill], List[SkillDiagnostic]]:
    diagnostics: List[SkillDiagnostic] = []
    is_declared_skill = re.split(r"[\\/]", re.sub(r"[\\/]+$", "", file_path))[-1] == "SKILL.md"
    try:
        raw_content = Path(file_path).read_text(encoding="utf-8")
    except OSError as error:
        diagnostics.append(SkillDiagnostic(code="read_failed", message=str(error), path=file_path))
        return None, diagnostics

    try:
        frontmatter, body = _parse_frontmatter(raw_content)
    except Exception as error:
        if is_declared_skill:
            diagnostics.append(SkillDiagnostic(code="parse_failed", message=str(error), path=file_path))
        return None, diagnostics

    description = frontmatter.get("description")
    description = description if isinstance(description, str) else None
    if not is_declared_skill and (not description or description.strip() == ""):
        return None, diagnostics

    for error in _validate_description(description):
        diagnostics.append(SkillDiagnostic(code="invalid_metadata", message=error, path=file_path))

    frontmatter_name = frontmatter.get("name")
    frontmatter_name = frontmatter_name if isinstance(frontmatter_name, str) else None
    name = frontmatter_name or parent_dir_name
    for error in _validate_name(name, parent_dir_name):
        diagnostics.append(SkillDiagnostic(code="invalid_metadata", message=error, path=file_path))

    if not description or description.strip() == "":
        return None, diagnostics

    return (
        Skill(
            name=name,
            description=description,
            content=body,
            file_path=file_path,
            disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
        ),
        diagnostics,
    )


def _validate_name(name: str, parent_dir_name: str) -> List[str]:
    errors: List[str] = []
    if name != parent_dir_name:
        errors.append(f'name "{name}" does not match parent directory "{parent_dir_name}"')
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: Optional[str]) -> List[str]:
    errors: List[str] = []
    if not description or description.strip() == "":
        errors.append("description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})")
    return errors


def _parse_frontmatter(content: str) -> Tuple[dict, str]:
    """Split YAML frontmatter from the body. Raises on YAML parse failure (pi returns Err)."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---"):
        return {}, normalized
    end_index = normalized.find("\n---", 3)
    if end_index == -1:
        return {}, normalized
    yaml_string = normalized[4:end_index]
    body = normalized[end_index + 4 :].strip()
    frontmatter = yaml.safe_load(yaml_string)
    if not isinstance(frontmatter, dict):
        # JS duck-typing: property access on a scalar/array yields undefined, so
        # a non-mapping document behaves like empty frontmatter.
        frontmatter = {}
    return frontmatter, body


def _resolve_kind(path: str, diagnostics: List[SkillDiagnostic]) -> Optional[Literal["file", "directory"]]:
    """lstat the path; symlinks resolve to their target's kind (pi's canonicalPath + fileInfo)."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as error:
        diagnostics.append(SkillDiagnostic(code="file_info_failed", message=str(error), path=path))
        return None
    if stat.S_ISREG(info.st_mode):
        return "file"
    if stat.S_ISDIR(info.st_mode):
        return "directory"
    try:
        target = os.stat(path)
    except FileNotFoundError:
        return None
    except OSError as error:
        diagnostics.append(SkillDiagnostic(code="file_info_failed", message=str(error), path=path))
        return None
    if stat.S_ISREG(target.st_mode):
        return "file"
    if stat.S_ISDIR(target.st_mode):
        return "directory"
    return None


def _dirname_env_path(path: str) -> str:
    normalized = re.sub(r"[\\/]+$", "", path)
    separator_index = max(normalized.rfind("/"), normalized.rfind("\\"))
    if separator_index == 2 and normalized[1] == ":":
        return normalized[:3]
    return "/" if separator_index <= 0 else normalized[:separator_index]


def _relative_env_path(root: str, path: str) -> str:
    normalized_root = re.sub(r"/+$", "", root.replace("\\", "/"))
    normalized_path = re.sub(r"/+$", "", path.replace("\\", "/"))
    if normalized_path == normalized_root:
        return ""
    if normalized_path.startswith(f"{normalized_root}/"):
        return normalized_path[len(normalized_root) + 1 :]
    return normalized_path.lstrip("/")

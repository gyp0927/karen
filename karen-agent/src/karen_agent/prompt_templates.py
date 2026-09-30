"""Prompt template loading and argument substitution (pi's `harness/prompt-templates.ts`).

Deviation from pi: pi loads through its ``ExecutionEnv`` capability layer;
karen uses direct synchronous ``pathlib`` I/O (consistent with M1/M2), so the
load functions are plain sync calls. YAML frontmatter is parsed with PyYAML
(pi uses the `yaml` npm package).
"""

from __future__ import annotations

import os
import re
import stat as stat_module
from pathlib import Path
from typing import Any, Callable, Generic, List, Literal, Optional, TypeVar, Union

import yaml
from karen_ai.types import KarenBase
from pydantic import ConfigDict

from .resources import PromptTemplate

__all__ = [
    "PromptTemplateDiagnosticCode",
    "PromptTemplateDiagnostic",
    "LoadPromptTemplatesResult",
    "SourcedPath",
    "SourcedPromptTemplate",
    "SourcedPromptTemplateDiagnostic",
    "LoadSourcedPromptTemplatesResult",
    "load_prompt_templates",
    "load_sourced_prompt_templates",
    "parse_command_args",
    "substitute_args",
    "format_prompt_template_invocation",
]

PromptTemplateDiagnosticCode = Literal["file_info_failed", "list_failed", "read_failed", "parse_failed"]


class PromptTemplateDiagnostic(KarenBase):
    """Warning produced while loading prompt templates."""

    #: Diagnostic severity. Currently only warnings are emitted.
    type: Literal["warning"] = "warning"
    #: Stable diagnostic code.
    code: PromptTemplateDiagnosticCode
    #: Human-readable diagnostic message.
    message: str
    #: Path associated with the diagnostic.
    path: str


class LoadPromptTemplatesResult(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_templates: List[PromptTemplate]
    diagnostics: List[PromptTemplateDiagnostic]


def load_prompt_templates(paths: Union[str, os.PathLike, List[Union[str, os.PathLike]]]) -> LoadPromptTemplatesResult:
    """Load prompt templates from one or more paths.

    Directory inputs load direct ``.md`` children non-recursively. File inputs
    load explicit ``.md`` files. Missing paths and non-markdown files are
    skipped. Read and parse failures are returned as diagnostics.
    """
    prompt_templates: List[PromptTemplate] = []
    diagnostics: List[PromptTemplateDiagnostic] = []
    path_list = paths if isinstance(paths, list) else [paths]
    for path in path_list:
        path_str = os.fspath(path)
        kind = _resolve_kind(path_str, diagnostics)
        if kind == "directory":
            loaded, diags = _load_templates_from_dir(path_str)
            prompt_templates.extend(loaded)
            diagnostics.extend(diags)
        elif kind == "file" and Path(path_str).name.endswith(".md"):
            template, diags = _load_template_from_file(path_str, Path(path_str).name)
            if template is not None:
                prompt_templates.append(template)
            diagnostics.extend(diags)
    return LoadPromptTemplatesResult(prompt_templates=prompt_templates, diagnostics=diagnostics)


TSource = TypeVar("TSource")
TPromptTemplate = TypeVar("TPromptTemplate", bound=PromptTemplate)


class SourcedPath(KarenBase):
    """One template path tagged with an application-defined provenance value."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str
    source: Any = None


class SourcedPromptTemplate(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_template: Any  # PromptTemplate or the mapped TPromptTemplate
    source: Any = None


class SourcedPromptTemplateDiagnostic(PromptTemplateDiagnostic):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Any = None


class LoadSourcedPromptTemplatesResult(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_templates: List[SourcedPromptTemplate]
    diagnostics: List[SourcedPromptTemplateDiagnostic]


def load_sourced_prompt_templates(
    inputs: List[SourcedPath],
    map_prompt_template: Optional[Callable[[PromptTemplate, Any], Any]] = None,
) -> LoadSourcedPromptTemplatesResult:
    """Load prompt templates from source-tagged paths.

    Source values are preserved exactly and attached to every loaded prompt
    template and diagnostic. karen-agent does not interpret source values;
    applications define their own provenance shape.
    """
    prompt_templates: List[SourcedPromptTemplate] = []
    diagnostics: List[SourcedPromptTemplateDiagnostic] = []
    for source_input in inputs:
        result = load_prompt_templates(source_input.path)
        for template in result.prompt_templates:
            prompt_templates.append(
                SourcedPromptTemplate(
                    prompt_template=(
                        map_prompt_template(template, source_input.source)
                        if map_prompt_template is not None
                        else template
                    ),
                    source=source_input.source,
                )
            )
        for diagnostic in result.diagnostics:
            diagnostics.append(
                SourcedPromptTemplateDiagnostic(
                    code=diagnostic.code,
                    message=diagnostic.message,
                    path=diagnostic.path,
                    source=source_input.source,
                )
            )
    return LoadSourcedPromptTemplatesResult(prompt_templates=prompt_templates, diagnostics=diagnostics)


def _resolve_kind(path: str, diagnostics: List[PromptTemplateDiagnostic]) -> Optional[str]:
    """fileInfo + symlink resolution; returns "file" | "directory" | None (skip)."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as error:
        diagnostics.append(
            PromptTemplateDiagnostic(code="file_info_failed", message=str(error), path=path)
        )
        return None
    if stat_module.S_ISREG(info.st_mode):
        return "file"
    if stat_module.S_ISDIR(info.st_mode):
        return "directory"
    if stat_module.S_ISLNK(info.st_mode):
        # Follow the link (pi's resolveKind via canonicalPath + fileInfo).
        try:
            target = os.stat(path)
        except FileNotFoundError:
            return None
        except OSError as error:
            diagnostics.append(
                PromptTemplateDiagnostic(code="file_info_failed", message=str(error), path=path)
            )
            return None
        if stat_module.S_ISREG(target.st_mode):
            return "file"
        if stat_module.S_ISDIR(target.st_mode):
            return "directory"
    return None


def _load_templates_from_dir(
    dir_path: str,
) -> Tuple[List[PromptTemplate], List[PromptTemplateDiagnostic]]:
    prompt_templates: List[PromptTemplate] = []
    diagnostics: List[PromptTemplateDiagnostic] = []
    try:
        names = os.listdir(dir_path)
    except OSError as error:
        diagnostics.append(PromptTemplateDiagnostic(code="list_failed", message=str(error), path=dir_path))
        return prompt_templates, diagnostics

    for name in sorted(names):
        child = os.path.join(dir_path, name)
        kind = _resolve_kind(child, diagnostics)
        if kind != "file" or not name.endswith(".md"):
            continue
        template, diags = _load_template_from_file(child, name)
        if template is not None:
            prompt_templates.append(template)
        diagnostics.extend(diags)
    return prompt_templates, diagnostics


def _load_template_from_file(
    file_path: str, file_name: str
) -> Tuple[Optional[PromptTemplate], List[PromptTemplateDiagnostic]]:
    diagnostics: List[PromptTemplateDiagnostic] = []
    try:
        raw_content = Path(file_path).read_text(encoding="utf-8")
    except OSError as error:
        diagnostics.append(PromptTemplateDiagnostic(code="read_failed", message=str(error), path=file_path))
        return None, diagnostics

    parsed = _parse_frontmatter(raw_content)
    if parsed is None:
        diagnostics.append(
            PromptTemplateDiagnostic(code="parse_failed", message="Invalid YAML frontmatter", path=file_path)
        )
        return None, diagnostics

    frontmatter, body = parsed
    first_line = next((line for line in body.split("\n") if line.strip()), None)
    description = frontmatter.get("description")
    description = description if isinstance(description, str) else ""
    if not description and first_line:
        description = first_line[:60]
        if len(first_line) > 60:
            description += "..."
    template = PromptTemplate(
        name=re.sub(r"\.md$", "", file_name, flags=re.IGNORECASE),
        description=description,
        content=body,
    )
    return template, diagnostics


def _parse_frontmatter(content: str) -> Optional[Tuple[dict, str]]:
    """Split YAML frontmatter from the body. Returns None on parse failure."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---"):
        return {}, normalized
    end_index = normalized.find("\n---", 3)
    if end_index == -1:
        return {}, normalized
    yaml_string = normalized[4:end_index]
    body = normalized[end_index + 4 :].strip()
    try:
        frontmatter = yaml.safe_load(yaml_string) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(frontmatter, dict):
        frontmatter = {}
    return frontmatter, body


def parse_command_args(args_string: str) -> List[str]:
    """Parse an argument string using simple shell-style single and double quotes."""
    args: List[str] = []
    current = ""
    in_quote: Optional[str] = None

    for char in args_string:
        if in_quote:
            if char == in_quote:
                in_quote = None
            else:
                current += char
        elif char in ('"', "'"):
            in_quote = char
        elif char in (" ", "\t"):
            if current:
                args.append(current)
                current = ""
        else:
            current += char
    if current:
        args.append(current)
    return args


def substitute_args(content: str, args: List[str]) -> str:
    """Substitute prompt template placeholders (``$1``, ``$@``, ``$ARGUMENTS``, ``${@:N}``, ``${@:N:L}``)."""

    def sub_positional(match: re.Match) -> str:
        index = int(match.group(1)) - 1
        return args[index] if 0 <= index < len(args) else ""

    def sub_range(match: re.Match) -> str:
        start = int(match.group(1)) - 1
        if start < 0:
            start = 0
        if match.group(2):
            return " ".join(args[start : start + int(match.group(2))])
        return " ".join(args[start:])

    result = re.sub(r"\$(\d+)", sub_positional, content)
    result = re.sub(r"\$\{@:(\d+)(?::(\d+))?\}", sub_range, result)
    all_args = " ".join(args)
    result = result.replace("$ARGUMENTS", all_args)
    result = result.replace("$@", all_args)
    return result


def format_prompt_template_invocation(template: PromptTemplate, args: Optional[List[str]] = None) -> str:
    """Format a prompt template invocation with positional arguments."""
    return substitute_args(template.content, args or [])

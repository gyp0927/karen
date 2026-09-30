"""The `edit` tool (pi's `harness/tools/edit.ts`)."""

from __future__ import annotations

import errno as errno_module
import json
import os
import stat as stat_module
from pathlib import Path
from typing import Any, Dict, List, Optional

from karen_ai import TextContent

from ..types import AgentTool, AgentToolResult
from .edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_tool_path

_REPLACE_EDIT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "oldText": {
            "type": "string",
            "description": "Exact text for one targeted replacement. It must be unique in the original "
            "file and must not overlap with any other edits[].oldText in the same call.",
        },
        "newText": {"type": "string", "description": "Replacement text for this targeted edit."},
    },
    "required": ["oldText", "newText"],
}

EDIT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
        "edits": {
            "type": "array",
            "items": _REPLACE_EDIT_SCHEMA,
            "description": "One or more targeted replacements. Each edit is matched against the original "
            "file, not incrementally. Do not include overlapping or nested edits. If two changes touch the "
            "same block or nearby lines, merge them into one edit instead.",
        },
    },
    "required": ["path", "edits"],
}

EDIT_DESCRIPTION = (
    "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
    "non-overlapping region of the original file. If two changes affect the same block or nearby lines, "
    "merge them into one edit instead of emitting overlapping edits. Do not include large unchanged "
    "regions just to connect distant changes."
)


def _is_single_edit_input(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("oldText"), str)
        and isinstance(value.get("newText"), str)
    )


def prepare_edit_arguments(input: Any) -> Any:
    """Tolerate common malformed shapes (pi's prepareEditArguments):

    - edits passed as a JSON string (array or single {oldText, newText} object)
    - edits passed as a single edit object instead of an array
    - legacy top-level oldText/newText merged into edits
    """
    if not isinstance(input, dict):
        return input
    args = dict(input)
    edits = args.get("edits")
    if isinstance(edits, str):
        try:
            parsed = json.loads(edits)
            if isinstance(parsed, list):
                args["edits"] = parsed
            elif _is_single_edit_input(parsed):
                args["edits"] = [parsed]
        except ValueError:
            pass
    elif _is_single_edit_input(edits):
        args["edits"] = [edits]

    if not isinstance(args.get("oldText"), str) or not isinstance(args.get("newText"), str):
        return args
    merged_edits: List[Any] = list(args["edits"]) if isinstance(args.get("edits"), list) else []
    merged_edits.append({"oldText": args["oldText"], "newText": args["newText"]})
    result = {key: value for key, value in args.items() if key not in ("oldText", "newText")}
    result["edits"] = merged_edits
    return result


def _validate_edit_input(input: Dict[str, Any]) -> tuple[str, List[Edit]]:
    edits = input.get("edits")
    if not isinstance(edits, list) or len(edits) == 0:
        raise ValueError("Edit tool input is invalid. edits must contain at least one replacement.")
    return input["path"], [Edit(old_text=e["oldText"], new_text=e["newText"]) for e in edits]


def _edit_access_error(path: str, error: OSError) -> RuntimeError:
    code = errno_module.errorcode.get(getattr(error, "errno", None), "UNKNOWN")
    return RuntimeError(f"Could not edit file: {path}. Error code: {code}.")


def create_edit_tool(cwd: Optional[str] = None) -> AgentTool:
    """Create the `edit` tool. `cwd` defaults to the process cwd at call time."""

    async def execute(tool_call_id: str, params: Dict[str, Any], signal, on_update) -> AgentToolResult:
        path, edits = _validate_edit_input(params)
        absolute_path = resolve_tool_path(cwd or os.getcwd(), path)

        async def apply() -> AgentToolResult:
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")
            target = Path(absolute_path)
            try:
                info = os.lstat(absolute_path)
            except OSError as error:
                raise _edit_access_error(path, error) from error
            # pi admits kind "file" | "symlink" (lstat-based); a broken symlink
            # then fails at the read below with an access error.
            if not (stat_module.S_ISREG(info.st_mode) or stat_module.S_ISLNK(info.st_mode)):
                raise RuntimeError(f"Could not edit file: {path}. Path is not a file.")

            try:
                raw = target.read_bytes()
            except OSError as error:
                raise _edit_access_error(path, error) from error
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            bom, content = strip_bom(raw.decode("utf-8", errors="replace"))
            original_ending = detect_line_ending(content)
            normalized_content = normalize_to_lf(content)
            result = apply_edits_to_normalized_content(normalized_content, edits, path)
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            final_content = bom + restore_line_endings(result.new_content, original_ending)
            try:
                target.write_bytes(final_content.encode("utf-8"))
            except OSError as error:
                raise _edit_access_error(path, error) from error
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            diff, first_changed_line = generate_diff_string(result.base_content, result.new_content)
            details: Dict[str, Any] = {
                "diff": diff,
                "patch": generate_unified_patch(path, result.base_content, result.new_content),
            }
            if first_changed_line is not None:
                details["firstChangedLine"] = first_changed_line
            return AgentToolResult(
                content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
                details=details,
            )

        return await with_file_mutation_queue(absolute_path, apply)

    return AgentTool(
        name="edit",
        label="edit",
        description=EDIT_DESCRIPTION,
        parameters=EDIT_SCHEMA,
        prepare_arguments=prepare_edit_arguments,
        execute=execute,
    )


__all__ = ["EDIT_DESCRIPTION", "EDIT_SCHEMA", "create_edit_tool", "prepare_edit_arguments"]

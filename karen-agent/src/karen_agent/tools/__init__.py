"""Built-in agent tools (pi's `harness/tools/`): read, write, edit, bash.

Each factory returns a plain `AgentTool`; hand them to the loop via
`AgentContext(tools=[...])`. All factories take an optional `cwd` (defaults to
the process cwd at call time) in place of pi's ExecutionToolContext.
"""

from __future__ import annotations

import os
from typing import List, Optional

from ..types import AgentTool
from .bash import (
    BASH_DESCRIPTION,
    BASH_SCHEMA,
    BashExecution,
    BashPrepare,
    create_bash_tool,
    create_shell_tool,
)
from .edit import EDIT_DESCRIPTION, EDIT_SCHEMA, create_edit_tool, prepare_edit_arguments
from .edit_diff import (
    AppliedEditsResult,
    Edit,
    FuzzyMatchResult,
    apply_edits_to_normalized_content,
    detect_line_ending,
    fuzzy_find_text,
    generate_diff_string,
    generate_unified_patch,
    normalize_for_fuzzy_match,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .file_mutation_queue import with_file_mutation_queue
from .image import detect_supported_image_mime_type, encode_base64
from .local_shell import (
    MAX_TIMEOUT_SECONDS,
    ExecutionError,
    ShellConfig,
    resolve_shell_config,
    run_shell_command,
    validate_timeout,
)
from .path_utils import normalize_tool_path, resolve_read_tool_path, resolve_tool_path
from .read import (
    READ_DESCRIPTION,
    READ_SCHEMA,
    ImageProcessingFailed,
    ProcessedImage,
    ReadImageProcessor,
    ReadImageProcessorResult,
    create_read_tool,
)
from .write import WRITE_DESCRIPTION, WRITE_SCHEMA, create_write_tool


def create_builtin_tools(cwd: Optional[str] = None, *, shell_path: Optional[str] = None) -> List[AgentTool]:
    """The four built-in tools sharing one cwd (karen convenience; pi has no equivalent)."""
    return [
        create_read_tool(cwd),
        create_write_tool(cwd),
        create_edit_tool(cwd),
        create_bash_tool(cwd, shell_path=shell_path),
    ]


__all__ = [
    "BASH_DESCRIPTION",
    "BASH_SCHEMA",
    "BashExecution",
    "BashPrepare",
    "EDIT_DESCRIPTION",
    "EDIT_SCHEMA",
    "MAX_TIMEOUT_SECONDS",
    "READ_DESCRIPTION",
    "READ_SCHEMA",
    "WRITE_DESCRIPTION",
    "WRITE_SCHEMA",
    "AppliedEditsResult",
    "Edit",
    "ExecutionError",
    "FuzzyMatchResult",
    "ImageProcessingFailed",
    "ProcessedImage",
    "ReadImageProcessor",
    "ReadImageProcessorResult",
    "ShellConfig",
    "apply_edits_to_normalized_content",
    "create_bash_tool",
    "create_shell_tool",
    "create_builtin_tools",
    "create_edit_tool",
    "create_read_tool",
    "create_write_tool",
    "detect_line_ending",
    "detect_supported_image_mime_type",
    "encode_base64",
    "fuzzy_find_text",
    "generate_diff_string",
    "generate_unified_patch",
    "normalize_for_fuzzy_match",
    "normalize_to_lf",
    "normalize_tool_path",
    "prepare_edit_arguments",
    "resolve_read_tool_path",
    "resolve_shell_config",
    "resolve_tool_path",
    "restore_line_endings",
    "run_shell_command",
    "strip_bom",
    "validate_timeout",
    "with_file_mutation_queue",
]

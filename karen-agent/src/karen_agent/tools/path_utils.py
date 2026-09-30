"""Tool path normalization and resolution (pi's `harness/tools/path-utils.ts`).

karen drops pi's FileSystem capability: paths resolve against an explicit cwd
with `os.path` (absolute-resolution only, no symlink chasing — pi's
`env.absolutePath`).
"""

from __future__ import annotations

import os
import re
import unicodedata

UNICODE_SPACES = re.compile("[  -   　]")
NARROW_NO_BREAK_SPACE = " "
_AM_PM_SPACE = re.compile(r" (AM|PM)\.", re.IGNORECASE)


def normalize_tool_path(path: str) -> str:
    """Replace exotic Unicode spaces and strip a leading '@' (pi behavior)."""
    normalized = UNICODE_SPACES.sub(" ", path)
    return normalized[1:] if normalized.startswith("@") else normalized


def resolve_tool_path(cwd: str, path: str) -> str:
    """Resolve a (normalized) tool path against `cwd` to an absolute path."""
    normalized = normalize_tool_path(path)
    if os.path.isabs(normalized):
        return os.path.abspath(normalized)
    return os.path.abspath(os.path.join(cwd, normalized))


def resolve_read_tool_path(cwd: str, path: str) -> str:
    """Resolve a read path, trying Unicode variants produced by macOS/iOS.

    Variants tried in order (first existing wins): the resolved path, AM/PM
    preceded by a narrow no-break space, NFD normalization, curly apostrophe,
    and NFD + curly apostrophe.
    """
    resolved = resolve_tool_path(cwd, path)
    variants = [
        resolved,
        _AM_PM_SPACE.sub(lambda m: f"{NARROW_NO_BREAK_SPACE}{m.group(1)}.", resolved),
        unicodedata.normalize("NFD", resolved),
        resolved.replace("'", "’"),
        unicodedata.normalize("NFD", resolved).replace("'", "’"),
    ]
    seen = set()
    for variant in variants:
        if variant in seen:
            continue
        seen.add(variant)
        if os.path.exists(variant):
            return variant
    return resolved


__all__ = ["normalize_tool_path", "resolve_read_tool_path", "resolve_tool_path"]

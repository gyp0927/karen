"""Glob-to-regex translation for the find/grep tools.

Two dialects, both documented where they differ from pi's binaries (karen
implements the walks in-process instead of shelling out to fd/rg — see
`walk.py`):

- **fd-style** (`find`): a pattern without `/` matches the basename; a
  pattern containing `/` matches the full search-root-relative path, with an
  implicit `**/` prefix unless it starts with `/`, `**/`, or is exactly `**`
  (pi passes `--full-path` and rewrites the pattern the same way). A leading
  `/` anchors the pattern at the search root.
- **gitignore-style** (`grep --glob`, ripgrep semantics): a pattern without
  `/` matches the basename at any depth; a pattern containing `/` is anchored
  at the search root. No negation support (`!` is matched literally).
"""

from __future__ import annotations

import re


def translate_glob(pattern: str) -> "re.Pattern[str]":
    """Compile one glob pattern to a full-match regex.

    `**/` becomes an optional any-depth prefix, a trailing/standalone `**`
    matches everything including separators, `*` matches within one path
    segment, `?` one non-separator char, and `[...]`/`[!...]` character
    classes pass through.
    """
    out = []
    i = 0
    n = len(pattern)
    while i < n:
        char = pattern[i]
        if char == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                # consume the run of stars
                j = i
                while j < n and pattern[j] == "*":
                    j += 1
                if j < n and pattern[j] == "/":
                    out.append("(?:.*/)?")
                    j += 1
                else:
                    out.append(".*")
                i = j
            else:
                out.append("[^/]*")
                i += 1
        elif char == "?":
            out.append("[^/]")
            i += 1
        elif char == "[":
            end = i + 1
            if end < n and pattern[end] in "!^":
                end += 1
            if end < n and pattern[end] == "]":
                end += 1
            while end < n and pattern[end] != "]":
                end += 1
            if end >= n:
                out.append(re.escape("["))
                i += 1
            else:
                body = pattern[i + 1 : end]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body + "]")
                i = end + 1
        else:
            out.append(re.escape(char))
            i += 1
    return re.compile("(?s)" + "".join(out) + r"\Z")


def compile_find_matcher(pattern: str):
    """fd-style matcher over search-root-relative posix paths (and basenames)."""
    if "/" not in pattern:
        regex = translate_glob(pattern)
        return lambda rel_path, is_dir: regex.fullmatch(rel_path.rsplit("/", 1)[-1]) is not None
    if pattern.startswith("/"):
        regex = translate_glob(pattern[1:])
        return lambda rel_path, is_dir: regex.fullmatch(rel_path) is not None
    effective = pattern if pattern.startswith("**/") or pattern == "**" else f"**/{pattern}"
    regex = translate_glob(effective)
    return lambda rel_path, is_dir: regex.fullmatch(rel_path) is not None


def compile_grep_glob_matcher(pattern: str):
    """gitignore-style matcher for `grep`'s --glob file filter."""
    if "/" not in pattern:
        regex = translate_glob(pattern)
        return lambda rel_path: regex.fullmatch(rel_path.rsplit("/", 1)[-1]) is not None
    regex = translate_glob(pattern.lstrip("/"))
    return lambda rel_path: regex.fullmatch(rel_path) is not None

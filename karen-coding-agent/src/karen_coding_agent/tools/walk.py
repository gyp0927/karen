"""Ignore-aware directory walker shared by the find and grep tools.

karen implements pi's fd/ripgrep walks in-process: pi shells out to fd/rg
binaries downloaded on demand from GitHub releases, which this environment
will not run. Semantics preserved:

- `.git` directories are always pruned;
- hidden files are included (pi passes `--hidden`);
- symlinks are not descended into (fd/rg default; the link itself is yielded
  as a file entry);
- `.gitignore` and `.ignore` files are respected; a deeper file overrides a
  shallower one, and within one directory `.ignore` outranks `.gitignore`
  (ripgrep precedence);
- entries are yielded sorted case-insensitively per directory — fd/rg output
  order is unspecified, so this only pins down what pi leaves open.

Known deviations: `.gitignore` files in parents of the search root and the
global gitignore are not consulted (fd applies them inside a git repo), and
`.fdignore` is not read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

from pathspec import GitIgnoreSpec

#: Ignore file names within one directory, weakest first (ripgrep precedence).
IGNORE_FILE_NAMES = (".gitignore", ".ignore")

#: (rel-prefix the spec is rooted at, spec); deeper specs come last.
_SpecStack = Tuple[Tuple[str, GitIgnoreSpec], ...]


@dataclass
class WalkEntry:
    """One walked path."""

    path: Path  # absolute
    rel_path: str  # posix, relative to the walk root, no trailing "/"
    is_dir: bool


def _load_specs(directory: Path) -> List[GitIgnoreSpec]:
    specs: List[GitIgnoreSpec] = []
    for name in IGNORE_FILE_NAMES:
        try:
            text = (directory / name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        specs.append(GitIgnoreSpec.from_lines(text.splitlines()))
    return specs


def _is_ignored(rel_path: str, is_dir: bool, specs: _SpecStack) -> bool:
    """Deepest spec with an opinion wins; `include=True` means ignored."""
    check_path = rel_path + "/" if is_dir else rel_path
    for prefix, spec in reversed(specs):
        include = spec.check_file(check_path[len(prefix):]).include
        if include is not None:
            return include
    return False


def walk(root: Path, *, include_dirs: bool = False, respect_ignore: bool = True) -> Iterator[WalkEntry]:
    """Yield entries under `root` depth-first, pruned by the ignore rules."""
    root = Path(root)
    specs: _SpecStack = tuple(("", spec) for spec in _load_specs(root)) if respect_ignore else ()
    yield from _walk(root, "", specs, include_dirs, respect_ignore)


def _walk(
    directory: Path,
    rel_prefix: str,
    specs: _SpecStack,
    include_dirs: bool,
    respect_ignore: bool,
) -> Iterator[WalkEntry]:
    try:
        children = sorted(os.scandir(directory), key=lambda entry: entry.name.lower())
    except OSError:
        return
    for child in children:
        is_dir = child.is_dir(follow_symlinks=False)
        if is_dir and child.name == ".git":
            continue
        rel_path = f"{rel_prefix}{child.name}"
        if respect_ignore and _is_ignored(rel_path, is_dir, specs):
            continue
        child_path = Path(child.path)
        if is_dir:
            if include_dirs:
                yield WalkEntry(child_path, rel_path, True)
            child_prefix = f"{rel_path}/"
            child_specs = (
                specs + tuple((child_prefix, spec) for spec in _load_specs(child_path))
                if respect_ignore
                else ()
            )
            yield from _walk(child_path, child_prefix, child_specs, include_dirs, respect_ignore)
        else:
            yield WalkEntry(child_path, rel_path, False)

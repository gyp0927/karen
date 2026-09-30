"""Shared diff computation utilities for the edit tool (pi's `tools/edit-diff.ts`).

Exact-match first, then fuzzy matching in a normalized space (trailing
whitespace stripped, smart quotes/dashes/spaces folded to ASCII). When fuzzy
matching is used, replacements are computed in normalized space and overlaid
back onto the original content line-wise so unchanged lines keep their bytes.

Diff generation uses difflib's SequenceMatcher (autojunk off) in place of
jsdiff; both produce GNU-compatible unified hunks.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from karen_ai.types import KarenBase

_LINE_WITH_ENDING = re.compile(r"[^\n]*\n|[^\n]+")
_SMART_SINGLE_QUOTES = re.compile("[‘’‚‛]")
_SMART_DOUBLE_QUOTES = re.compile("[“”„‟]")
_UNICODE_DASHES = re.compile("[‐‑‒–—―−]")
_SPECIAL_SPACES = re.compile("[  -   　]")


def detect_line_ending(content: str) -> str:
    """Return "\\r\\n" or "\\n" based on which newline appears first."""
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1:
        return "\n"
    if crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fuzzy matching (pi's progressive transformations)."""
    text = unicodedata.normalize("NFKC", text)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _SMART_SINGLE_QUOTES.sub("'", text)  # smart single quotes → '
    text = _SMART_DOUBLE_QUOTES.sub('"', text)  # smart double quotes → "
    # U+2010 hyphen, U+2011 non-breaking hyphen, U+2012 figure dash, U+2013 en-dash,
    # U+2014 em-dash, U+2015 horizontal bar, U+2212 minus → -
    text = _UNICODE_DASHES.sub("-", text)
    # U+00A0 NBSP, U+2002-U+200A, U+202F narrow NBSP, U+205F, U+3000 → regular space
    text = _SPECIAL_SPACES.sub(" ", text)
    return text


def split_lines_with_endings(content: str) -> List[str]:
    return _LINE_WITH_ENDING.findall(content)


@dataclass
class _LineSpan:
    start: int
    end: int


class Edit(KarenBase):
    """One targeted replacement (pi's `{oldText, newText}`)."""

    old_text: str
    new_text: str


@dataclass
class _MatchedEdit:
    edit_index: int
    match_index: int
    match_length: int
    new_text: str


@dataclass
class FuzzyMatchResult:
    found: bool
    #: Index where the match starts (in the content to use for replacement).
    index: int
    #: Length of the matched text.
    match_length: int
    #: Whether fuzzy matching was used (False = exact match).
    used_fuzzy_match: bool
    #: Content to use for replacement operations: original on exact match,
    #: fuzzy-normalized content otherwise.
    content_for_replacement: str


@dataclass
class AppliedEditsResult:
    base_content: str
    new_content: str


def _get_line_spans(content: str) -> List[_LineSpan]:
    offset = 0
    spans = []
    for line in split_lines_with_endings(content):
        spans.append(_LineSpan(start=offset, end=offset + len(line)))
        offset = spans[-1].end
    return spans


def _get_replacement_line_range(lines: List[_LineSpan], match_index: int, match_length: int) -> Tuple[int, int]:
    replacement_start = match_index
    replacement_end = match_index + match_length

    start_line = -1
    for i, line in enumerate(lines):
        if line.start <= replacement_start < line.end:
            start_line = i
            break
    if start_line == -1:
        raise ValueError("Replacement range is outside the base content.")

    end_line = start_line
    while end_line < len(lines) and lines[end_line].end < replacement_end:
        end_line += 1
    if end_line >= len(lines):
        raise ValueError("Replacement range is outside the base content.")

    return start_line, end_line + 1


def _apply_replacements(content: str, replacements: List[_MatchedEdit], offset: int = 0) -> str:
    result = content
    for replacement in reversed(replacements):
        match_index = replacement.match_index - offset
        result = (
            result[:match_index] + replacement.new_text + result[match_index + replacement.match_length :]
        )
    return result


def apply_replacements_preserving_unchanged_lines(
    original_content: str, base_content: str, replacements: List[_MatchedEdit]
) -> str:
    """Apply replacements matched against `base_content` to `original_content`
    while preserving unchanged line blocks from the original.

    Each replacement is widened to the lines it touches; touched lines are
    rewritten from the normalized base, all other lines are copied back from
    `original_content`.
    """
    original_lines = split_lines_with_endings(original_content)
    base_lines = _get_line_spans(base_content)
    if len(original_lines) != len(base_lines):
        raise ValueError(
            "Cannot preserve unchanged lines because the base content has a different line count."
        )

    groups: List[dict] = []
    for replacement in sorted(replacements, key=lambda r: r.match_index):
        start_line, end_line = _get_replacement_line_range(
            base_lines, replacement.match_index, replacement.match_length
        )
        current = groups[-1] if groups else None
        if current and start_line < current["end_line"]:
            current["end_line"] = max(current["end_line"], end_line)
            current["replacements"].append(replacement)
            continue
        groups.append({"start_line": start_line, "end_line": end_line, "replacements": [replacement]})

    original_line_index = 0
    result = ""
    for group in groups:
        result += "".join(original_lines[original_line_index : group["start_line"]])
        group_start_offset = base_lines[group["start_line"]].start
        group_end_offset = base_lines[group["end_line"] - 1].end
        result += _apply_replacements(
            base_content[group_start_offset:group_end_offset], group["replacements"], group_start_offset
        )
        original_line_index = group["end_line"]
    result += "".join(original_lines[original_line_index:])
    return result


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    """Find old_text in content, exact match first, then fuzzy match."""
    exact_index = content.find(old_text)
    if exact_index != -1:
        return FuzzyMatchResult(
            found=True,
            index=exact_index,
            match_length=len(old_text),
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)

    if fuzzy_index == -1:
        return FuzzyMatchResult(
            found=False,
            index=-1,
            match_length=0,
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    return FuzzyMatchResult(
        found=True,
        index=fuzzy_index,
        match_length=len(fuzzy_old_text),
        used_fuzzy_match=True,
        content_for_replacement=fuzzy_content,
    )


def strip_bom(content: str) -> Tuple[str, str]:
    """Strip UTF-8 BOM if present; return (bom, text_without_bom)."""
    return ("﻿", content[1:]) if content.startswith("﻿") else ("", content)


def _count_occurrences(content: str, old_text: str) -> int:
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    return fuzzy_content.count(fuzzy_old_text)


def _not_found_error(path: str, edit_index: int, total_edits: int) -> ValueError:
    if total_edits == 1:
        return ValueError(
            f"Could not find the exact text in {path}. The old text must match exactly "
            "including all whitespace and newlines."
        )
    return ValueError(
        f"Could not find edits[{edit_index}] in {path}. The oldText must match exactly "
        "including all whitespace and newlines."
    )


def _duplicate_error(path: str, edit_index: int, total_edits: int, occurrences: int) -> ValueError:
    if total_edits == 1:
        return ValueError(
            f"Found {occurrences} occurrences of the text in {path}. The text must be unique. "
            "Please provide more context to make it unique."
        )
    return ValueError(
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. Each oldText must be "
        "unique. Please provide more context to make it unique."
    )


def _empty_old_text_error(path: str, edit_index: int, total_edits: int) -> ValueError:
    if total_edits == 1:
        return ValueError(f"oldText must not be empty in {path}.")
    return ValueError(f"edits[{edit_index}].oldText must not be empty in {path}.")


def _no_change_error(path: str, total_edits: int) -> ValueError:
    if total_edits == 1:
        return ValueError(
            f"No changes made to {path}. The replacement produced identical content. This might "
            "indicate an issue with special characters or the text not existing as expected."
        )
    return ValueError(f"No changes made to {path}. The replacements produced identical content.")


def apply_edits_to_normalized_content(
    normalized_content: str, edits: List[Edit], path: str
) -> AppliedEditsResult:
    """Apply one or more exact-text replacements to LF-normalized content.

    All edits are matched against the same original content, then applied in
    reverse order so offsets stay stable. If any edit needs fuzzy matching, the
    operation runs in fuzzy-normalized space and overlays the changes back onto
    the original content line-wise.
    """
    normalized_edits = [Edit(old_text=normalize_to_lf(e.old_text), new_text=normalize_to_lf(e.new_text)) for e in edits]

    for i, edit in enumerate(normalized_edits):
        if len(edit.old_text) == 0:
            raise _empty_old_text_error(path, i, len(normalized_edits))

    initial_matches = [fuzzy_find_text(normalized_content, edit.old_text) for edit in normalized_edits]
    used_fuzzy_match = any(match.used_fuzzy_match for match in initial_matches)
    replacement_base_content = (
        normalize_for_fuzzy_match(normalized_content) if used_fuzzy_match else normalized_content
    )

    matched_edits: List[_MatchedEdit] = []
    for i, edit in enumerate(normalized_edits):
        match_result = fuzzy_find_text(replacement_base_content, edit.old_text)
        if not match_result.found:
            raise _not_found_error(path, i, len(normalized_edits))

        occurrences = _count_occurrences(replacement_base_content, edit.old_text)
        if occurrences > 1:
            raise _duplicate_error(path, i, len(normalized_edits), occurrences)

        matched_edits.append(
            _MatchedEdit(
                edit_index=i,
                match_index=match_result.index,
                match_length=match_result.match_length,
                new_text=edit.new_text,
            )
        )

    matched_edits.sort(key=lambda m: m.match_index)
    for i in range(1, len(matched_edits)):
        previous = matched_edits[i - 1]
        current = matched_edits[i]
        if previous.match_index + previous.match_length > current.match_index:
            raise ValueError(
                f"edits[{previous.edit_index}] and edits[{current.edit_index}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    base_content = normalized_content
    new_content = (
        apply_replacements_preserving_unchanged_lines(normalized_content, replacement_base_content, matched_edits)
        if used_fuzzy_match
        else _apply_replacements(replacement_base_content, matched_edits)
    )

    if base_content == new_content:
        raise _no_change_error(path, len(normalized_edits))

    return AppliedEditsResult(base_content=base_content, new_content=new_content)


# ---------------------------------------------------------------------------
# Diff / patch generation
# ---------------------------------------------------------------------------


@dataclass
class _DiffPart:
    value: str
    added: bool = False
    removed: bool = False


def _diff_line_parts(old_content: str, new_content: str) -> List[_DiffPart]:
    """Line-level diff parts in jsdiff's diffLines shape (removals before additions)."""
    old_tokens = split_lines_with_endings(old_content)
    new_tokens = split_lines_with_endings(new_content)
    matcher = SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
    parts: List[_DiffPart] = []
    for tag, a1, a2, b1, b2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(_DiffPart(value="".join(old_tokens[a1:a2])))
        elif tag == "delete":
            parts.append(_DiffPart(value="".join(old_tokens[a1:a2]), removed=True))
        elif tag == "insert":
            parts.append(_DiffPart(value="".join(new_tokens[b1:b2]), added=True))
        else:  # replace → removed part, then added part (jsdiff ordering)
            parts.append(_DiffPart(value="".join(old_tokens[a1:a2]), removed=True))
            parts.append(_DiffPart(value="".join(new_tokens[b1:b2]), added=True))
    return parts


def generate_diff_string(
    old_content: str, new_content: str, context_lines: int = 4
) -> Tuple[str, Optional[int]]:
    """Display-oriented diff with line numbers and context.

    Returns (diff, first_changed_line) where the line number is in the new file.
    """
    parts = _diff_line_parts(old_content, new_content)
    output: List[str] = []

    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")
    max_line_num = max(len(old_lines), len(new_lines))
    line_num_width = len(str(max_line_num))

    old_line_num = 1
    new_line_num = 1
    last_was_change = False
    first_changed_line: Optional[int] = None

    for i, part in enumerate(parts):
        raw = part.value.split("\n")
        if raw and raw[-1] == "":
            raw.pop()

        if part.added or part.removed:
            if first_changed_line is None:
                first_changed_line = new_line_num
            for line in raw:
                if part.added:
                    output.append(f"+{str(new_line_num).rjust(line_num_width)} {line}")
                    new_line_num += 1
                else:
                    output.append(f"-{str(old_line_num).rjust(line_num_width)} {line}")
                    old_line_num += 1
            last_was_change = True
        else:
            next_part_is_change = i < len(parts) - 1 and (parts[i + 1].added or parts[i + 1].removed)
            has_leading_change = last_was_change
            has_trailing_change = next_part_is_change
            skip_marker = f" {' ' * line_num_width} ..."

            if has_leading_change and has_trailing_change:
                if len(raw) <= context_lines * 2:
                    for line in raw:
                        output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                        old_line_num += 1
                        new_line_num += 1
                else:
                    for line in raw[:context_lines]:
                        output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                        old_line_num += 1
                        new_line_num += 1
                    skipped = len(raw) - context_lines * 2
                    output.append(skip_marker)
                    old_line_num += skipped
                    new_line_num += skipped
                    for line in raw[len(raw) - context_lines :]:
                        output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                        old_line_num += 1
                        new_line_num += 1
            elif has_leading_change:
                shown = raw[:context_lines]
                for line in shown:
                    output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                    old_line_num += 1
                    new_line_num += 1
                skipped = len(raw) - len(shown)
                if skipped > 0:
                    output.append(skip_marker)
                    old_line_num += skipped
                    new_line_num += skipped
            elif has_trailing_change:
                skipped = max(0, len(raw) - context_lines)
                if skipped > 0:
                    output.append(skip_marker)
                    old_line_num += skipped
                    new_line_num += skipped
                for line in raw[skipped:]:
                    output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                    old_line_num += 1
                    new_line_num += 1
            else:
                old_line_num += len(raw)
                new_line_num += len(raw)

            last_was_change = False

    return "\n".join(output), first_changed_line


def _format_range_unified(start: int, count: int) -> str:
    """Convert a 0-based start + line count to the unified-diff range format."""
    beginning = start + 1  # lines start numbering with one
    if count == 1:
        return str(beginning)
    if count == 0:
        beginning -= 1  # empty ranges begin at the line just before the range
    return f"{beginning},{count}"


def generate_unified_patch(path: str, old_content: str, new_content: str, context_lines: int = 4) -> str:
    """Standard unified patch in jsdiff's createTwoFilesPatch shape
    (Index/=== header, file-name-only ---/+++ lines, no-newline markers)."""
    old_lines = split_lines_with_endings(old_content)
    new_lines = split_lines_with_endings(new_content)
    matcher = SequenceMatcher(None, old_lines, new_lines, autojunk=False)

    chunks = [f"Index: {path}\n{'=' * 67}\n--- {path}\n+++ {path}\n"]

    def emit(prefix: str, token: str) -> None:
        chunks.append(prefix + token)
        if not token.endswith("\n"):
            chunks.append("\n\\ No newline at end of file\n")

    for group in matcher.get_grouped_opcodes(context_lines):
        first, last = group[0], group[-1]
        old_count = last[2] - first[1]
        new_count = last[4] - first[3]
        old_range = _format_range_unified(first[1], old_count)
        new_range = _format_range_unified(first[3], new_count)
        chunks.append(f"@@ -{old_range} +{new_range} @@\n")
        for tag, a1, a2, b1, b2 in group:
            if tag == "equal":
                for token in old_lines[a1:a2]:
                    emit(" ", token)
            elif tag == "delete":
                for token in old_lines[a1:a2]:
                    emit("-", token)
            elif tag == "insert":
                for token in new_lines[b1:b2]:
                    emit("+", token)
            else:  # replace → removals first, then additions
                for token in old_lines[a1:a2]:
                    emit("-", token)
                for token in new_lines[b1:b2]:
                    emit("+", token)

    return "".join(chunks)


__all__ = [
    "AppliedEditsResult",
    "Edit",
    "FuzzyMatchResult",
    "apply_edits_to_normalized_content",
    "apply_replacements_preserving_unchanged_lines",
    "detect_line_ending",
    "fuzzy_find_text",
    "generate_diff_string",
    "generate_unified_patch",
    "normalize_for_fuzzy_match",
    "normalize_to_lf",
    "restore_line_endings",
    "split_lines_with_endings",
    "strip_bom",
]

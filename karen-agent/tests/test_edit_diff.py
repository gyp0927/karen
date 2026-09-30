"""Edit application + diff generation (pi's edit-diff.ts behaviors)."""

import pytest

from karen_agent.tools.edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    fuzzy_find_text,
    generate_diff_string,
    generate_unified_patch,
    normalize_for_fuzzy_match,
    normalize_to_lf,
    restore_line_endings,
    split_lines_with_endings,
    strip_bom,
)


def _edit(old, new):
    return Edit(old_text=old, new_text=new)


# --- line endings / BOM ----------------------------------------------------


def test_detect_line_ending():
    assert detect_line_ending("a\r\nb\n") == "\r\n"  # CRLF appears first
    assert detect_line_ending("a\nb\r\n") == "\n"
    assert detect_line_ending("no newlines") == "\n"
    assert detect_line_ending("\n\r\n") == "\n"


def test_normalize_and_restore_line_endings():
    assert normalize_to_lf("a\r\nb\rc\n") == "a\nb\nc\n"
    assert restore_line_endings("a\nb\n", "\r\n") == "a\r\nb\r\n"
    assert restore_line_endings("a\n", "\n") == "a\n"


def test_split_lines_with_endings():
    assert split_lines_with_endings("") == []
    assert split_lines_with_endings("a\n") == ["a\n"]
    assert split_lines_with_endings("a\nb") == ["a\n", "b"]
    assert split_lines_with_endings("a\n\nb\n") == ["a\n", "\n", "b\n"]


def test_strip_bom():
    assert strip_bom("﻿abc") == ("﻿", "abc")
    assert strip_bom("abc") == ("", "abc")


# --- fuzzy normalization ---------------------------------------------------


def test_normalize_for_fuzzy_match():
    assert normalize_for_fuzzy_match("it’s “quoted”") == 'it\'s "quoted"'
    assert normalize_for_fuzzy_match("a—b–c−d") == "a-b-c-d"
    assert normalize_for_fuzzy_match("a b c") == "a b c"
    # Per-line trimEnd keeps the trailing empty segment (same as pi).
    assert normalize_for_fuzzy_match("line   \nnext\t\n") == "line\nnext\n"
    assert normalize_for_fuzzy_match("ﬁle") == "file"  # NFKC ligature


def test_fuzzy_find_exact_first():
    result = fuzzy_find_text('say "hi"', 'say "hi"')
    assert result.found and not result.used_fuzzy_match
    assert result.index == 0


def test_fuzzy_find_normalizes_smart_quotes():
    result = fuzzy_find_text('say “hi”', 'say "hi"')
    assert result.found and result.used_fuzzy_match
    assert result.content_for_replacement == 'say "hi"'


def test_fuzzy_find_missing():
    assert fuzzy_find_text("abc", "xyz").found is False


# --- apply_edits_to_normalized_content --------------------------------------


def test_apply_single_edit():
    result = apply_edits_to_normalized_content("hello world\nfoo\n", [_edit("world", "there")], "f")
    assert result.new_content == "hello there\nfoo\n"
    assert result.base_content == "hello world\nfoo\n"


def test_apply_multiple_disjoint_edits():
    result = apply_edits_to_normalized_content(
        "a=1\nb=2\nc=3\n", [_edit("a=1", "a=10"), _edit("c=3", "c=30")], "f"
    )
    assert result.new_content == "a=10\nb=2\nc=30\n"


def test_empty_old_text_error_messages():
    with pytest.raises(ValueError, match=r"^oldText must not be empty in f\.$"):
        apply_edits_to_normalized_content("abc", [_edit("", "x")], "f")
    with pytest.raises(ValueError, match=r"^edits\[1\]\.oldText must not be empty in f\.$"):
        apply_edits_to_normalized_content("abc", [_edit("a", "x"), _edit("", "y")], "f")


def test_not_found_error_messages():
    with pytest.raises(ValueError) as single:
        apply_edits_to_normalized_content("abc", [_edit("xyz", "x")], "f")
    assert str(single.value) == (
        "Could not find the exact text in f. The old text must match exactly including all "
        "whitespace and newlines."
    )
    with pytest.raises(ValueError) as multi:
        apply_edits_to_normalized_content("abc", [_edit("a", "x"), _edit("xyz", "y")], "f")
    assert str(multi.value) == (
        "Could not find edits[1] in f. The oldText must match exactly including all whitespace "
        "and newlines."
    )


def test_duplicate_occurrence_error_messages():
    with pytest.raises(ValueError) as single:
        apply_edits_to_normalized_content("x\nx\n", [_edit("x", "y")], "f")
    assert str(single.value) == (
        "Found 2 occurrences of the text in f. The text must be unique. Please provide more "
        "context to make it unique."
    )
    with pytest.raises(ValueError) as multi:
        apply_edits_to_normalized_content("x\nx\ny", [_edit("y", "z"), _edit("x", "w")], "f")
    assert "Found 2 occurrences of edits[1] in f." in str(multi.value)


def test_overlapping_edits_error():
    with pytest.raises(ValueError) as excinfo:
        apply_edits_to_normalized_content("abcdef", [_edit("abc", "x"), _edit("bcd", "y")], "f")
    assert str(excinfo.value) == (
        "edits[0] and edits[1] overlap in f. Merge them into one edit or target disjoint regions."
    )


def test_overlap_error_uses_sorted_positions():
    with pytest.raises(ValueError) as excinfo:
        apply_edits_to_normalized_content("abcdef", [_edit("bcd", "y"), _edit("abc", "x")], "f")
    assert str(excinfo.value).startswith("edits[1] and edits[0] overlap")


def test_no_change_error():
    with pytest.raises(ValueError) as single:
        apply_edits_to_normalized_content("abc", [_edit("abc", "abc")], "f")
    assert "The replacement produced identical content." in str(single.value)
    with pytest.raises(ValueError) as multi:
        apply_edits_to_normalized_content("abc", [_edit("a", "a"), _edit("b", "b")], "f")
    assert str(multi.value) == "No changes made to f. The replacements produced identical content."


def test_fuzzy_edit_preserves_unchanged_lines():
    # Smart quotes elsewhere in the file must survive even though the matched
    # line is replaced in normalized (ASCII-quote) space.
    content = 'a “x” b\nkeep “y”  \n'
    result = apply_edits_to_normalized_content(content, [_edit('a "x" b', 'a "z" b')], "f")
    assert result.new_content == 'a "z" b\nkeep “y”  \n'


# --- generate_diff_string ---------------------------------------------------


def test_diff_string_simple_change():
    old = "line1\nline2\nline3\nline4\nline5\n"
    new = "line1\nCHANGED\nline3\nline4\nline5\n"
    diff, first = generate_diff_string(old, new)
    assert first == 2
    assert diff.split("\n") == [
        " 1 line1",
        "-2 line2",
        "+2 CHANGED",
        " 3 line3",
        " 4 line4",
        " 5 line5",
    ]


def test_diff_string_skips_long_unchanged_middle():
    old_lines = [f"line{i}" for i in range(1, 31)]
    new_lines = list(old_lines)
    new_lines[4] = "CHANGED_A"
    new_lines[24] = "CHANGED_B"
    diff, first = generate_diff_string("\n".join(old_lines) + "\n", "\n".join(new_lines) + "\n")
    assert first == 5
    rows = diff.split("\n")
    # 31 lines → line numbers are padded to width 2 (pi padStart behavior).
    assert "- 5 line5" in rows
    assert "+ 5 CHANGED_A" in rows
    assert "-25 line25" in rows
    assert "+25 CHANGED_B" in rows
    assert rows.count("    ...") == 2  # middle gap + trailing gap (" " + " "*width + " ...")


def test_diff_string_first_changed_line_none_when_identical():
    diff, first = generate_diff_string("a\nb\n", "a\nb\n")
    assert diff == ""
    assert first is None


# --- generate_unified_patch --------------------------------------------------


def test_unified_patch_simple_change():
    patch = generate_unified_patch("f.txt", "a\nb\nc\n", "a\nX\nc\n")
    assert patch == (
        "Index: f.txt\n"
        + "=" * 67
        + "\n--- f.txt\n+++ f.txt\n"
        + "@@ -1,3 +1,3 @@\n"
        + " a\n-b\n+X\n c\n"
    )


def test_unified_patch_no_newline_marker():
    patch = generate_unified_patch("f.txt", "a\nb", "a\nc")
    assert "-b\n\\ No newline at end of file\n+c\n\\ No newline at end of file\n" in patch


def test_unified_patch_insert_at_top():
    patch = generate_unified_patch("f.txt", "b\n", "a\nb\n")
    # context_lines=4 pulls the following unchanged line into the hunk (GNU behavior).
    assert "@@ -1 +1,2 @@\n+a\n b\n" in patch


def test_unified_patch_respects_context_lines():
    old = "".join(f"l{i}\n" for i in range(1, 21))
    new = old.replace("l5\n", "X5\n").replace("l15\n", "X15\n")
    patch = generate_unified_patch("f", old, new, context_lines=2)
    assert patch.count("@@") == 4  # two hunks (each header has one opening and one closing @@)
    assert "l1\n" not in patch  # far-away context excluded

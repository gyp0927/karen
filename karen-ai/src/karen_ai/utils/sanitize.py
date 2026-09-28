"""Lone-surrogate cleanup for outgoing text, mirroring utils/sanitize-unicode.ts."""

from __future__ import annotations


def sanitize_surrogates(text: str) -> str:
    """Remove lone UTF-16 surrogates that strict JSON encoders reject."""
    return text.encode("utf-8", "replace").decode("utf-8", "replace") if _has_lone_surrogate(text) else text


def _has_lone_surrogate(text: str) -> bool:
    for i, ch in enumerate(text):
        code = ord(ch)
        if 0xD800 <= code <= 0xDBFF:
            # High surrogate: valid only when followed by a low surrogate.
            nxt = ord(text[i + 1]) if i + 1 < len(text) else 0
            if not (0xDC00 <= nxt <= 0xDFFF):
                return True
        elif 0xDC00 <= code <= 0xDFFF:
            # Low surrogate: valid only when preceded by a high surrogate.
            prv = ord(text[i - 1]) if i > 0 else 0
            if not (0xD800 <= prv <= 0xDBFF):
                return True
    return False

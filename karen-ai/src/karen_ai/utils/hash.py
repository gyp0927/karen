"""Fast deterministic hash to shorten long strings, mirroring utils/hash.ts.

Bit-exact port of pi-ai's shortHash (a cyrb53 variant): all arithmetic is
mod 2^32, so masking after each multiply keeps the same bit pattern as
JavaScript's Math.imul.
"""

from __future__ import annotations

_MASK = 0xFFFFFFFF
_BASE36_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def _base36(value: int) -> str:
    if value == 0:
        return "0"
    digits: list[str] = []
    while value:
        digits.append(_BASE36_DIGITS[value % 36])
        value //= 36
    return "".join(reversed(digits))


def _utf16_code_units(text: str) -> list[int]:
    """JS `charCodeAt` semantics: iterate UTF-16 code units, not code points."""
    raw = text.encode("utf-16-le", "surrogatepass")
    return [raw[i] | (raw[i + 1] << 8) for i in range(0, len(raw), 2)]


def short_hash(text: str) -> str:
    h1 = 0xDEADBEEF
    h2 = 0x41C6CE57
    for ch in _utf16_code_units(text):
        h1 = ((h1 ^ ch) * 2654435761) & _MASK
        h2 = ((h2 ^ ch) * 1597334677) & _MASK
    h1 = (((h1 ^ (h1 >> 16)) * 2246822507) & _MASK) ^ (((h2 ^ (h2 >> 13)) * 3266489909) & _MASK)
    h2 = (((h2 ^ (h2 >> 16)) * 2246822507) & _MASK) ^ (((h1 ^ (h1 >> 13)) * 3266489909) & _MASK)
    return _base36(h2) + _base36(h1)

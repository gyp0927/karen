"""OpenAI prompt-cache key clamping, mirroring api/openai-prompt-cache.ts."""

from __future__ import annotations

from typing import Optional

OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH = 64


def clamp_openai_prompt_cache_key(key: Optional[str]) -> Optional[str]:
    if key is None:
        return None
    # Slice by code point (Array.from semantics), not UTF-16 units.
    chars = list(key)
    if len(chars) <= OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH:
        return key
    return "".join(chars[:OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH])

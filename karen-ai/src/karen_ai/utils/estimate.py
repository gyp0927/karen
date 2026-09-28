"""Rough token estimation for context-size clamps.

Mirrors pi-ai's utils/estimate.ts contract: a cheap heuristic (no tokenizer
dependency) used only to keep `max_tokens` inside the context window.
"""

from __future__ import annotations

from ..types import TranscriptContext
from .text import content_text

# Conservative chars-per-token heuristic across providers.
_CHARS_PER_TOKEN = 4


def estimate_context_tokens(context: TranscriptContext) -> int:
    total_chars = 0
    for message in context.messages:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                text = getattr(block, "text", None) or getattr(block, "thinking", None)
                if isinstance(text, str):
                    total_chars += len(text)
                arguments = getattr(block, "arguments", None)
                if arguments:
                    total_chars += len(str(arguments))
    return total_chars // _CHARS_PER_TOKEN + 1

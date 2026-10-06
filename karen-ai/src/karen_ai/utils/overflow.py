"""Context-overflow detection (pi-ai's `utils/overflow.ts`).

Detects when a provider rejected — or silently truncated — a request because
the input exceeded the model's context window. Used by the app layer to trigger
"overflow" compaction and one bounded compact-and-retry attempt.

Python port notes: the JS regexes translate verbatim (`re.IGNORECASE` for the
`/i` flag); `Usage.cache_read` follows karen-ai's snake_case naming.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..types import AssistantMessage

# Regex patterns matching context-overflow error messages from different
# providers. Provider-specific examples:
#
# - Anthropic: "prompt is too long: 213462 tokens > 200000 maximum"
# - Anthropic: "413 {\"error\":{\"type\":\"request_too_large\",\"message\":\"Request exceeds the maximum size\"}}"
# - OpenAI: "Your input exceeds the context window of this model"
# - OpenAI/LiteLLM: "Requested token count exceeds the model's maximum context length of 131072 tokens"
# - OpenAI-compatible: "Input length (265330) exceeds model's maximum context length (262144)."
# - Google: "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)"
# - xAI: "This model's maximum prompt length is 131072 but the request contains 537812 tokens"
# - Groq: "Please reduce the length of the messages or completion"
# - OpenRouter: "This endpoint's maximum context length is X tokens. However, you requested about Y tokens"
# - OpenRouter/Poolside: "Input length X exceeds the maximum allowed input length of Y tokens."
# - Together AI: "The input (X tokens) is longer than the model's context length (Y tokens)."
# - llama.cpp: "the request exceeds the available context size, try increasing it"
# - LM Studio: "tokens to keep from the initial prompt is greater than the context length"
# - GitHub Copilot: "prompt token count of X exceeds the limit of Y"
# - MiniMax: "invalid params, context window exceeds limit"
# - Kimi For Coding: "Your request exceeded model token limit: X (requested: Y)"
# - DS4: "Prompt has X tokens, but the configured context size is Y tokens"
# - Cerebras: "400/413 status code (no body)"
# - Mistral: "Prompt contains X tokens ... too large for model with Y maximum context length"
# - z.ai: `{"code":"1261","message":"Prompt too long"}` or silent overflow via usage.input > contextWindow
# - Xiaomi MiMo: truncates input to fill contextWindow exactly, then returns finish_reason "length"
#   with output=0 (no room left to generate). Detected via stop_reason "length" + zero output +
#   input filling the context window.
# - DashScope/Qwen: "Range of input length should be [1, X]" (HTTP 400 invalid_parameter_error)
# - Ollama: some deployments truncate silently, others return errors like
#   "prompt too long; exceeded max context length by X tokens"
OVERFLOW_PATTERNS = [
    re.compile(r"prompt (?:is )?too long", re.IGNORECASE),  # Anthropic and z.ai token overflow
    re.compile(r"request_too_large", re.IGNORECASE),  # Anthropic request byte-size overflow (HTTP 413)
    re.compile(r"input is too long for requested model", re.IGNORECASE),  # Amazon Bedrock
    re.compile(r"exceeds the context window", re.IGNORECASE),  # OpenAI (Completions & Responses API)
    re.compile(  # OpenAI-compatible proxies (LiteLLM)
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
        re.IGNORECASE,
    ),
    re.compile(r"input token count.*exceeds the maximum", re.IGNORECASE),  # Google (Gemini)
    re.compile(r"maximum prompt length is \d+", re.IGNORECASE),  # xAI (Grok)
    re.compile(r"reduce the length of the messages", re.IGNORECASE),  # Groq
    re.compile(r"maximum context length is \d+ tokens", re.IGNORECASE),  # OpenRouter (most backends)
    re.compile(  # OpenRouter/Poolside
        r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?", re.IGNORECASE
    ),
    re.compile(  # Together AI
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
        re.IGNORECASE,
    ),
    re.compile(r"exceeds the limit of \d+", re.IGNORECASE),  # GitHub Copilot
    re.compile(r"exceeds the available context size", re.IGNORECASE),  # llama.cpp server
    re.compile(r"greater than the context length", re.IGNORECASE),  # LM Studio
    re.compile(r"context window exceeds limit", re.IGNORECASE),  # MiniMax
    re.compile(r"exceeded model token limit", re.IGNORECASE),  # Kimi For Coding
    re.compile(r"too large for model with \d+ maximum context length", re.IGNORECASE),  # Mistral
    re.compile(  # DS4 server
        r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",
        re.IGNORECASE,
    ),
    re.compile(  # z.ai non-standard finish_reason surfaced as error text
        r"model_context_window_exceeded", re.IGNORECASE
    ),
    re.compile(  # Ollama explicit overflow error
        r"prompt too long; exceeded (?:max )?context length", re.IGNORECASE
    ),
    re.compile(r"range of input length should be", re.IGNORECASE),  # DashScope / Qwen Token Plan
    re.compile(r"context[_ ]length[_ ]exceeded", re.IGNORECASE),  # Generic fallback
    re.compile(r"too many tokens", re.IGNORECASE),  # Generic fallback
    re.compile(r"token limit exceeded", re.IGNORECASE),  # Generic fallback
]

CEREBRAS_BODYLESS_OVERFLOW_PATTERN = re.compile(r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)", re.IGNORECASE)

# Patterns that indicate non-overflow errors (rate limiting, server errors).
# Matching messages are excluded from overflow detection even if they also
# match an OVERFLOW_PATTERN. Example: Bedrock formats throttling errors as
# "ThrottlingException: Too many tokens, please wait before trying again."
# which would match the /too many tokens/i overflow pattern without this
# exclusion.
NON_OVERFLOW_PATTERNS = [
    # AWS Bedrock non-overflow errors (human-readable prefixes from formatBedrockError)
    re.compile(r"^(Throttling error|Service unavailable):", re.IGNORECASE),
    re.compile(r"rate limit", re.IGNORECASE),  # Generic rate limiting
    re.compile(r"too many requests", re.IGNORECASE),  # Generic HTTP 429 style
]


def is_context_overflow(message: AssistantMessage, context_window: Optional[int] = None) -> bool:
    """Check if an assistant message represents a context overflow error.

    Three cases:
    1. Error-based overflow: most providers return stop_reason "error" with a
       specific error message pattern.
    2. Silent overflow: some providers accept overflow requests and return
       successfully; detected via usage.input + cache_read > context_window.
    3. Length-stop overflow: Xiaomi MiMo returns "length" with zero output
       when the input fills the context window.

    Detection is reliable for providers that return a detectable error
    (Anthropic, OpenAI, Google, xAI, Groq, Cerebras, Mistral, OpenRouter,
    Together AI, llama.cpp, LM Studio, Kimi For Coding, DS4, DashScope/Qwen,
    z.ai when it reports). It is unreliable for z.ai silent overflow (pass
    `context_window` to catch it via usage), Xiaomi MiMo (pass
    `context_window` for the filled-context + zero-output signal), and Ollama
    silent truncation (undetectable without the expected token count).

    Custom providers may need their own check before calling this function.
    """
    # Case 1: error message patterns
    if message.stop_reason == "error" and message.error_message:
        # Skip known non-overflow patterns (throttling / rate-limit)
        if not any(p.search(message.error_message) for p in NON_OVERFLOW_PATTERNS):
            if any(p.search(message.error_message) for p in OVERFLOW_PATTERNS):
                return True
            if message.provider == "cerebras" and CEREBRAS_BODYLESS_OVERFLOW_PATTERN.search(
                message.error_message
            ):
                return True

    # Case 2: silent overflow (z.ai style) - successful but usage exceeds context
    if context_window and message.stop_reason == "stop":
        if message.usage.input + message.usage.cache_read > context_window:
            return True

    # Case 3: length-stop overflow (Xiaomi MiMo style) - server truncates oversized
    # input to fit the context window, leaving no room for output: stop_reason
    # "length" with output=0 and input+cache_read filling the context window.
    if context_window and message.stop_reason == "length" and message.usage.output == 0:
        if message.usage.input + message.usage.cache_read >= context_window * 0.99:
            return True

    return False


def is_recoverable_length(message: AssistantMessage, desired_max_output: int) -> bool:
    """Check whether a length stop ended below the intended output limit.

    Such responses may be caused by context pressure or provider-side
    truncation, so callers can make one bounded compact-and-retry attempt.
    `desired_max_output` must be the original limit before any context-based
    clamping.
    """
    return (
        message.stop_reason == "length"
        and desired_max_output > 0
        and message.usage.output < desired_max_output
    )


def get_overflow_patterns() -> List["re.Pattern[str]"]:
    """Get the overflow patterns (a copy), for testing purposes."""
    return list(OVERFLOW_PATTERNS)

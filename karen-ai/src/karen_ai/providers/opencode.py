"""OpenCode Zen provider (multi-API gateway), mirroring opencode.ts."""

from __future__ import annotations

from ..api import (
    anthropic_messages_api,
    google_generative_ai_api,
    openai_completions_api,
    openai_responses_api,
)
from ._catalog import catalog_provider
from .opencode_headers import with_opencode_session_header


def opencode_provider():
    return catalog_provider(
        id="opencode",
        name="OpenCode Zen",
        key_name="OpenCode API key",
        env_vars=["OPENCODE_API_KEY"],
        api={
            "anthropic-messages": with_opencode_session_header(anthropic_messages_api()),
            "google-generative-ai": with_opencode_session_header(google_generative_ai_api()),
            "openai-completions": with_opencode_session_header(openai_completions_api()),
            "openai-responses": with_opencode_session_header(openai_responses_api()),
        },
    )

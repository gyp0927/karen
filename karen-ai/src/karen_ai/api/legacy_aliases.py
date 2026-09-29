"""Deprecated per-API stream aliases, mirroring pi-ai's `legacy-api-aliases.ts`.

These are the pre-`createModels()` entry points, kept so ported call sites keep
working. New code should use `karen_ai.api.<module>.stream` or a provider's
`ProviderStreams`.
"""

from __future__ import annotations

from . import (
    anthropic_messages,
    azure_openai_responses,
    google_generative_ai,
    google_vertex,
    mistral_conversations,
    openai_codex_responses,
    openai_completions,
    openai_responses,
)

#: @deprecated Use `karen_ai.api.anthropic_messages.stream` or `anthropic_messages_api().stream`.
stream_anthropic = anthropic_messages.stream
#: @deprecated Use `karen_ai.api.anthropic_messages.stream_simple`.
stream_simple_anthropic = anthropic_messages.stream_simple

#: @deprecated Use `karen_ai.api.azure_openai_responses.stream`.
stream_azure_openai_responses = azure_openai_responses.stream
#: @deprecated Use `karen_ai.api.azure_openai_responses.stream_simple`.
stream_simple_azure_openai_responses = azure_openai_responses.stream_simple

#: @deprecated Use `karen_ai.api.google_generative_ai.stream`.
stream_google = google_generative_ai.stream
#: @deprecated Use `karen_ai.api.google_generative_ai.stream_simple`.
stream_simple_google = google_generative_ai.stream_simple

#: @deprecated Use `karen_ai.api.google_vertex.stream`.
stream_google_vertex = google_vertex.stream
#: @deprecated Use `karen_ai.api.google_vertex.stream_simple`.
stream_simple_google_vertex = google_vertex.stream_simple

#: @deprecated Use `karen_ai.api.mistral_conversations.stream`.
stream_mistral = mistral_conversations.stream
#: @deprecated Use `karen_ai.api.mistral_conversations.stream_simple`.
stream_simple_mistral = mistral_conversations.stream_simple

#: @deprecated Use `karen_ai.api.openai_codex_responses.stream`.
stream_openai_codex_responses = openai_codex_responses.stream
#: @deprecated Use `karen_ai.api.openai_codex_responses.stream_simple`.
stream_simple_openai_codex_responses = openai_codex_responses.stream_simple

#: @deprecated Use `karen_ai.api.openai_completions.stream`.
stream_openai_completions = openai_completions.stream
#: @deprecated Use `karen_ai.api.openai_completions.stream_simple`.
stream_simple_openai_completions = openai_completions.stream_simple

#: @deprecated Use `karen_ai.api.openai_responses.stream`.
stream_openai_responses = openai_responses.stream
#: @deprecated Use `karen_ai.api.openai_responses.stream_simple`.
stream_simple_openai_responses = openai_responses.stream_simple

__all__ = [
    "stream_anthropic",
    "stream_azure_openai_responses",
    "stream_google",
    "stream_google_vertex",
    "stream_mistral",
    "stream_openai_codex_responses",
    "stream_openai_completions",
    "stream_openai_responses",
    "stream_simple_anthropic",
    "stream_simple_azure_openai_responses",
    "stream_simple_google",
    "stream_simple_google_vertex",
    "stream_simple_mistral",
    "stream_simple_openai_codex_responses",
    "stream_simple_openai_completions",
    "stream_simple_openai_responses",
]

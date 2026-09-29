"""API adapters: each module exposes the uniform `stream` / `stream_simple` contract."""

from ..lazy import ProviderStreams
from . import (
    anthropic_messages,
    azure_openai_responses,
    bedrock_converse_stream,
    google_generative_ai,
    google_vertex,
    mistral_conversations,
    openai_codex_responses,
    openai_completions,
    openai_responses,
    pi_messages,
)


def anthropic_messages_api() -> ProviderStreams:
    return ProviderStreams(stream=anthropic_messages.stream, stream_simple=anthropic_messages.stream_simple)


def openai_completions_api() -> ProviderStreams:
    return ProviderStreams(stream=openai_completions.stream, stream_simple=openai_completions.stream_simple)


def openai_responses_api() -> ProviderStreams:
    return ProviderStreams(stream=openai_responses.stream, stream_simple=openai_responses.stream_simple)


def google_generative_ai_api() -> ProviderStreams:
    return ProviderStreams(stream=google_generative_ai.stream, stream_simple=google_generative_ai.stream_simple)


def google_vertex_api() -> ProviderStreams:
    return ProviderStreams(stream=google_vertex.stream, stream_simple=google_vertex.stream_simple)


def mistral_conversations_api() -> ProviderStreams:
    return ProviderStreams(stream=mistral_conversations.stream, stream_simple=mistral_conversations.stream_simple)


def bedrock_converse_stream_api() -> ProviderStreams:
    return ProviderStreams(stream=bedrock_converse_stream.stream, stream_simple=bedrock_converse_stream.stream_simple)


def azure_openai_responses_api() -> ProviderStreams:
    return ProviderStreams(stream=azure_openai_responses.stream, stream_simple=azure_openai_responses.stream_simple)


def pi_messages_api() -> ProviderStreams:
    return ProviderStreams(stream=pi_messages.stream, stream_simple=pi_messages.stream_simple)


def openai_codex_responses_api() -> ProviderStreams:
    return ProviderStreams(
        stream=openai_codex_responses.stream, stream_simple=openai_codex_responses.stream_simple
    )


__all__ = [
    "ProviderStreams",
    "anthropic_messages",
    "azure_openai_responses",
    "bedrock_converse_stream",
    "google_generative_ai",
    "google_vertex",
    "mistral_conversations",
    "openai_codex_responses",
    "openai_completions",
    "openai_responses",
    "pi_messages",
    "anthropic_messages_api",
    "azure_openai_responses_api",
    "bedrock_converse_stream_api",
    "google_generative_ai_api",
    "google_vertex_api",
    "mistral_conversations_api",
    "openai_codex_responses_api",
    "openai_completions_api",
    "openai_responses_api",
    "pi_messages_api",
]

"""API adapters: each module exposes the uniform `stream` / `stream_simple` contract."""

from ..lazy import ProviderStreams
from . import anthropic_messages, openai_completions


def anthropic_messages_api() -> ProviderStreams:
    return ProviderStreams(stream=anthropic_messages.stream, stream_simple=anthropic_messages.stream_simple)


def openai_completions_api() -> ProviderStreams:
    return ProviderStreams(stream=openai_completions.stream, stream_simple=openai_completions.stream_simple)


__all__ = [
    "ProviderStreams",
    "anthropic_messages",
    "openai_completions",
    "anthropic_messages_api",
    "openai_completions_api",
]

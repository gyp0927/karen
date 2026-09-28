"""Built-in providers."""

from .anthropic import anthropic_provider
from .compatible import openai_compatible_provider
from .deepseek import deepseek_provider
from .faux import (
    faux_assistant_message,
    faux_model,
    faux_text,
    faux_thinking,
    faux_tool_call,
    register_faux_provider,
)
from .google import google_provider
from .google_vertex import google_vertex_provider
from .mistral import mistral_provider
from .openai import openai_provider
from .openrouter import openrouter_provider


def builtin_providers():
    """Every built-in provider (the ported subset of pi-ai's builtinProviders())."""
    return [
        anthropic_provider(),
        deepseek_provider(),
        google_provider(),
        google_vertex_provider(),
        mistral_provider(),
        openai_provider(),
        openrouter_provider(),
    ]


def builtin_models(options=None):
    """A `Models` collection with every built-in provider registered."""
    from ..models import create_models

    models = create_models(options)
    for provider in builtin_providers():
        models.set_provider(provider)
    return models


__all__ = [
    "anthropic_provider",
    "google_provider",
    "google_vertex_provider",
    "mistral_provider",
    "openai_provider",
    "deepseek_provider",
    "openrouter_provider",
    "openai_compatible_provider",
    "builtin_providers",
    "builtin_models",
    "faux_assistant_message",
    "faux_model",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
    "register_faux_provider",
]

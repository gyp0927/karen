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
from .openai import openai_provider
from .openrouter import openrouter_provider

__all__ = [
    "anthropic_provider",
    "openai_provider",
    "deepseek_provider",
    "openrouter_provider",
    "openai_compatible_provider",
    "faux_assistant_message",
    "faux_model",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
    "register_faux_provider",
]

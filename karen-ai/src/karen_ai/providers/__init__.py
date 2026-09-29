"""Built-in providers."""

from .amazon_bedrock import amazon_bedrock_provider
from .ant_ling import ant_ling_provider
from .anthropic import anthropic_provider
from .azure_openai_responses import azure_openai_responses_provider
from .baseten import baseten_provider
from .cerebras import cerebras_provider
from .cloudflare_ai_gateway import cloudflare_ai_gateway_provider
from .cloudflare_workers_ai import cloudflare_workers_ai_provider
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
from .fireworks import fireworks_provider
from .github_copilot import github_copilot_provider
from .google import google_provider
from .google_vertex import google_vertex_provider
from .groq import groq_provider
from .huggingface import huggingface_provider
from .kimi_coding import kimi_coding_provider
from .meta import meta_provider
from .minimax import minimax_provider
from .minimax_cn import minimax_cn_provider
from .mistral import mistral_provider
from .moonshotai import moonshotai_provider
from .moonshotai_cn import moonshotai_cn_provider
from .nvidia import nvidia_provider
from .openai import openai_provider
from .openai_codex import openai_codex_provider
from .opencode import opencode_provider
from .opencode_go import opencode_go_provider
from .openrouter import openrouter_provider
from .qwen_token_plan import qwen_token_plan_provider
from .qwen_token_plan_cn import qwen_token_plan_cn_provider
from .qwen_token_plan_individual import qwen_token_plan_individual_provider
from .radius import radius_provider
from .together import together_provider
from .typesafe import typesafe_provider
from .vercel_ai_gateway import vercel_ai_gateway_provider
from .xai import xai_provider
from .xiaomi import xiaomi_provider
from .xiaomi_token_plan_ams import xiaomi_token_plan_ams_provider
from .xiaomi_token_plan_cn import xiaomi_token_plan_cn_provider
from .xiaomi_token_plan_sgp import xiaomi_token_plan_sgp_provider
from .zai import zai_provider
from .zai_coding_cn import zai_coding_cn_provider


def builtin_providers():
    """Every built-in provider, mirroring pi-ai's builtinProviders()."""
    return [
        amazon_bedrock_provider(),
        ant_ling_provider(),
        anthropic_provider(),
        azure_openai_responses_provider(),
        baseten_provider(),
        cerebras_provider(),
        cloudflare_ai_gateway_provider(),
        cloudflare_workers_ai_provider(),
        deepseek_provider(),
        fireworks_provider(),
        github_copilot_provider(),
        google_provider(),
        google_vertex_provider(),
        groq_provider(),
        huggingface_provider(),
        kimi_coding_provider(),
        meta_provider(),
        minimax_provider(),
        minimax_cn_provider(),
        mistral_provider(),
        moonshotai_provider(),
        moonshotai_cn_provider(),
        nvidia_provider(),
        openai_provider(),
        openai_codex_provider(),
        opencode_provider(),
        opencode_go_provider(),
        openrouter_provider(),
        qwen_token_plan_provider(),
        qwen_token_plan_cn_provider(),
        qwen_token_plan_individual_provider(),
        radius_provider(),
        together_provider(),
        typesafe_provider(),
        vercel_ai_gateway_provider(),
        xai_provider(),
        xiaomi_provider(),
        xiaomi_token_plan_ams_provider(),
        xiaomi_token_plan_cn_provider(),
        xiaomi_token_plan_sgp_provider(),
        zai_provider(),
        zai_coding_cn_provider(),
    ]


def builtin_models(options=None):
    """A `Models` collection with every built-in provider registered."""
    from ..models import create_models

    models = create_models(options)
    for provider in builtin_providers():
        models.set_provider(provider)
    return models


__all__ = [
    "amazon_bedrock_provider",
    "ant_ling_provider",
    "anthropic_provider",
    "azure_openai_responses_provider",
    "baseten_provider",
    "cerebras_provider",
    "cloudflare_ai_gateway_provider",
    "cloudflare_workers_ai_provider",
    "deepseek_provider",
    "fireworks_provider",
    "github_copilot_provider",
    "google_provider",
    "google_vertex_provider",
    "groq_provider",
    "huggingface_provider",
    "kimi_coding_provider",
    "meta_provider",
    "minimax_provider",
    "minimax_cn_provider",
    "mistral_provider",
    "moonshotai_provider",
    "moonshotai_cn_provider",
    "nvidia_provider",
    "openai_provider",
    "openai_codex_provider",
    "opencode_provider",
    "opencode_go_provider",
    "openrouter_provider",
    "qwen_token_plan_provider",
    "qwen_token_plan_cn_provider",
    "qwen_token_plan_individual_provider",
    "radius_provider",
    "together_provider",
    "typesafe_provider",
    "vercel_ai_gateway_provider",
    "xai_provider",
    "xiaomi_provider",
    "xiaomi_token_plan_ams_provider",
    "xiaomi_token_plan_cn_provider",
    "xiaomi_token_plan_sgp_provider",
    "zai_provider",
    "zai_coding_cn_provider",
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

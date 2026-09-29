"""Every constant table the generator needs.

Port of the constant block of pi-ai's `scripts/generate-models.ts` (the
`COPILOT_STATIC_HEADERS` .. `ANTHROPIC_PROMPT_CACHE` range), plus the static
model lists that `generateModels()` appends verbatim.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Base URLs and static headers
# ---------------------------------------------------------------------------

BEDROCK_BASE_URL_EU = "https://bedrock-runtime.eu-central-1.amazonaws.com"
BEDROCK_BASE_URL_US = "https://bedrock-runtime.us-east-1.amazonaws.com"

TOGETHER_BASE_URL = "https://api.together.ai/v1"
AI_GATEWAY_MODELS_URL = "https://ai-gateway.vercel.sh/v1"
AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh"
VERTEX_BASE_URL = "https://{location}-aiplatform.googleapis.com"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_HEADERS = {"NVCF-POLL-SECONDS": "3600"}

CLOUDFLARE_WORKERS_AI_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"
CLOUDFLARE_WORKERS_AI_REST_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai"
CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai"
)
CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/anthropic"
)
CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat"
)

COPILOT_STATIC_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}

# ---------------------------------------------------------------------------
# Costs and numeric helpers
# ---------------------------------------------------------------------------


def round_cost(value: float) -> float:
    """Mimic JavaScript's `Number(value.toFixed(6))`."""
    return float(f"{value:.6f}")


OPENAI_LONG_CONTEXT_INPUT_THRESHOLD = 272000

#: Keep current OpenAI prices authoritative until models.dev and passthrough
#: catalogs catch up. https://developers.openai.com/api/docs/pricing
OPENAI_STANDARD_COSTS: Dict[str, Dict[str, float]] = {
    "gpt-5.6-luna": {"input": 0.2, "output": 1.2, "cacheRead": 0.02, "cacheWrite": 0.25},
    "gpt-5.6-sol": {"input": 4, "output": 20, "cacheRead": 0.4, "cacheWrite": 5},
    "gpt-5.6-terra": {"input": 2, "output": 12, "cacheRead": 0.2, "cacheWrite": 2.5},
    "gpt-6-astra": {"input": 10, "output": 50, "cacheRead": 1, "cacheWrite": 12.5},
    "gpt-6-luna": {"input": 0.1, "output": 0.5, "cacheRead": 0.01, "cacheWrite": 0.125},
    "gpt-6-sol": {"input": 2, "output": 10, "cacheRead": 0.2, "cacheWrite": 2.5},
}


def with_open_ai_long_context_pricing(cost: Dict[str, float]) -> Dict[str, Any]:
    """The standard rates plus the long-context pricing tier."""
    return {
        **cost,
        "tiers": [
            {
                "inputTokensAbove": OPENAI_LONG_CONTEXT_INPUT_THRESHOLD,
                "input": round_cost(cost["input"] * 2),
                "output": round_cost(cost["output"] * 1.5),
                "cacheRead": round_cost(cost["cacheRead"] * 2),
                "cacheWrite": round_cost(cost["cacheWrite"] * 2),
            }
        ],
    }


# Keep the generated default no less restrictive than coding-agent's historical
# image preprocessing. Provider limits can narrow this profile, but unknown
# providers retain the cache-safe 2000px / 4.5 MiB behavior.
DEFAULT_IMAGE_RESIZE = {
    "maxWidth": 2000,
    "maxHeight": 2000,
    "maxBytes": 4.5 * 1024 * 1024,
    "jpegQuality": 80,
}

ANTHROPIC_PROMPT_CACHE = {"short": 300, "long": 3600}

# ---------------------------------------------------------------------------
# Compat tables
# ---------------------------------------------------------------------------

TOGETHER_BASE_COMPAT: Dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
    "supportsStrictMode": False,
    "supportsLongCacheRetention": False,
}
TOGETHER_TOGGLE_REASONING_COMPAT = {**TOGETHER_BASE_COMPAT, "thinkingFormat": "together"}
TOGETHER_REASONING_EFFORT_COMPAT = {
    **TOGETHER_BASE_COMPAT,
    "supportsReasoningEffort": True,
    "thinkingFormat": "openai",
}
TOGETHER_TOGGLE_REASONING_EFFORT_COMPAT = {
    **TOGETHER_TOGGLE_REASONING_COMPAT,
    "supportsReasoningEffort": True,
}

TOGETHER_REASONING_ONLY_MODELS = {"deepseek-ai/DeepSeek-R1", "MiniMaxAI/MiniMax-M2.7"}
TOGETHER_REASONING_EFFORT_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
TOGETHER_TOGGLE_REASONING_EFFORT_MODELS = {"deepseek-ai/DeepSeek-V4-Pro"}

NVIDIA_OPENAI_COMPAT: Dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
    "supportsStrictMode": False,
    "supportsLongCacheRetention": False,
}

XAI_RESPONSES_COMPAT: Dict[str, Any] = {"supportsLongCacheRetention": False}

OPENAI_COMPLETIONS_DEFAULT_COMPAT: Dict[str, Any] = {
    "supportsStore": True,
    "supportsDeveloperRole": True,
    "supportsReasoningEffort": True,
    "supportsUsageInStreaming": True,
    "supportsFinishReason": True,
    "maxTokensField": "max_completion_tokens",
    "requiresToolResultName": False,
    "requiresAssistantAfterToolResult": False,
    "requiresThinkingAsText": False,
    "requiresReasoningContentOnAssistantMessages": False,
    "thinkingFormat": "openai",
    "openRouterRouting": {},
    "vercelGatewayRouting": {},
    "chatTemplateKwargs": {},
    "chatTemplateArgs": {},
    "zaiToolStream": False,
    "supportsStrictMode": False,
    "supportsOpenAIGrammarTools": False,
    "supportsMidConvoSystemMessages": False,
    "supportsMidConvoToolAdditions": False,
    "sendSessionAffinityHeaders": False,
    "supportsLongCacheRetention": True,
}

DEEPSEEK_COMPAT: Dict[str, Any] = {
    "requiresReasoningContentOnAssistantMessages": True,
    "thinkingFormat": "deepseek",
}

ANT_LING_COMPAT: Dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
    "supportsLongCacheRetention": False,
}

FIREWORKS_ANTHROPIC_COMPAT: Dict[str, Any] = {
    "allowEmptySignature": True,
    "sendSessionAffinityHeaders": True,
    "supportsEagerToolInputStreaming": False,
    "supportsCacheControlOnTools": False,
    "supportsLongCacheRetention": False,
}
FIREWORKS_OPENAI_COMPAT: Dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "sendSessionAffinityHeaders": True,
    "supportsLongCacheRetention": False,
}
FIREWORKS_KIMI_K3_COMPAT: Dict[str, Any] = {
    **FIREWORKS_OPENAI_COMPAT,
    "requiresReasoningContentOnAssistantMessages": True,
    "thinkingFormat": "openai",
}

BASETEN_BASE_COMPAT: Dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "supportsUsageInStreaming": True,
    "maxTokensField": "max_tokens",
    "supportsStrictMode": True,
    # Baseten automatic prompt caching needs session affinity so related
    # requests land on the same replica. See:
    # https://docs.baseten.co/inference/model-apis/pricing-and-limits
    "sendSessionAffinityHeaders": True,
    "supportsLongCacheRetention": False,
}
BASETEN_REASONING_EFFORT_COMPAT = {
    **BASETEN_BASE_COMPAT,
    "supportsReasoningEffort": True,
    "thinkingFormat": "openai",
}
BASETEN_TOGGLE_REASONING_COMPAT = {
    **BASETEN_BASE_COMPAT,
    "thinkingFormat": "baseten",
    "chatTemplateArgs": {"enable_thinking": {"$var": "thinking.enabled"}},
}
BASETEN_TOGGLE_REASONING_EFFORT_COMPAT = {
    **BASETEN_REASONING_EFFORT_COMPAT,
    "thinkingFormat": "baseten",
    "chatTemplateArgs": {"enable_thinking": {"$var": "thinking.enabled"}},
}

MOONSHOT_COMPAT: Dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
    "supportsStrictMode": False,
    "thinkingFormat": "deepseek",
}

XIAOMI_COMPAT: Dict[str, Any] = {
    "requiresReasoningContentOnAssistantMessages": True,
    "thinkingFormat": "deepseek",
}

QWEN_TOKEN_PLAN_COMPAT: Dict[str, Any] = {
    "thinkingFormat": "qwen",
    "supportsDeveloperRole": False,
    "supportsStore": False,
    "supportsReasoningEffort": True,
}

# ---------------------------------------------------------------------------
# Thinking-level maps
# ---------------------------------------------------------------------------

TOGETHER_FIXED_REASONING_LEVEL_MAP = {"off": None, "minimal": None, "low": None, "medium": None}
TOGETHER_REASONING_EFFORT_LEVEL_MAP = {"off": None, "minimal": None}
TOGETHER_DEEPSEEK_V4_THINKING_LEVEL_MAP = {
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
}
TOGETHER_TOGGLE_REASONING_LEVEL_MAP = {"minimal": None, "low": None, "medium": None}

DEEPSEEK_V4_THINKING_LEVEL_MAP = {
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "max": "max",
}
DEEPSEEK_V4_FLASH_THINKING_LEVEL_MAP = {**DEEPSEEK_V4_THINKING_LEVEL_MAP, "low": "low"}

OPENCODE_GO_GLM52_THINKING_LEVEL_MAP = {
    "off": None,
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "max": "max",
}

ANT_LING_RING_THINKING_LEVEL_MAP = {
    "off": None,
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": "xhigh",
}

QWEN_TOKEN_PLAN_FALLBACK_THINKING_LEVEL_MAP = {
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": "max",
}

BASETEN_TOGGLE_THINKING_LEVEL_MAP = {
    "off": "off",
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": None,
}
BASETEN_GLM52_THINKING_LEVEL_MAP = {
    "off": "none",
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": "max",
}

# Checked manually against the authenticated GitHub Copilot /models endpoint on 2026-06-15.
# Keep this to narrow corrections over models.dev metadata instead of snapshotting Copilot's catalog.
GITHUB_COPILOT_THINKING_LEVEL_OVERRIDES: Dict[str, Dict[str, Optional[str]]] = {
    "claude-opus-4.7": {"minimal": "low"},
    "claude-opus-4.8": {"minimal": "low"},
    "claude-opus-5": {"minimal": "low"},
    "claude-sonnet-4.6": {"minimal": "low", "max": "max"},
}

ANTHROPIC_ALLOWED_FALLBACK_MODELS: Dict[str, List[str]] = {
    "claude-fable-5": ["claude-opus-4-8", "claude-opus-5"],
    "claude-opus-5": ["claude-opus-4-8"],
}

# ---------------------------------------------------------------------------
# Model-id sets
# ---------------------------------------------------------------------------

NVIDIA_NIM_UNSUPPORTED_MODELS = {
    "abacusai/dracarys-llama-3.1-70b-instruct",
    "bytedance/seed-oss-36b-instruct",
    "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v4-pro",
    "google/gemma-2-2b-it",
    "google/gemma-3n-e2b-it",
    "google/gemma-3n-e4b-it",
    "google/gemma-4-31b-it",
    "meta/llama-3.2-1b-instruct",
    "meta/llama-4-maverick-17b-128e-instruct",
    "microsoft/phi-4-mini-instruct",
    "minimaxai/minimax-m2.7",
    "mistralai/mistral-nemotron",
    "nvidia/nemotron-mini-4b-instruct",
    "qwen/qwen3-next-80b-a3b-instruct",
    "qwen/qwen3.5-397b-a17b",
    "sarvamai/sarvam-m",
    "upstage/solar-10.7b-instruct",
}

ZAI_TOOL_STREAM_UNSUPPORTED_MODELS = {"glm-4.5", "glm-4.5-air", "glm-4.5-flash", "glm-4.5v"}

EAGER_TOOL_INPUT_STREAMING_UNSUPPORTED_ANTHROPIC_MODELS = {
    "github-copilot:claude-haiku-4.5",
    "github-copilot:claude-sonnet-4",
    "github-copilot:claude-sonnet-4.5",
}

# Verified against Fireworks Messages raw_output on 2026-09-10 (#9323).
# Fall back to verified support when models.dev omits effort metadata; this is
# not an allowlist. Any Fireworks Messages model advertising effort uses adaptive thinking.
FIREWORKS_ADAPTIVE_THINKING_FALLBACK_MODELS = {
    "accounts/fireworks/models/deepseek-v4-flash-0731",
    "accounts/fireworks/models/deepseek-v4-flash-vision-exp",
    "accounts/fireworks/models/deepseek-v4-pro-0813",
    "accounts/fireworks/models/qwen3p8-max",
    "accounts/fireworks/models/qwen3p8-2p4t-a95b",
}

QWEN_TOKEN_PLAN_REASONING_EFFORT_FALLBACK_MODEL_IDS = {"glm-5", "glm-5.1"}
# Retired preview id — models.dev may still list it after GA ships.
QWEN_TOKEN_PLAN_EXCLUDED_MODEL_IDS = {"qwen3.8-max-preview"}
QWEN_TOKEN_PLAN_PROVIDER_IDS = {"qwen-token-plan", "qwen-token-plan-cn", "qwen-token-plan-individual"}
# QwenCloud Token Plan Individual text-model allowlist, verified 2026-09-03.
# Retired models remain excluded above even if the public catalog lags.
# https://docs.qwencloud.com/token-plan/personal/token-plan-personal-overview
QWEN_TOKEN_PLAN_INDIVIDUAL_MODEL_IDS = {
    "deepseek-v4-flash-0731",
    "deepseek-v4-pro",
    "deepseek-v4-pro-0813",
    "glm-5.2",
    "qwen3.6-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.8-flash",
    "qwen3.8-max",
}

KIMI_K3_MAX_TOKENS = 131072
KIMI_K3_COST = {"input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 0}
# Kimi Coding is subscription-backed, so models.dev reports zero cost. Use the
# equivalent Moonshot API rates to estimate the value of subscription usage.
KIMI_CODING_IMPLIED_COSTS: Dict[str, Dict[str, float]] = {
    "k3": KIMI_K3_COST,
    "kimi-for-coding": {"input": 0.95, "output": 4, "cacheRead": 0.19, "cacheWrite": 0},
    "kimi-for-coding-highspeed": {"input": 1.9, "output": 8, "cacheRead": 0.38, "cacheWrite": 0},
    "kimi-k2-thinking": {"input": 0.6, "output": 2.5, "cacheRead": 0.15, "cacheWrite": 0},
}
OPENROUTER_KIMI_K3_MODEL_IDS = {"moonshotai/kimi-k3", "~moonshotai/kimi-latest"}

BEDROCK_INFERENCE_PROFILE_ONLY_MODEL_IDS = {"anthropic.claude-opus-5"}
MODELS_DEV_OPENAI_UNSUPPORTED_MODEL_IDS = {"gpt-5.6"}

OPENAI_TOOL_SEARCH_MODEL_IDS = {
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-pro",
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-6-astra",
    "gpt-6-sol",
    "gpt-6-luna",
}
OPENAI_ADDITIONAL_TOOLS_MODEL_IDS = OPENAI_TOOL_SEARCH_MODEL_IDS
OPENAI_MID_CONVO_SYSTEM_MESSAGE_MODEL_IDS = OPENAI_TOOL_SEARCH_MODEL_IDS
OPENAI_CODEX_ADDITIONAL_TOOLS_MODEL_IDS = {
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-6-astra",
    "gpt-6-sol",
    "gpt-6-luna",
}
OPENAI_SHORT_CONTEXT_CAPPED_MODEL_IDS = {
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-6-astra",
    "gpt-6-sol",
    "gpt-6-luna",
}
OPENAI_LONG_CONTEXT_PRICING_MODEL_IDS = {
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-6-astra",
    "gpt-6-sol",
    "gpt-6-luna",
}
OPENAI_RESPONSES_NONE_REASONING_MODELS = {
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-6-sol",
    "gpt-6-luna",
}
XAI_BUILTIN_EXCLUDED_MODEL_IDS = {
    "grok-3",
    "grok-3-fast",
    "grok-4.20-0309-non-reasoning",
    "grok-4.20-0309-reasoning",
    "grok-build-0.1",
    "grok-code-fast-1",
}
OPENCODE_LONG_CACHE_RETENTION_UNSUPPORTED_MODELS = {
    "opencode:deepseek-v4-flash",
    "opencode:deepseek-v4-pro",
    "opencode:kimi-k2.5",
    "opencode:kimi-k2.6",
    "opencode:minimax-m2.7",
    "opencode-go:kimi-k2.6",
}

# GitHub's "Models with extended capabilities" table lists these Copilot models as supporting
# the extended 1 million token context window.
GITHUB_COPILOT_EXTENDED_CONTEXT_MODELS = {
    "claude-fable-5",
    "claude-opus-4.6",
    "claude-opus-4.7",
    "claude-opus-4.8",
    "claude-opus-5",
    "claude-opus-5.5",
    "claude-sonnet-4.6",
    "claude-sonnet-5",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.5",
    "gpt-6-astra",
    "gpt-6-luna",
    "gpt-6-sol",
}

VERIFIED_ANTHROPIC_MID_CONVO_EFFORT_PROVIDERS = {"anthropic", "openrouter"}
# OpenRouter rejects `configuration_update` system messages on Opus 5 ("Mid-conversation
# reasoning effort (configuration_update) is not supported on anthropic/claude-opus-5-20260723")
# while accepting them on Fable 5.1, so gate that model there.
MID_CONVO_EFFORT_UNSUPPORTED_ANTHROPIC_MODELS = {"openrouter:anthropic/claude-opus-5"}

# Responses endpoints verified (OpenAI, ChatGPT Codex backend, GitHub Copilot,
# opencode zen) or documented (Azure OpenAI, Cloudflare AI Gateway) to pass
# OpenAI custom grammar tools through. OpenAI rejects `type: "custom"` tools
# for pre-GPT-5 models (gpt-4.x, gpt-4o, o-series).
OPENAI_GRAMMAR_TOOL_PROVIDERS = {
    "openai",
    "openai-codex",
    "azure-openai-responses",
    "github-copilot",
    "opencode",
    "cloudflare-ai-gateway",
}
OPENAI_GRAMMAR_TOOL_APIS = {"openai-responses", "azure-openai-responses", "openai-codex-responses"}

OPENAI_RESPONSES_PROXY_PROVIDERS = {"opencode", "opencode-go", "github-copilot"}

MINIMAX_DIRECT_SUPPORTED_IDS = {"MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M3"}

KIMI_ALIASES = {"k2p5", "k2p6", "k2p7"}

AZURE_CONTEXT_WINDOW_OVERRIDES: Dict[str, int] = {
    "gpt-5.4": 1050000,
    "gpt-5.5": 1050000,
    "gpt-5.6-luna": 1050000,
    "gpt-5.6-sol": 1050000,
    "gpt-5.6-terra": 1050000,
}

# ---------------------------------------------------------------------------
# Static catalog entries
# ---------------------------------------------------------------------------

# Workers AI has no unauthenticated catalog and models.dev does not list its
# System One models yet. Cloudflare publishes pricing only in the dashboard.
# https://developers.cloudflare.com/ai/models/typesafe/jev/
CLOUDFLARE_WORKERS_AI_CLASSIFIER_MODELS: List[Dict[str, Any]] = [
    {
        "type": "classifier",
        "id": "typesafe/jev",
        "name": "Jev",
        "api": "cloudflare-workers-ai-system-one",
        "provider": "cloudflare-workers-ai",
        "baseUrl": CLOUDFLARE_WORKERS_AI_REST_BASE_URL,
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 32000,
    }
]

#: Add Claude Opus 5.5 until models.dev includes it.
#: https://platform.claude.com/docs/en/models/opus-5-5/overview
MISSING_ANTHROPIC_MODEL: Dict[str, Any] = {
    "id": "claude-opus-5-5",
    "name": "Claude Opus 5.5",
    "api": "anthropic-messages",
    "provider": "anthropic",
    "baseUrl": "https://api.anthropic.com",
    "reasoning": True,
    "thinkingLevelMap": {
        "off": None,
        "minimal": None,
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "max",
    },
    "input": ["text", "image"],
    "cost": {"input": 4, "output": 20, "cacheRead": 0.2, "cacheWrite": 5},
    "contextWindow": 1000000,
    "maxTokens": 128000,
}

#: The authenticated Copilot catalog advertised these models on 2026-09-22,
#: but models.dev did not include them yet.
MISSING_COPILOT_MODELS: List[Dict[str, Any]] = [
    {
        "id": "claude-opus-5.5",
        "name": "Claude Opus 5.5",
        "api": "anthropic-messages",
        "provider": "github-copilot",
        "baseUrl": "https://api.individual.githubcopilot.com",
        "reasoning": True,
        "thinkingLevelMap": {
            "off": None,
            "minimal": None,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
            "max": "max",
        },
        "input": ["text", "image"],
        "cost": {"input": 4, "output": 20, "cacheRead": 0.2, "cacheWrite": 5},
        "contextWindow": 1000000,
        "maxTokens": 128000,
        "headers": dict(COPILOT_STATIC_HEADERS),
    },
    *[
        {
            "id": model_id,
            "name": "GPT-6 Sol" if model_id == "gpt-6-sol" else "GPT-6 Luna",
            "api": "openai-responses",
            "provider": "github-copilot",
            "baseUrl": "https://api.individual.githubcopilot.com",
            "reasoning": True,
            "input": ["text", "image"],
            "cost": with_open_ai_long_context_pricing(OPENAI_STANDARD_COSTS[model_id]),
            "contextWindow": 1000000,
            "maxTokens": 128000,
            "headers": dict(COPILOT_STATIC_HEADERS),
        }
        for model_id in ("gpt-6-sol", "gpt-6-luna")
    ],
]

MISSING_OPENAI_MODELS: List[Dict[str, Any]] = [
    {
        "id": model_id,
        "name": name,
        "api": "openai-responses",
        "baseUrl": "https://api.openai.com/v1",
        "provider": "openai",
        "reasoning": True,
        "input": ["text", "image"],
        "cost": with_open_ai_long_context_pricing(OPENAI_STANDARD_COSTS[model_id]),
        "contextWindow": OPENAI_LONG_CONTEXT_INPUT_THRESHOLD,
        "maxTokens": 128000,
    }
    for model_id, name in (
        ("gpt-6-astra", "GPT-6 Astra"),
        ("gpt-6-sol", "GPT-6 Sol"),
        ("gpt-6-luna", "GPT-6 Luna"),
        ("gpt-5.6-sol", "GPT-5.6 Sol"),
        ("gpt-5.6-terra", "GPT-5.6 Terra"),
        ("gpt-5.6-luna", "GPT-5.6 Luna"),
    )
] + [
    {
        "id": "gpt-5-chat-latest",
        "name": "GPT-5 Chat Latest",
        "api": "openai-responses",
        "baseUrl": "https://api.openai.com/v1",
        "provider": "openai",
        "reasoning": False,
        "input": ["text", "image"],
        "cost": {"input": 1.25, "output": 10, "cacheRead": 0.125, "cacheWrite": 0},
        "contextWindow": 128000,
        "maxTokens": 16384,
    },
]

DEEPSEEK_STATIC_MODELS: List[Dict[str, Any]] = [
    {
        "id": "deepseek-flash",
        "name": "DeepSeek V4.1 Flash",
        "api": "openai-completions",
        "baseUrl": "https://api.deepseek.com",
        "provider": "deepseek",
        "reasoning": True,
        "thinkingLevelMap": dict(DEEPSEEK_V4_FLASH_THINKING_LEVEL_MAP),
        "input": ["text", "image"],
        # DeepSeek also offers time-based off-peak rates, which the cost schema cannot represent yet.
        "cost": {"input": 0.3, "output": 1.2, "cacheRead": 0.006, "cacheWrite": 0},
        "contextWindow": 1000000,
        "maxTokens": 384000,
        "compat": dict(DEEPSEEK_COMPAT),
    },
    {
        "id": "deepseek-v4-pro",
        "name": "DeepSeek V4 Pro",
        "api": "openai-completions",
        "baseUrl": "https://api.deepseek.com",
        "provider": "deepseek",
        "reasoning": True,
        "input": ["text"],
        "cost": {"input": 1.32, "output": 3.96, "cacheRead": 0.044, "cacheWrite": 0},
        "contextWindow": 1000000,
        "maxTokens": 384000,
        "compat": dict(DEEPSEEK_COMPAT),
    },
]

ANT_LING_STATIC_MODELS: List[Dict[str, Any]] = [
    {
        "id": "Ling-2.6-flash",
        "name": "Ling 2.6 Flash",
        "api": "openai-completions",
        "baseUrl": "https://api.ant-ling.com/v1",
        "provider": "ant-ling",
        "reasoning": False,
        "input": ["text"],
        "cost": {"input": 0.01, "output": 0.02, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 262144,
        "maxTokens": 65536,
        "compat": dict(ANT_LING_COMPAT),
    },
    {
        "id": "Ling-2.6-1T",
        "name": "Ling 2.6 1T",
        "api": "openai-completions",
        "baseUrl": "https://api.ant-ling.com/v1",
        "provider": "ant-ling",
        "reasoning": False,
        "input": ["text"],
        "cost": {"input": 0.06, "output": 0.25, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 262144,
        "maxTokens": 65536,
        "compat": dict(ANT_LING_COMPAT),
    },
    {
        "id": "Ring-2.6-1T",
        "name": "Ring 2.6 1T",
        "api": "openai-completions",
        "baseUrl": "https://api.ant-ling.com/v1",
        "provider": "ant-ling",
        "reasoning": True,
        "input": ["text"],
        "cost": {"input": 0.06, "output": 0.25, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 262144,
        "maxTokens": 65536,
        "compat": {**ANT_LING_COMPAT, "thinkingFormat": "ant-ling"},
    },
]

CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_CONTEXT = 272000
CODEX_GPT_56_CONTEXT = 272000
CODEX_SPARK_CONTEXT = 128000
CODEX_MAX_TOKENS = 128000

#: OpenAI Codex (ChatGPT OAuth) models. These are not fetched from models.dev;
#: we keep a small, explicit list to avoid aliases. Older model limits are based
#: on observed server behavior; GPT-5.6 and GPT-6 use Codex's 272k default
#: catalog limit.
CODEX_MODELS: List[Dict[str, Any]] = [
    *[
        {
            "id": model_id,
            "name": name,
            "api": "openai-codex-responses",
            "provider": "openai-codex",
            "baseUrl": CODEX_BASE_URL,
            "reasoning": True,
            "input": ["text", "image"],
            "cost": with_open_ai_long_context_pricing(OPENAI_STANDARD_COSTS[model_id]),
            "contextWindow": CODEX_CONTEXT,
            "maxTokens": CODEX_MAX_TOKENS,
        }
        for model_id, name in (
            ("gpt-6-astra", "GPT-6 Astra"),
            ("gpt-6-sol", "GPT-6 Sol"),
            ("gpt-6-luna", "GPT-6 Luna"),
        )
    ],
    {
        "id": "gpt-5.3-codex-spark",
        "name": "GPT-5.3 Codex Spark",
        "api": "openai-codex-responses",
        "provider": "openai-codex",
        "baseUrl": CODEX_BASE_URL,
        "reasoning": True,
        "input": ["text"],
        "cost": {"input": 1.75, "output": 14, "cacheRead": 0.175, "cacheWrite": 0},
        "contextWindow": CODEX_SPARK_CONTEXT,
        "maxTokens": CODEX_MAX_TOKENS,
    },
    {
        "id": "gpt-5.5",
        "name": "GPT-5.5",
        "api": "openai-codex-responses",
        "provider": "openai-codex",
        "baseUrl": CODEX_BASE_URL,
        "reasoning": True,
        "input": ["text", "image"],
        "cost": with_open_ai_long_context_pricing({"input": 5, "output": 30, "cacheRead": 0.5, "cacheWrite": 0}),
        "contextWindow": CODEX_CONTEXT,
        "maxTokens": CODEX_MAX_TOKENS,
    },
    *[
        {
            "id": model_id,
            "name": name,
            "api": "openai-codex-responses",
            "provider": "openai-codex",
            "baseUrl": CODEX_BASE_URL,
            "reasoning": True,
            "input": ["text", "image"],
            "cost": with_open_ai_long_context_pricing(OPENAI_STANDARD_COSTS[model_id]),
            "contextWindow": CODEX_GPT_56_CONTEXT,
            "maxTokens": CODEX_MAX_TOKENS,
        }
        for model_id, name in (
            ("gpt-5.6-luna", "GPT-5.6 Luna"),
            ("gpt-5.6-sol", "GPT-5.6 Sol"),
            ("gpt-5.6-terra", "GPT-5.6 Terra"),
        )
    ],
]

#: Add missing Mistral Medium 3.5 model until models.dev includes it.
MISTRAL_STATIC_MODEL: Dict[str, Any] = {
    "id": "mistral-medium-3.5",
    "name": "Mistral Medium 3.5",
    "api": "mistral-conversations",
    "provider": "mistral",
    "baseUrl": "https://api.mistral.ai",
    "reasoning": True,
    "input": ["text", "image"],
    "cost": {"input": 1.5, "output": 7.5, "cacheRead": 0, "cacheWrite": 0},
    "contextWindow": 262144,  # 256k tokens
    "maxTokens": 262144,
}

#: Add "auto" alias for openrouter/auto.
OPENROUTER_AUTO_MODEL: Dict[str, Any] = {
    "id": "auto",
    "name": "Auto",
    "api": "openai-completions",
    "provider": "openrouter",
    "baseUrl": "https://openrouter.ai/api/v1",
    "reasoning": True,
    "input": ["text", "image"],
    # we dont know about the costs because OpenRouter auto routes to different models
    # and then charges you for the underlying used model
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    "contextWindow": 2000000,
    "maxTokens": 30000,
}

#: Add "fusion" alias for openrouter/fusion. OpenRouter exposes Fusion as a
#: router alias/plugin entry point; its model metadata does not advertise
#: tools, but the alias resolves to a concrete model that can invoke caller
#: tools and has the openrouter:fusion server tool auto-injected.
OPENROUTER_FUSION_MODEL: Dict[str, Any] = {
    "id": "openrouter/fusion",
    "name": "OpenRouter: Fusion",
    "api": "openai-completions",
    "provider": "openrouter",
    "baseUrl": "https://openrouter.ai/api/v1",
    "reasoning": True,
    "input": ["text"],
    # we dont know about the costs because Fusion routes to multiple models
    # and then charges you for the underlying used models
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    "contextWindow": 1000000,
    "maxTokens": 30000,
}


def bedrock_base_url(model_id: str) -> str:
    """Bedrock runtime endpoint for a model id."""
    return BEDROCK_BASE_URL_EU if model_id.startswith("eu.") else BEDROCK_BASE_URL_US


def normalize_nvidia_model_id(model_id: str) -> str:
    """Nvidia's underscored catalog ids map onto its dotted live ids."""
    return model_id.lower().replace("_", ".")


def is_gemma_4_model(model_id: str) -> bool:
    return re.search("gemma-?4", model_id.lower()) is not None


def get_together_compat(model_id: str, reasoning: bool) -> Dict[str, Any]:
    if not reasoning:
        return dict(TOGETHER_BASE_COMPAT)
    if model_id in TOGETHER_REASONING_EFFORT_MODELS:
        return dict(TOGETHER_REASONING_EFFORT_COMPAT)
    if model_id in TOGETHER_TOGGLE_REASONING_EFFORT_MODELS:
        return dict(TOGETHER_TOGGLE_REASONING_EFFORT_COMPAT)
    if model_id in TOGETHER_REASONING_ONLY_MODELS:
        return dict(TOGETHER_BASE_COMPAT)
    return dict(TOGETHER_TOGGLE_REASONING_COMPAT)


def get_together_thinking_level_map(model_id: str, reasoning: bool) -> Optional[Dict[str, Optional[str]]]:
    if not reasoning:
        return None
    if model_id in TOGETHER_REASONING_EFFORT_MODELS:
        return dict(TOGETHER_REASONING_EFFORT_LEVEL_MAP)
    if model_id in TOGETHER_TOGGLE_REASONING_EFFORT_MODELS:
        return dict(TOGETHER_DEEPSEEK_V4_THINKING_LEVEL_MAP)
    if model_id in TOGETHER_REASONING_ONLY_MODELS:
        return dict(TOGETHER_FIXED_REASONING_LEVEL_MAP)
    return dict(TOGETHER_TOGGLE_REASONING_LEVEL_MAP)


__all__ = [name for name in dir() if name.isupper()] + [
    "bedrock_base_url",
    "get_together_compat",
    "get_together_thinking_level_map",
    "is_gemma_4_model",
    "normalize_nvidia_model_id",
    "round_cost",
    "with_open_ai_long_context_pricing",
]

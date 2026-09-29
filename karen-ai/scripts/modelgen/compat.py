"""Per-api compat detection and the catalog metadata appliers.

Port of the `apply*Metadata` / `detectOpenAICompletionsCompat` /
`getAnthropicMessagesCompat` block of pi-ai's `scripts/generate-models.ts`.

Models are plain JSON-shaped dicts here, exactly as they will be serialized, so
every helper mutates the dict in place the way its TypeScript counterpart does.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from .reasoning_options import get_effort_thinking_level_map
from .tables import (
    ANTHROPIC_ALLOWED_FALLBACK_MODELS,
    ANTHROPIC_PROMPT_CACHE,
    DEFAULT_IMAGE_RESIZE,
    DEEPSEEK_V4_FLASH_THINKING_LEVEL_MAP,
    DEEPSEEK_V4_THINKING_LEVEL_MAP,
    EAGER_TOOL_INPUT_STREAMING_UNSUPPORTED_ANTHROPIC_MODELS,
    GITHUB_COPILOT_THINKING_LEVEL_OVERRIDES,
    MID_CONVO_EFFORT_UNSUPPORTED_ANTHROPIC_MODELS,
    OPENAI_COMPLETIONS_DEFAULT_COMPAT,
    OPENAI_CODEX_ADDITIONAL_TOOLS_MODEL_IDS,
    OPENAI_GRAMMAR_TOOL_APIS,
    OPENAI_GRAMMAR_TOOL_PROVIDERS,
    OPENAI_ADDITIONAL_TOOLS_MODEL_IDS,
    OPENAI_MID_CONVO_SYSTEM_MESSAGE_MODEL_IDS,
    OPENAI_RESPONSES_NONE_REASONING_MODELS,
    OPENAI_RESPONSES_PROXY_PROVIDERS,
    OPENAI_TOOL_SEARCH_MODEL_IDS,
    VERIFIED_ANTHROPIC_MID_CONVO_EFFORT_PROVIDERS,
    TOGETHER_REASONING_ONLY_MODELS,
    is_gemma_4_model,
)

Model = MutableMapping[str, Any]
ThinkingLevelMap = Dict[str, Optional[str]]


# ---------------------------------------------------------------------------
# Model-id predicates
# ---------------------------------------------------------------------------


def is_anthropic_adaptive_thinking_model(model_id: str) -> bool:
    return any(
        fragment in model_id
        for fragment in (
            "opus-4-6",
            "opus-4.6",
            "opus-4-7",
            "opus-4.7",
            "opus-4-8",
            "opus-4.8",
            "opus-5",
            "opus.5",
            "sonnet-4-6",
            "sonnet-4.6",
            "sonnet-5",
            "sonnet.5",
            "fable-5",
            "mythos-5",
        )
    )


def is_anthropic_temperature_unsupported_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(fragment in lowered for fragment in ("opus-4-7", "opus-4.7", "opus-4-8", "opus-4.8", "opus-5", "opus.5"))


def supports_anthropic_mid_convo_effort(model_id: str) -> bool:
    normalized = re.sub(r"^~?anthropic/", "", model_id.lower())
    return bool(
        re.fullmatch(r"claude-opus-(?:5|5[.-]5)(?:-\d{8})?", normalized)
        or re.fullmatch(r"claude-(?:fable|mythos)-5(?:[.-]1)(?:-\d{8})?", normalized)
    )


def supports_anthropic_mid_convo_system_messages(model_id: str) -> bool:
    return bool(
        re.fullmatch(r"claude-opus-(?:4[.-]8|5(?:[.-]5)?)(?:-\d{8})?", model_id)
        or re.fullmatch(r"claude-(?:fable|mythos)-5(?:[.-]1)?(?:-\d{8})?", model_id)
    )


def supports_open_ai_xhigh(model_id: str) -> bool:
    return any(
        fragment in model_id
        for fragment in ("gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5", "gpt-5.6", "gpt-6")
    )


def supports_open_ai_max(model: Model) -> bool:
    model_id = model.get("id", "")
    return ("gpt-5.6" in model_id or "gpt-6" in model_id) and model.get("api") in (
        "openai-responses",
        "azure-openai-responses",
        "openai-codex-responses",
        "openai-completions",
    )


def is_anthropic_fallback_metadata_model(model: Model) -> bool:
    if model.get("provider") != "anthropic" or model.get("api") != "anthropic-messages":
        return False
    model_id = model.get("id", "")
    return model_id in ANTHROPIC_ALLOWED_FALLBACK_MODELS or any(
        model_id in fallback_ids for fallback_ids in ANTHROPIC_ALLOWED_FALLBACK_MODELS.values()
    )


# ---------------------------------------------------------------------------
# Thinking-level maps
# ---------------------------------------------------------------------------


def merge_thinking_level_map(model: Model, level_map: Mapping[str, Optional[str]]) -> None:
    model["thinkingLevelMap"] = {**(model.get("thinkingLevelMap") or {}), **level_map}


def merge_anthropic_messages_compat(model: Model, compat: Mapping[str, Any]) -> None:
    model["compat"] = {**(model.get("compat") or {}), **compat}


def merge_openai_completions_compat(model: Model, compat: Mapping[str, Any]) -> None:
    model["compat"] = {**(model.get("compat") or {}), **compat}


def get_google_thinking_level_map(
    model_id: str, reasoning_options: Optional[Sequence[Mapping[str, Any]]]
) -> Optional[ThinkingLevelMap]:
    effort_map = get_effort_thinking_level_map(reasoning_options)
    if effort_map:
        return effort_map
    if is_gemma_4_model(model_id):
        return {"off": None, "minimal": "MINIMAL", "low": None, "medium": None, "high": "HIGH"}
    return None


def get_anthropic_messages_compat(provider: str, model_id: str) -> Optional[Dict[str, Any]]:
    compat: Dict[str, Any] = {}
    if (
        provider in VERIFIED_ANTHROPIC_MID_CONVO_EFFORT_PROVIDERS
        and supports_anthropic_mid_convo_effort(model_id)
        and f"{provider}:{model_id}" not in MID_CONVO_EFFORT_UNSUPPORTED_ANTHROPIC_MODELS
    ):
        compat["supportsMidConvoEffort"] = True
    if provider == "anthropic" and supports_anthropic_mid_convo_system_messages(model_id):
        compat["supportsMidConvoSystemMessages"] = True
        compat["supportsMidConvoToolChanges"] = True
    # OpenCode Zen and GitHub Copilot forward mid-conversation system messages but reject
    # `tool_addition`/`tool_removal` blocks, so tool changes stay top-level there.
    if provider in ("opencode", "github-copilot") and supports_anthropic_mid_convo_system_messages(model_id):
        compat["supportsMidConvoSystemMessages"] = True
    if f"{provider}:{model_id}" in EAGER_TOOL_INPUT_STREAMING_UNSUPPORTED_ANTHROPIC_MODELS:
        compat["supportsEagerToolInputStreaming"] = False
    if provider == "xiaomi" or provider.startswith("xiaomi-token-plan-"):
        compat["allowEmptySignature"] = True
    return compat or None


# ---------------------------------------------------------------------------
# OpenAI completions compat detection
# ---------------------------------------------------------------------------


def detect_openai_completions_compat(model: Model) -> Dict[str, Any]:
    """Derive the compat profile a provider's endpoint needs from its identity."""
    provider = model.get("provider", "")
    base_url = model.get("baseUrl", "")
    model_id = model.get("id", "")

    is_zai = provider in ("zai", "zai-coding-cn") or "api.z.ai" in base_url or "open.bigmodel.cn" in base_url
    is_together = provider == "together" or "api.together.ai" in base_url or "api.together.xyz" in base_url
    is_moonshot = provider in ("moonshotai", "moonshotai-cn") or "api.moonshot." in base_url
    is_open_router = provider == "openrouter" or "openrouter.ai" in base_url
    is_cloudflare_workers_ai = provider == "cloudflare-workers-ai" or "api.cloudflare.com" in base_url
    is_cloudflare_ai_gateway = provider == "cloudflare-ai-gateway" or "gateway.ai.cloudflare.com" in base_url
    is_nvidia = provider == "nvidia" or "integrate.api.nvidia.com" in base_url
    is_ant_ling = provider == "ant-ling" or "api.ant-ling.com" in base_url
    is_cerebras = provider == "cerebras" or "cerebras.ai" in base_url
    is_together_reasoning_only = is_together and model_id in TOGETHER_REASONING_ONLY_MODELS
    is_deepseek = provider == "deepseek" or "deepseek.com" in base_url.lower()

    is_non_standard = (
        is_nvidia
        or is_cerebras
        or provider == "xai"
        or "api.x.ai" in base_url
        or is_together
        or "chutes.ai" in base_url
        or is_deepseek
        or is_zai
        or is_moonshot
        or provider == "opencode"
        or "opencode.ai" in base_url
        or is_cloudflare_workers_ai
        or is_cloudflare_ai_gateway
        or is_ant_ling
    )

    use_max_tokens = (
        "chutes.ai" in base_url
        or is_deepseek
        or is_moonshot
        or is_cloudflare_ai_gateway
        or is_together
        or is_nvidia
        or is_ant_ling
        or is_zai
    )

    is_grok = provider == "xai" or "api.x.ai" in base_url
    is_open_router_developer_role_model = is_open_router and (
        model_id.startswith("anthropic/") or model_id.startswith("openai/")
    )
    cache_control_format = "anthropic" if provider == "openrouter" and re.match(r"^~?anthropic/", model_id) else None

    if is_deepseek:
        thinking_format = "deepseek"
    elif is_zai:
        thinking_format = "zai"
    elif is_together and not is_together_reasoning_only:
        thinking_format = "together"
    elif is_ant_ling:
        thinking_format = "ant-ling"
    elif is_open_router:
        thinking_format = "openrouter"
    else:
        thinking_format = "openai"

    compat: Dict[str, Any] = {
        "supportsStore": not is_non_standard,
        "supportsDeveloperRole": is_open_router_developer_role_model or (not is_non_standard and not is_open_router),
        "supportsReasoningEffort": not (is_grok or is_zai or is_moonshot or is_together or is_cloudflare_ai_gateway or is_nvidia or is_ant_ling),
        "supportsUsageInStreaming": True,
        "supportsFinishReason": True,
        "maxTokensField": "max_tokens" if use_max_tokens else "max_completion_tokens",
        "requiresToolResultName": False,
        "requiresAssistantAfterToolResult": False,
        "requiresThinkingAsText": False,
        "requiresReasoningContentOnAssistantMessages": is_deepseek,
        "thinkingFormat": thinking_format,
        "openRouterRouting": {},
        "vercelGatewayRouting": {},
        "chatTemplateKwargs": {},
        "chatTemplateArgs": {},
        "zaiToolStream": False,
        # Preserve built-in behavior as explicit metadata against the conservative runtime default.
        "supportsStrictMode": not (is_moonshot or is_together or is_cloudflare_ai_gateway or is_nvidia or is_cerebras),
        "supportsOpenAIGrammarTools": False,
        "supportsMidConvoSystemMessages": False,
        "supportsMidConvoToolAdditions": False,
        "sendSessionAffinityHeaders": is_open_router,
        "supportsLongCacheRetention": not (is_together or is_cloudflare_workers_ai or is_cloudflare_ai_gateway or is_nvidia or is_ant_ling),
    }
    if cache_control_format:
        compat["cacheControlFormat"] = cache_control_format
    return compat


def _is_plain_empty_object(value: Any) -> bool:
    return isinstance(value, dict) and not value


def openai_completions_compat_delta(compat: Mapping[str, Any]) -> Dict[str, Any]:
    """Only emit the entries that differ from the runtime defaults."""
    delta: Dict[str, Any] = {}
    for key, value in compat.items():
        default = OPENAI_COMPLETIONS_DEFAULT_COMPAT.get(key)
        if _is_plain_empty_object(value) and _is_plain_empty_object(default):
            continue
        if value != default:
            delta[key] = value
    return delta


# ---------------------------------------------------------------------------
# Metadata appliers
# ---------------------------------------------------------------------------


def apply_openai_completions_compat_metadata(model: Model) -> None:
    if model.get("api") != "openai-completions":
        return
    detected = openai_completions_compat_delta(detect_openai_completions_compat(model))
    compat = {**detected, **(model.get("compat") or {})}
    if compat:
        model["compat"] = compat
    else:
        model.pop("compat", None)


def apply_anthropic_messages_compat_metadata(model: Model) -> None:
    if model.get("api") != "anthropic-messages":
        return
    compat = get_anthropic_messages_compat(model.get("provider", ""), model.get("id", ""))
    if compat:
        merge_anthropic_messages_compat(model, compat)
        if compat.get("supportsMidConvoEffort"):
            merge_thinking_level_map(model, {"off": None})


def apply_anthropic_allowed_fallback_model_metadata(models: Sequence[Model]) -> None:
    models_by_id = {model.get("id"): model for model in models}
    for model_id, fallback_model_ids in ANTHROPIC_ALLOWED_FALLBACK_MODELS.items():
        model = models_by_id.get(model_id)
        if model is None:
            continue

        compatible_fallback_ids = (
            [fallback_id for fallback_id in fallback_model_ids if supports_anthropic_mid_convo_effort(fallback_id)]
            if (model.get("compat") or {}).get("supportsMidConvoEffort")
            else fallback_model_ids
        )
        allowed_fallback_models = []
        for fallback_model_id in compatible_fallback_ids:
            fallback_model = models_by_id.get(fallback_model_id)
            if fallback_model is not None:
                allowed_fallback_models.append(
                    {
                        "provider": fallback_model.get("provider"),
                        "model": fallback_model.get("id"),
                        "cost": fallback_model.get("cost"),
                    }
                )
        if allowed_fallback_models:
            merge_anthropic_messages_compat(model, {"allowedFallbackModels": allowed_fallback_models})


def apply_strict_tool_compat_metadata(model: Model) -> None:
    if model.get("provider") in ("openai", "cloudflare-ai-gateway") and model.get("api") == "openai-responses":
        model["compat"] = {**(model.get("compat") or {}), "supportsStrictMode": True}
    elif model.get("provider") == "anthropic" and model.get("api") == "anthropic-messages":
        merge_anthropic_messages_compat(model, {"supportsStrictTools": True})


def apply_openai_grammar_tool_compat_metadata(model: Model) -> None:
    if model.get("api") not in OPENAI_GRAMMAR_TOOL_APIS or model.get("provider") not in OPENAI_GRAMMAR_TOOL_PROVIDERS:
        return
    match = re.match(r"^gpt-(\d+)", model.get("id", ""))
    if not match or int(match.group(1)) < 5:
        return
    model["compat"] = {**(model.get("compat") or {}), "supportsOpenAIGrammarTools": True}


def apply_openai_tool_search_metadata(model: Model) -> None:
    is_openai_responses = model.get("provider") == "openai" and model.get("api") == "openai-responses"
    is_openai_codex = model.get("provider") == "openai-codex" and model.get("api") == "openai-codex-responses"
    model_id = model.get("id", "")
    if not (is_openai_responses or is_openai_codex) or model_id not in OPENAI_TOOL_SEARCH_MODEL_IDS:
        return
    supports_additional_tools = (
        is_openai_responses and model_id in OPENAI_ADDITIONAL_TOOLS_MODEL_IDS
    ) or (is_openai_codex and model_id in OPENAI_CODEX_ADDITIONAL_TOOLS_MODEL_IDS)
    model["compat"] = {
        **(model.get("compat") or {}),
        **({"supportsAdditionalTools": True} if supports_additional_tools else {}),
        "supportsToolSearch": True,
    }


def apply_openai_completions_transcript_metadata(model: Model) -> None:
    """Mid-conversation system messages on the OpenAI-compatible endpoints."""
    if model.get("api") != "openai-completions":
        return
    provider = model.get("provider", "")
    model_id = model.get("id", "")
    is_kimi_k3 = (
        (provider.startswith("moonshot") and model_id == "kimi-k3")
        or (provider == "fireworks" and "kimi-k3" in model_id)
        or (provider in ("opencode", "opencode-go") and model_id == "kimi-k3")
    )
    is_moonshot_kimi_k2 = provider.startswith("moonshot") and model_id in (
        "kimi-k2.6",
        "kimi-k2.7-code",
        "kimi-k2.7-code-highspeed",
    )
    is_text_only = (
        is_moonshot_kimi_k2
        or (provider == "github-copilot" and model_id == "kimi-k3")
        or (provider == "deepseek" and model_id == "deepseek-v4-pro")
        or (
            provider == "openrouter"
            and model_id.startswith("openai/")
            and model_id[len("openai/") :] in OPENAI_MID_CONVO_SYSTEM_MESSAGE_MODEL_IDS
        )
    )
    if not is_kimi_k3 and not is_text_only:
        return
    model["compat"] = {
        **(model.get("compat") or {}),
        "supportsMidConvoSystemMessages": True,
        **({"supportsMidConvoToolAdditions": True} if is_kimi_k3 else {}),
    }


def apply_openai_responses_transcript_metadata(model: Model) -> None:
    """Newer OpenAI Responses models accept developer messages mid-conversation."""
    is_openai_responses = model.get("provider") == "openai" and model.get("api") == "openai-responses"
    is_openai_codex = model.get("provider") == "openai-codex" and model.get("api") == "openai-codex-responses"
    is_proxied_responses = (
        model.get("provider") in OPENAI_RESPONSES_PROXY_PROVIDERS and model.get("api") == "openai-responses"
    )
    if (
        not (is_openai_responses or is_openai_codex or is_proxied_responses)
        or model.get("id") not in OPENAI_MID_CONVO_SYSTEM_MESSAGE_MODEL_IDS
    ):
        return
    model["compat"] = {
        **(model.get("compat") or {}),
        "supportsMidConvoSystemMessages": True,
        **({"supportsAdditionalTools": True} if is_proxied_responses else {}),
    }


def apply_openai_explicit_prompt_cache_metadata(model: Model) -> None:
    """GPT-5.6-family models are the ones that bill and accept explicit caching."""
    if model.get("provider") != "openai" or model.get("api") != "openai-responses":
        return
    if not ((model.get("cost") or {}).get("cacheWrite", 0) > 0):
        return
    model["compat"] = {**(model.get("compat") or {}), "supportsExplicitPromptCacheMode": True}


def apply_prompt_cache_metadata(model: Model) -> None:
    if model.get("provider") == "anthropic" and model.get("api") == "anthropic-messages":
        model["promptCache"] = dict(ANTHROPIC_PROMPT_CACHE)
    # Do not add OpenAI lifetimes yet. Before enabling warming for explicit
    # OpenAI caches, re-evaluate it using observed expiry, replay, and billing
    # behavior; a documented TTL alone does not establish full cache loss.


def apply_image_input_metadata(model: Model) -> None:
    if "image" not in (model.get("input") or ()):
        return

    provider = model.get("provider")
    if provider == "anthropic":
        provider_limits: Optional[Dict[str, Any]] = {
            "maxRequestBytes": 32 * 1024 * 1024,
            "images": {
                "maxPerRequest": 100 if model.get("type") != "image" and model.get("contextWindow") == 200000 else 600
            },
        }
    elif provider == "amazon-bedrock":
        provider_limits = {"images": {"maxPerMessage": 20}}
    elif provider == "openai":
        provider_limits = {"maxRequestBytes": 512 * 1024 * 1024, "images": {"maxPerRequest": 1500}}
    elif provider == "google":
        provider_limits = {"maxRequestBytes": 20 * 1024 * 1024, "images": {"maxPerRequest": 3600}}
    else:
        provider_limits = None

    configured_images = ((model.get("inputLimits") or {}).get("images")) or {}
    model["inputLimits"] = {
        **(provider_limits or {}),
        **(model.get("inputLimits") or {}),
        "images": {
            **((provider_limits or {}).get("images") or {}),
            **configured_images,
            "resize": {**DEFAULT_IMAGE_RESIZE, **(configured_images.get("resize") or {})},
        },
    }


def supports_direct_reasoning_effort(model: Model) -> bool:
    api = model.get("api")
    if api == "anthropic-messages":
        return (model.get("compat") or {}).get("forceAdaptiveThinking") is True
    if api in ("openai-responses", "azure-openai-responses", "openai-codex-responses"):
        return True
    if api != "openai-completions":
        return False

    compat = {**detect_openai_completions_compat(model), **(model.get("compat") or {})}
    return compat.get("thinkingFormat") == "openai" and bool(compat.get("supportsReasoningEffort"))


def apply_models_dev_reasoning_option_metadata(model: Model, reasoning_options_by_model: Mapping[str, Any]) -> None:
    reasoning_options = reasoning_options_by_model.get(f"{model.get('provider')}:{model.get('id')}")
    if not reasoning_options or not supports_direct_reasoning_effort(model):
        return
    level_map = get_effort_thinking_level_map(reasoning_options)
    if level_map:
        merge_thinking_level_map(model, level_map)


def apply_thinking_level_metadata(model: Model, reasoning_options_by_model: Mapping[str, Any]) -> None:
    """Every thinking-level correction pi-ai bakes into the generated catalogs."""
    api = model.get("api")
    model_id = model.get("id", "")
    provider = model.get("provider", "")

    if api in ("openai-responses", "azure-openai-responses") and model_id.startswith("gpt-5"):
        merge_thinking_level_map(model, {"off": None})
    if model_id in ("gpt-6-astra", "gpt-6-sol", "gpt-6-luna") and api in (
        "openai-responses",
        "azure-openai-responses",
        "openai-codex-responses",
    ):
        merge_thinking_level_map(
            model,
            {
                "off": None if model_id == "gpt-6-astra" else "none",
                "minimal": None,
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "xhigh",
                "max": "max",
            },
        )
    if provider == "github-copilot" and model_id.startswith("gpt-5"):
        merge_thinking_level_map(model, {"minimal": "low"})
    if (
        api == "openai-responses"
        and provider == "openai"
        and model_id in OPENAI_RESPONSES_NONE_REASONING_MODELS
    ):
        merge_thinking_level_map(model, {"off": "none"})
    # xAI models without verified effort options must not send the undocumented
    # "none"/"minimal" efforts.
    if provider == "xai" and api == "openai-responses" and model.get("thinkingLevelMap") is None:
        merge_thinking_level_map(model, {"off": None, "minimal": None})
    if supports_open_ai_xhigh(model_id):
        merge_thinking_level_map(model, {"xhigh": "xhigh"})
    if supports_open_ai_max(model):
        merge_thinking_level_map(model, {"max": "max"})
    if provider == "openai" and model_id == "gpt-5.5":
        merge_thinking_level_map(model, {"minimal": None})
    if model_id.endswith("gpt-5.5-pro"):
        merge_thinking_level_map(model, {"off": None, "minimal": None, "low": None})

    # Anthropic adaptive-thinking effort support (per Anthropic adaptive thinking docs):
    # - "max" is available on all adaptive-thinking Claude models.
    # - "xhigh" is only available on Opus 4.7/4.8/5, Sonnet 5, and Fable 5.
    if any(fragment in model_id for fragment in ("opus-4-6", "opus-4.6", "sonnet-4-6", "sonnet-4.6")):
        merge_thinking_level_map(model, {"max": "max"})
    if any(
        fragment in model_id
        for fragment in (
            "opus-4-7",
            "opus-4.7",
            "opus-4-8",
            "opus-4.8",
            "opus-5",
            "opus.5",
            "sonnet-5",
            "sonnet.5",
        )
    ):
        merge_thinking_level_map(model, {"xhigh": "xhigh", "max": "max"})
    if "fable-5" in model_id:
        merge_thinking_level_map(model, {"off": None, "xhigh": "xhigh", "max": "max"})

    if api == "anthropic-messages" and is_anthropic_adaptive_thinking_model(model_id):
        merge_anthropic_messages_compat(model, {"forceAdaptiveThinking": True})
    if api == "anthropic-messages" and is_anthropic_temperature_unsupported_model(model_id):
        merge_anthropic_messages_compat(model, {"supportsTemperature": False})

    if api == "openai-completions" and "deepseek-v4" in model_id and model.get("thinkingLevelMap") is None:
        if provider == "openrouter":
            merge_thinking_level_map(model, {**DEEPSEEK_V4_THINKING_LEVEL_MAP, "xhigh": "xhigh", "max": None})
        elif provider in ("deepseek", "opencode", "opencode-go") and "deepseek-v4-flash" in model_id:
            merge_thinking_level_map(model, DEEPSEEK_V4_FLASH_THINKING_LEVEL_MAP)
        else:
            merge_thinking_level_map(model, DEEPSEEK_V4_THINKING_LEVEL_MAP)

    if provider == "groq" and model_id == "qwen/qwen3.6-27b":
        merge_thinking_level_map(model, {"minimal": None, "low": None, "medium": None, "high": "default"})
    if provider == "openai-codex" and supports_open_ai_xhigh(model_id):
        merge_thinking_level_map(model, {"minimal": "low"})
    if provider in ("moonshotai", "moonshotai-cn") and model_id in ("kimi-k2.7-code", "kimi-k2.7-code-highspeed"):
        # Kimi K2.7 Code is always-thinking. Official docs say
        # `thinking: { type: "disabled" }` is rejected, and callers can omit
        # the thinking parameter to use the enabled default.
        merge_thinking_level_map(model, {"off": None})
    if provider == "openrouter" and model_id.startswith("inception/mercury-2"):
        # Mercury 2 in instant mode (reasoning_effort: "none") disables tool calling.
        # Mark "off" unsupported so the openai-completions provider omits the reasoning param
        # instead of defaulting to {reasoning:{effort:"none"}}.
        merge_thinking_level_map(model, {"off": None})
    if provider == "openrouter" and model_id == "z-ai/glm-5.2":
        merge_thinking_level_map(model, {"xhigh": "xhigh"})

    if provider == "fireworks":
        if api == "anthropic-messages" and (model.get("compat") or {}).get("forceAdaptiveThinking"):
            # Qwen Max currently advertises only a toggle. Prefer upstream effort
            # metadata once available instead of replacing it with this fallback.
            if model_id == "accounts/fireworks/models/qwen3p8-max" and not model.get("thinkingLevelMap"):
                model["thinkingLevelMap"] = get_effort_thinking_level_map(
                    [{"type": "effort", "values": ["low", "medium", "xhigh"]}]
                )
            reasoning_options = reasoning_options_by_model.get(f"{provider}:{model_id}")
            if any(option.get("type") == "toggle" for option in reasoning_options or ()) or model_id == (
                "accounts/fireworks/models/qwen3p8-2p4t-a95b"
            ):
                merge_thinking_level_map(model, {"off": "none"})
            if model_id == "accounts/fireworks/models/deepseek-v4-pro-0813":
                merge_thinking_level_map(model, {"low": "low"})
        if "glm-5p2" in model_id:
            # GLM 5.2 and its fast router support off/high/max. Fireworks maps low
            # and medium to high, so do not expose those aliases as distinct levels.
            merge_thinking_level_map(model, {"off": "none", "minimal": None, "low": None, "medium": None, "max": "max"})
        if "kimi-k3" in model_id:
            # Fireworks maps medium to high on both APIs; do not expose it as a distinct level.
            merge_thinking_level_map(model, {"medium": None})

    if provider == "opencode-go" and model_id == "glm-5.2":
        merge_thinking_level_map(
            model,
            {"off": None, "minimal": None, "low": None, "medium": None, "high": "high", "max": "max"},
        )
    if provider == "opencode-go" and model_id == "kimi-k2.6":
        # OpenCode Go exposes Kimi K2.6 thinking as on/off, not distinct effort tiers.
        merge_thinking_level_map(model, {"minimal": None, "low": None, "medium": None})
    if provider == "opencode" and model_id == "grok-build-0.1":
        # OpenCode Zen Grok Build reasons by default but rejects explicit reasoningEffort.
        merge_thinking_level_map(model, {"off": None, "minimal": None, "low": None, "medium": None})
    if provider == "ant-ling" and model.get("reasoning"):
        # Ring reasons by default. Only high/xhigh have documented explicit effort controls.
        merge_thinking_level_map(model, {"off": None, "minimal": None, "low": None, "medium": None, "high": "high", "xhigh": "xhigh"})
    if provider == "github-copilot":
        override = GITHUB_COPILOT_THINKING_LEVEL_OVERRIDES.get(model_id)
        if override:
            merge_thinking_level_map(model, override)


#: The appliers that run over every chat model, in pi-ai's order.
CHAT_METADATA_APPLIERS: List[Any] = [
    apply_openai_completions_compat_metadata,
    apply_anthropic_messages_compat_metadata,
    apply_models_dev_reasoning_option_metadata,
    apply_thinking_level_metadata,
    apply_strict_tool_compat_metadata,
    apply_openai_grammar_tool_compat_metadata,
    apply_openai_tool_search_metadata,
    apply_openai_completions_transcript_metadata,
    apply_openai_responses_transcript_metadata,
    apply_openai_explicit_prompt_cache_metadata,
    apply_prompt_cache_metadata,
    apply_image_input_metadata,
]


def apply_chat_metadata(model: Model, reasoning_options_by_model: Mapping[str, Any]) -> None:
    """Run every chat-model applier over one model, in pi-ai's order."""
    for applier in CHAT_METADATA_APPLIERS:
        if applier in (apply_models_dev_reasoning_option_metadata, apply_thinking_level_metadata):
            applier(model, reasoning_options_by_model)
        else:
            applier(model)


__all__ = [
    "CHAT_METADATA_APPLIERS",
    "apply_anthropic_allowed_fallback_model_metadata",
    "apply_anthropic_messages_compat_metadata",
    "apply_chat_metadata",
    "apply_image_input_metadata",
    "apply_models_dev_reasoning_option_metadata",
    "apply_openai_completions_compat_metadata",
    "apply_openai_completions_transcript_metadata",
    "apply_openai_explicit_prompt_cache_metadata",
    "apply_openai_grammar_tool_compat_metadata",
    "apply_openai_responses_transcript_metadata",
    "apply_openai_tool_search_metadata",
    "apply_prompt_cache_metadata",
    "apply_strict_tool_compat_metadata",
    "apply_thinking_level_metadata",
    "detect_openai_completions_compat",
    "get_anthropic_messages_compat",
    "get_google_thinking_level_map",
    "is_anthropic_adaptive_thinking_model",
    "is_anthropic_fallback_metadata_model",
    "is_anthropic_temperature_unsupported_model",
    "merge_anthropic_messages_compat",
    "merge_openai_completions_compat",
    "merge_thinking_level_map",
    "openai_completions_compat_delta",
    "supports_anthropic_mid_convo_effort",
    "supports_anthropic_mid_convo_system_messages",
    "supports_direct_reasoning_effort",
    "supports_open_ai_max",
    "supports_open_ai_xhigh",
]

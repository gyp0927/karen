"""models.dev -> chat model catalog.

Port of `loadModelsDevData`, `loadModelsDevClassifierModels` and the
`process*Models` helpers in pi-ai's `scripts/generate-models.ts`.

`reasoning_options_by_model` is the module-level `modelsDevReasoningOptions` map
from pi-ai, passed explicitly so the pipeline stays testable without globals. It
records the raw models.dev reasoning options per `provider:model` so the later
metadata pass can decide whether a model actually supports direct effort
control.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from .compat import get_google_thinking_level_map
from .model_data import assert_exact_model_ids
from .reasoning_options import get_effort_thinking_level_map
from .sources import get_json
from .tables import (
    BASETEN_BASE_COMPAT,
    BASETEN_GLM52_THINKING_LEVEL_MAP,
    BASETEN_REASONING_EFFORT_COMPAT,
    BASETEN_TOGGLE_REASONING_COMPAT,
    BASETEN_TOGGLE_REASONING_EFFORT_COMPAT,
    BASETEN_TOGGLE_THINKING_LEVEL_MAP,
    BEDROCK_INFERENCE_PROFILE_ONLY_MODEL_IDS,
    CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL,
    CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL,
    CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL,
    CLOUDFLARE_WORKERS_AI_BASE_URL,
    COPILOT_STATIC_HEADERS,
    FIREWORKS_ADAPTIVE_THINKING_FALLBACK_MODELS,
    FIREWORKS_ANTHROPIC_COMPAT,
    FIREWORKS_KIMI_K3_COMPAT,
    FIREWORKS_OPENAI_COMPAT,
    KIMI_ALIASES,
    KIMI_CODING_IMPLIED_COSTS,
    KIMI_K3_COST,
    MODELS_DEV_OPENAI_UNSUPPORTED_MODEL_IDS,
    MOONSHOT_COMPAT,
    NVIDIA_BASE_URL,
    NVIDIA_HEADERS,
    NVIDIA_NIM_UNSUPPORTED_MODELS,
    NVIDIA_OPENAI_COMPAT,
    OPENCODE_LONG_CACHE_RETENTION_UNSUPPORTED_MODELS,
    QWEN_TOKEN_PLAN_COMPAT,
    QWEN_TOKEN_PLAN_EXCLUDED_MODEL_IDS,
    QWEN_TOKEN_PLAN_FALLBACK_THINKING_LEVEL_MAP,
    QWEN_TOKEN_PLAN_INDIVIDUAL_MODEL_IDS,
    QWEN_TOKEN_PLAN_REASONING_EFFORT_FALLBACK_MODEL_IDS,
    TOGETHER_BASE_URL,
    VERTEX_BASE_URL,
    XAI_RESPONSES_COMPAT,
    XIAOMI_COMPAT,
    ZAI_TOOL_STREAM_UNSUPPORTED_MODELS,
    bedrock_base_url,
    get_together_compat,
    get_together_thinking_level_map,
    normalize_nvidia_model_id,
    round_cost,
)

ModelsDevCatalog = Mapping[str, Mapping[str, Any]]


# ---------------------------------------------------------------------------
# Small models.dev accessors
# ---------------------------------------------------------------------------


def _models(data: ModelsDevCatalog, key: str) -> Mapping[str, Any]:
    return (data.get(key) or {}).get("models") or {}


def _tool_callable(model: Mapping[str, Any]) -> bool:
    return model.get("tool_call") is True


def _not_deprecated(model: Mapping[str, Any]) -> bool:
    return model.get("status") != "deprecated"


def _supports_image(model: Mapping[str, Any]) -> bool:
    return "image" in ((model.get("modalities") or {}).get("input") or ())


def _cost(model: Mapping[str, Any]) -> Dict[str, float]:
    cost = model.get("cost") or {}
    return {
        "input": cost.get("input") or 0,
        "output": cost.get("output") or 0,
        "cacheRead": cost.get("cache_read") or 0,
        "cacheWrite": cost.get("cache_write") or 0,
    }


def _limits(model: Mapping[str, Any], default_context: int = 4096) -> Dict[str, int]:
    limit = model.get("limit") or {}
    return {
        "contextWindow": limit.get("context") or default_context,
        "maxTokens": limit.get("output") or 4096,
    }


def _record(reasoning_options_by_model: MutableMapping[str, Any], provider: str, model_id: str, model: Mapping[str, Any]) -> None:
    if model.get("reasoning_options") is not None:
        reasoning_options_by_model[f"{provider}:{model_id}"] = model["reasoning_options"]


def get_models_dev_cost(cost: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """models.dev cost metadata, including the long-context pricing tiers."""
    tiers: List[Dict[str, Any]] = []
    for tier in (cost or {}).get("tiers") or ():
        context = tier.get("tier") or {}
        if context.get("type") != "context" or context.get("size") is None:
            continue
        tiers.append(
            {
                "inputTokensAbove": context["size"],
                "input": tier.get("input") or 0,
                "output": tier.get("output") or 0,
                "cacheRead": tier.get("cache_read") or 0,
                "cacheWrite": tier.get("cache_write") or 0,
            }
        )

    result: Dict[str, Any] = {
        "input": (cost or {}).get("input") or 0,
        "output": (cost or {}).get("output") or 0,
        "cacheRead": (cost or {}).get("cache_read") or 0,
        "cacheWrite": (cost or {}).get("cache_write") or 0,
    }
    if tiers:
        result["tiers"] = tiers
    return result


# ---------------------------------------------------------------------------
# Per-provider processors
# ---------------------------------------------------------------------------


def process_zai_models(data: ModelsDevCatalog, reasoning_options_by_model: MutableMapping[str, Any]) -> List[Dict[str, Any]]:
    variants = (
        ("zai-coding-plan", "zai", "https://api.z.ai/api/coding/paas/v4"),
        ("zhipuai-coding-plan", "zai-coding-cn", "https://open.bigmodel.cn/api/coding/paas/v4"),
    )
    models: List[Dict[str, Any]] = []

    for source, provider, base_url in variants:
        for model_id, model in _models(data, source).items():
            if not _tool_callable(model):
                continue
            thinking_level_map = get_effort_thinking_level_map(model.get("reasoning_options") or [])
            is_glm52 = model_id in ("glm-5.2", "glm-5.2-highspeed")
            if thinking_level_map and is_glm52:
                thinking_level_map["off"] = "none"
            supports_reasoning_effort = thinking_level_map is not None
            reference_cost = (_models(data, "zai").get(model_id) or {}).get("cost") or model.get("cost")

            models.append(
                {
                    "id": model_id,
                    "name": model.get("name") or model_id,
                    "api": "openai-completions",
                    "provider": provider,
                    "baseUrl": base_url,
                    "reasoning": model.get("reasoning") is True,
                    **({"thinkingLevelMap": thinking_level_map} if thinking_level_map else {}),
                    "input": ["text", "image"] if _supports_image(model) else ["text"],
                    "cost": {
                        "input": (reference_cost or {}).get("input") or 0,
                        "output": (reference_cost or {}).get("output") or 0,
                        "cacheRead": (reference_cost or {}).get("cache_read") or 0,
                        "cacheWrite": (reference_cost or {}).get("cache_write") or 0,
                    },
                    "compat": {
                        "supportsDeveloperRole": False,
                        "thinkingFormat": "zai",
                        **({"supportsReasoningEffort": True} if supports_reasoning_effort else {}),
                        **({} if model_id in ZAI_TOOL_STREAM_UNSUPPORTED_MODELS else {"zaiToolStream": True}),
                    },
                    "contextWindow": (model.get("limit") or {}).get("context") or 4096,
                    "maxTokens": (model.get("limit") or {}).get("output") or 4096,
                }
            )
            _record(reasoning_options_by_model, provider, model_id, model)

    return models


def process_baseten_models(provider: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not provider or not provider.get("models"):
        return []

    base_url = "https://inference.baseten.co/v1"
    models: List[Dict[str, Any]] = []

    for model_id, model in provider["models"].items():
        if not _not_deprecated(model):
            continue

        reasoning = model.get("reasoning") is True
        reasoning_options = model.get("reasoning_options") or []
        is_glm52 = model_id in ("zai-org/GLM-5.2", "zai-org/GLM-5.2-Fast")
        supports_toggle = any(option.get("type") == "toggle" for option in reasoning_options) or is_glm52
        supports_effort = any(option.get("type") == "effort" for option in reasoning_options) or is_glm52

        if supports_toggle and supports_effort:
            compat = BASETEN_TOGGLE_REASONING_EFFORT_COMPAT
        elif supports_toggle:
            compat = BASETEN_TOGGLE_REASONING_COMPAT
        elif supports_effort:
            compat = BASETEN_REASONING_EFFORT_COMPAT
        else:
            compat = BASETEN_BASE_COMPAT

        if is_glm52:
            thinking_level_map = BASETEN_GLM52_THINKING_LEVEL_MAP
        elif supports_toggle:
            thinking_level_map = BASETEN_TOGGLE_THINKING_LEVEL_MAP
        else:
            thinking_level_map = get_effort_thinking_level_map(reasoning_options)

        # Baseten's GLM-5.2 endpoints are text-only despite models.dev reporting image input.
        supports_image_input = not is_glm52 and _supports_image(model)

        models.append(
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "api": "openai-completions",
                "provider": "baseten",
                "baseUrl": base_url,
                "reasoning": reasoning,
                **({"thinkingLevelMap": dict(thinking_level_map)} if thinking_level_map else {}),
                "input": ["text", "image"] if supports_image_input else ["text"],
                "cost": _cost(model),
                "compat": dict(compat),
                **_limits(model),
            }
        )

    return models


def process_google_models(data: ModelsDevCatalog) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []

    google_models = _models(data, "google")
    if google_models:
        for model_id, model in google_models.items():
            if not _tool_callable(model):
                continue
            if model_id == "gemini-flash-latest":
                source = google_models.get("gemini-3.5-flash") or model
            elif model_id == "gemini-flash-lite-latest":
                source = google_models.get("gemini-3.1-flash-lite") or model
            else:
                source = model
            thinking_level_map = get_google_thinking_level_map(model_id, source.get("reasoning_options") or [])

            models.append(
                {
                    "id": model_id,
                    "name": model.get("name") or model_id,
                    "api": "google-generative-ai",
                    "provider": "google",
                    "baseUrl": "https://generativelanguage.googleapis.com/v1beta",
                    "reasoning": source.get("reasoning") is True,
                    **({"thinkingLevelMap": thinking_level_map} if thinking_level_map else {}),
                    "input": ["text", "image"] if _supports_image(source) else ["text"],
                    "cost": _cost(source),
                    **_limits(source),
                }
            )

    # The google-vertex models.dev catalog also includes Claude, OpenAI, and other
    # MaaS models that do not use the @google/genai Gemini streaming path.
    vertex_models = _models(data, "google-vertex")
    if vertex_models:
        for model_id, model in vertex_models.items():
            if not _tool_callable(model) or not model_id.startswith("gemini-"):
                continue
            if model_id == "gemini-3.1-flash-lite-preview":
                continue
            if model_id == "gemini-flash-latest":
                source = vertex_models.get("gemini-3.5-flash") or model
            elif model_id == "gemini-flash-lite-latest":
                source = vertex_models.get("gemini-3.1-flash-lite") or model
            else:
                source = model
            thinking_level_map = get_google_thinking_level_map(model_id, source.get("reasoning_options") or [])
            # models.dev reports Vertex cache_read/cache_write values for Gemini 2.5 Flash that
            # do not match the official Gemini API standard pricing table. pi only accounts
            # cachedContentTokenCount as cacheRead.
            cache_read = 0.03 if model_id == "gemini-2.5-flash" else (source.get("cost") or {}).get("cache_read") or 0

            models.append(
                {
                    "id": model_id,
                    "name": model.get("name") or model_id,
                    "api": "google-vertex",
                    "provider": "google-vertex",
                    "baseUrl": VERTEX_BASE_URL,
                    "reasoning": source.get("reasoning") is True,
                    **({"thinkingLevelMap": thinking_level_map} if thinking_level_map else {}),
                    "input": ["text", "image"] if _supports_image(source) else ["text"],
                    "cost": {
                        "input": (source.get("cost") or {}).get("input") or 0,
                        "output": (source.get("cost") or {}).get("output") or 0,
                        "cacheRead": cache_read,
                        "cacheWrite": 0,
                    },
                    **_limits(source),
                }
            )

    return models


def process_fireworks_models(
    provider: Optional[Mapping[str, Any]], reasoning_options_by_model: MutableMapping[str, Any]
) -> List[Dict[str, Any]]:
    if not provider or not provider.get("models"):
        return []

    models: List[Dict[str, Any]] = []
    for model_id, model in provider["models"].items():
        if not _tool_callable(model):
            continue

        common: Dict[str, Any] = {
            "id": model_id,
            "name": model.get("name") or model_id,
            "provider": "fireworks",
            "reasoning": model.get("reasoning") is True,
            "input": ["text", "image"] if _supports_image(model) else ["text"],
            "cost": _cost(model),
            **_limits(model),
        }

        if "glm-" in model_id:
            models.append(
                {
                    **common,
                    "api": "openai-completions",
                    "baseUrl": "https://api.fireworks.ai/inference/v1",
                    "compat": dict(FIREWORKS_OPENAI_COMPAT),
                }
            )
        elif "kimi-k3" in model_id:
            models.append(
                {
                    **common,
                    "api": "openai-completions",
                    "baseUrl": "https://api.fireworks.ai/inference/v1",
                    "compat": dict(FIREWORKS_KIMI_K3_COMPAT),
                }
            )
        else:
            models.append(
                {
                    **common,
                    "api": "anthropic-messages",
                    # Fireworks Anthropic-compatible API - SDK appends /v1/messages.
                    "baseUrl": "https://api.fireworks.ai/inference",
                    # Fireworks prompt caching uses automatic prefix matching + session affinity.
                    # x-session-affinity routes requests to the same replica for cache hits.
                    # cache_control on tools and eager_input_streaming are not supported.
                    # See: https://docs.fireworks.ai/tools-sdks/anthropic-compatibility
                    # Use adaptive thinking for cataloged effort controls, with verified
                    # fallbacks where models.dev is incomplete. New models need no allowlist entry.
                    "compat": {
                        **FIREWORKS_ANTHROPIC_COMPAT,
                        **(
                            {"forceAdaptiveThinking": True}
                            if any(
                                option.get("type") == "effort"
                                for option in (model.get("reasoning_options") or [])
                            )
                            or model_id in FIREWORKS_ADAPTIVE_THINKING_FALLBACK_MODELS
                            else {}
                        ),
                    },
                }
            )
        _record(reasoning_options_by_model, "fireworks", model_id, model)

    return models


# ---------------------------------------------------------------------------
# Classifier models
# ---------------------------------------------------------------------------


def load_models_dev_classifier_models(strict: bool = False) -> List[Dict[str, Any]]:
    """TypeSafe's System One decision model from models.dev."""
    try:
        print("Fetching classifier models from models.dev API...")
        data = get_json("https://models.dev/models.json?type=decision")
        metadata = data.get("typesafe/jev-latest")
        if not metadata or metadata.get("type") != "decision":
            raise RuntimeError("models.dev did not return decision model typesafe/jev-latest")
        return [
            {
                "type": "classifier",
                "id": "jev-latest",
                "name": metadata["name"],
                "api": "typesafe-system-one",
                "provider": "typesafe",
                "baseUrl": "https://api.typesafe.ai/v1/",
                "input": ["text", "image"] if "image" in ((metadata.get("modalities") or {}).get("input") or ()) else ["text"],
                # The canonical models.dev entry has no direct-provider pricing and
                # System One reports no token usage.
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": (metadata.get("limit") or {}).get("context") or 64000,
            }
        ]
    except Exception as error:  # noqa: BLE001 - mirror pi-ai's catch-all
        print(f"Failed to load models.dev classifier data: {error}")
        if strict:
            raise
        return []


# ---------------------------------------------------------------------------
# The models.dev pipeline
# ---------------------------------------------------------------------------


def _append(models: List[Dict[str, Any]], entry: Dict[str, Any]) -> None:
    models.append(entry)


def load_models_dev_data(
    data: ModelsDevCatalog,
    reasoning_options_by_model: MutableMapping[str, Any],
    nvidia_nim_model_ids: Mapping[str, str],
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """Map the whole models.dev catalog onto Pi model entries."""
    models: List[Dict[str, Any]] = []

    # Amazon Bedrock
    for model_id, model in _models(data, "amazon-bedrock").items():
        if not _tool_callable(model) or model_id in BEDROCK_INFERENCE_PROFILE_ONLY_MODEL_IDS:
            continue
        if model_id.startswith("ai21.jamba"):
            # These models doesn't support tool use in streaming mode
            continue
        if model_id.startswith("mistral.mistral-7b-instruct-v0"):
            # These models doesn't support system messages
            continue

        _append(
            models,
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "api": "bedrock-converse-stream",
                "provider": "amazon-bedrock",
                "baseUrl": bedrock_base_url(model_id),
                "reasoning": model.get("reasoning") is True,
                "input": ["text", "image"] if _supports_image(model) else ["text"],
                "cost": _cost(model),
                **_limits(model),
                **({"compat": {"supportsStrictMode": True}} if model.get("structured_output") is True else {}),
            },
        )
        _record(reasoning_options_by_model, "amazon-bedrock", model_id, model)

    # Anthropic
    for model_id, model in _models(data, "anthropic").items():
        if not _tool_callable(model):
            continue
        _append(
            models,
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "api": "anthropic-messages",
                "provider": "anthropic",
                "baseUrl": "https://api.anthropic.com",
                "reasoning": model.get("reasoning") is True,
                "input": ["text", "image"] if _supports_image(model) else ["text"],
                "cost": _cost(model),
                **_limits(model),
            },
        )
        _record(reasoning_options_by_model, "anthropic", model_id, model)

    models.extend(process_google_models(data))

    # OpenAI
    for model_id, model in _models(data, "openai").items():
        if not _tool_callable(model) or model_id in MODELS_DEV_OPENAI_UNSUPPORTED_MODEL_IDS:
            continue
        _append(
            models,
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "api": "openai-responses",
                "provider": "openai",
                "baseUrl": "https://api.openai.com/v1",
                "reasoning": model.get("reasoning") is True,
                "input": ["text", "image"] if _supports_image(model) else ["text"],
                "cost": _cost(model),
                **_limits(model),
            },
        )
        _record(reasoning_options_by_model, "openai", model_id, model)

    # Groq
    for model_id, model in _models(data, "groq").items():
        if not _tool_callable(model):
            continue
        _append(
            models,
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "api": "openai-completions",
                "provider": "groq",
                "baseUrl": "https://api.groq.com/openai/v1",
                "reasoning": model.get("reasoning") is True,
                "input": ["text", "image"] if _supports_image(model) else ["text"],
                "cost": _cost(model),
                **_limits(model),
            },
        )
        _record(reasoning_options_by_model, "groq", model_id, model)

    # Cerebras
    for model_id, model in _models(data, "cerebras").items():
        if not _tool_callable(model):
            continue
        _append(
            models,
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "api": "openai-completions",
                "provider": "cerebras",
                "baseUrl": "https://api.cerebras.ai/v1",
                "reasoning": model.get("reasoning") is True,
                "input": ["text", "image"] if _supports_image(model) else ["text"],
                "cost": _cost(model),
                **_limits(model),
            },
        )
        _record(reasoning_options_by_model, "cerebras", model_id, model)

    # Cloudflare Workers AI
    for model_id, model in _models(data, "cloudflare-workers-ai").items():
        if not _tool_callable(model):
            continue
        _append(
            models,
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "api": "openai-completions",
                "provider": "cloudflare-workers-ai",
                "baseUrl": CLOUDFLARE_WORKERS_AI_BASE_URL,
                "reasoning": model.get("reasoning") is True,
                "input": ["text", "image"] if _supports_image(model) else ["text"],
                "cost": _cost(model),
                **_limits(model),
                "compat": {"sendSessionAffinityHeaders": True},
            },
        )
        _record(reasoning_options_by_model, "cloudflare-workers-ai", model_id, model)

    # Cloudflare AI Gateway
    cloudflare_ai_gateway_model_ids = set()
    for prefixed_id, model in _models(data, "cloudflare-ai-gateway").items():
        if not _tool_callable(model):
            continue
        slash_index = prefixed_id.find("/")
        if slash_index == -1:
            continue
        upstream = prefixed_id[:slash_index]
        native_id = prefixed_id[slash_index + 1 :]

        if upstream == "openai":
            api = "openai-responses"
            base_url = CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL
            model_id = native_id
        elif upstream == "anthropic":
            api = "anthropic-messages"
            base_url = CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL
            model_id = native_id
        elif upstream == "workers-ai":
            api = "openai-completions"
            base_url = CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL
            model_id = prefixed_id
        else:
            continue

        # Gateway passthroughs forward session affinity headers to upstreams that
        # use them for cache/routing affinity.
        compat = {"sendSessionAffinityHeaders": True} if upstream in ("anthropic", "workers-ai") else None

        cloudflare_ai_gateway_model_ids.add(model_id)
        _append(
            models,
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "api": api,
                "provider": "cloudflare-ai-gateway",
                "baseUrl": base_url,
                "reasoning": model.get("reasoning") is True,
                "input": ["text", "image"] if _supports_image(model) else ["text"],
                "cost": _cost(model),
                **_limits(model),
                **({"compat": compat} if compat else {}),
            },
        )
        _record(reasoning_options_by_model, "cloudflare-ai-gateway", model_id, model)

    # The gateway proxies Workers AI through its OpenAI-compatible /compat endpoint,
    # but models.dev may omit or intermittently drop those `workers-ai/*` entries
    # from the AI Gateway catalog. Mirror the Workers AI catalog under the documented
    # prefix so the gateway keeps its OpenAI-compatible models stable.
    for model_id, model in _models(data, "cloudflare-workers-ai").items():
        if not _tool_callable(model):
            continue
        mirrored_id = f"workers-ai/{model_id}"
        if mirrored_id in cloudflare_ai_gateway_model_ids:
            continue
        cloudflare_ai_gateway_model_ids.add(mirrored_id)
        _append(
            models,
            {
                "id": mirrored_id,
                "name": model.get("name") or mirrored_id,
                "api": "openai-completions",
                "provider": "cloudflare-ai-gateway",
                "baseUrl": CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL,
                "reasoning": model.get("reasoning") is True,
                "input": ["text", "image"] if _supports_image(model) else ["text"],
                "cost": _cost(model),
                **_limits(model),
                "compat": {"sendSessionAffinityHeaders": True},
            },
        )
        _record(reasoning_options_by_model, "cloudflare-ai-gateway", mirrored_id, model)

    # xAI
    for model_id, model in _models(data, "xai").items():
        if not _tool_callable(model):
            continue
        _append(
            models,
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "api": "openai-responses",
                "provider": "xai",
                "baseUrl": "https://api.x.ai/v1",
                "compat": dict(XAI_RESPONSES_COMPAT),
                "reasoning": model.get("reasoning") is True,
                "input": ["text", "image"] if _supports_image(model) else ["text"],
                "cost": get_models_dev_cost(model.get("cost")),
                **_limits(model),
            },
        )
        _record(reasoning_options_by_model, "xai", model_id, model)

    # Meta
    for model_id, model in _models(data, "meta").items():
        if not _tool_callable(model):
            continue
        _append(
            models,
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "api": "openai-responses",
                "provider": "meta",
                "baseUrl": "https://api.meta.ai/v1",
                "reasoning": model.get("reasoning") is True,
                "input": ["text", "image"] if _supports_image(model) else ["text"],
                "cost": _cost(model),
                **_limits(model),
            },
        )
        _record(reasoning_options_by_model, "meta", model_id, model)

    models.extend(process_zai_models(data, reasoning_options_by_model))

    # Mistral
    for model_id, model in _models(data, "mistral").items():
        if not _tool_callable(model):
            continue
        cost = model.get("cost") or {}
        cache_read = cost.get("cache_read")
        if cache_read is None:
            cache_read = round_cost((cost.get("input") or 0) * 0.1) if cost.get("input") else 0
        _append(
            models,
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "api": "mistral-conversations",
                "provider": "mistral",
                "baseUrl": "https://api.mistral.ai",
                "reasoning": model.get("reasoning") is True,
                "input": ["text", "image"] if _supports_image(model) else ["text"],
                "cost": {
                    "input": cost.get("input") or 0,
                    "output": cost.get("output") or 0,
                    "cacheRead": cache_read,
                    "cacheWrite": cost.get("cache_write") or 0,
                },
                **_limits(model),
            },
        )
        _record(reasoning_options_by_model, "mistral", model_id, model)

    # Hugging Face
    for model_id, model in _models(data, "huggingface").items():
        if not _tool_callable(model):
            continue
        _append(
            models,
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "api": "openai-completions",
                "provider": "huggingface",
                "baseUrl": "https://router.huggingface.co/v1",
                "reasoning": model.get("reasoning") is True,
                "input": ["text", "image"] if _supports_image(model) else ["text"],
                "cost": _cost(model),
                "compat": {"supportsDeveloperRole": False},
                **_limits(model),
            },
        )
        _record(reasoning_options_by_model, "huggingface", model_id, model)

    models.extend(process_fireworks_models(data.get("fireworks-ai"), reasoning_options_by_model))

    # NVIDIA NIM
    for model_id, model in _models(data, "nvidia").items():
        if not _tool_callable(model):
            continue
        modalities = model.get("modalities") or {}
        if "text" not in (modalities.get("input") or ()) or "text" not in (modalities.get("output") or ()):
            continue

        live_model_id = nvidia_nim_model_ids.get(model_id) or nvidia_nim_model_ids.get(
            normalize_nvidia_model_id(model_id)
        )
        if not live_model_id or live_model_id in NVIDIA_NIM_UNSUPPORTED_MODELS:
            continue

        _append(
            models,
            {
                "id": live_model_id,
                "name": model.get("name") or live_model_id,
                "api": "openai-completions",
                "provider": "nvidia",
                "baseUrl": NVIDIA_BASE_URL,
                "headers": dict(NVIDIA_HEADERS),
                "reasoning": model.get("reasoning") is True,
                "input": ["text", "image"] if _supports_image(model) else ["text"],
                "cost": _cost(model),
                "compat": dict(NVIDIA_OPENAI_COMPAT),
                **_limits(model),
            },
        )
        _record(reasoning_options_by_model, "nvidia", live_model_id, model)

    # Together AI
    together_provider = data.get("together") or data.get("togetherai") or data.get("together-ai")
    for model_id, model in ((together_provider or {}).get("models") or {}).items():
        if not _tool_callable(model) or not _not_deprecated(model):
            continue

        reasoning = model.get("reasoning") is True
        thinking_level_map = get_together_thinking_level_map(model_id, reasoning)
        _append(
            models,
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "api": "openai-completions",
                "provider": "together",
                "baseUrl": TOGETHER_BASE_URL,
                "reasoning": reasoning,
                **({"thinkingLevelMap": thinking_level_map} if thinking_level_map else {}),
                "input": ["text", "image"] if _supports_image(model) else ["text"],
                "cost": _cost(model),
                "compat": get_together_compat(model_id, reasoning),
                **_limits(model),
            },
        )
        _record(reasoning_options_by_model, "together", model_id, model)

    models.extend(process_baseten_models(data.get("baseten")))

    # OpenCode (Zen and Go). API mapping based on the provider.npm field:
    # - @ai-sdk/openai -> openai-responses
    # - @ai-sdk/anthropic -> anthropic-messages
    # - @ai-sdk/google -> google-generative-ai
    # - null/undefined/@ai-sdk/openai-compatible -> openai-completions
    opencode_variants = (
        ("opencode", "opencode", "https://opencode.ai/zen"),
        ("opencode-go", "opencode-go", "https://opencode.ai/zen/go"),
    )
    for key, provider, base_path in opencode_variants:
        provider_models = _models(data, key)
        if not provider_models:
            continue

        for model_id, model in provider_models.items():
            if not _tool_callable(model) or not _not_deprecated(model):
                continue

            npm = (model.get("provider") or {}).get("npm")
            compat: Optional[Dict[str, Any]] = None
            if npm == "@ai-sdk/openai":
                api = "openai-responses"
                base_url = f"{base_path}/v1"
                compat = {"sessionAffinityFormat": "openai-nosession"}
            elif npm == "@ai-sdk/anthropic":
                api = "anthropic-messages"
                # Anthropic SDK appends /v1/messages to baseURL
                base_url = base_path
            elif npm == "@ai-sdk/google":
                api = "google-generative-ai"
                base_url = f"{base_path}/v1"
            elif npm == "@ai-sdk/alibaba":
                api = "openai-completions"
                base_url = f"{base_path}/v1"
                compat = {"cacheControlFormat": "anthropic"}
            else:
                # null, undefined, or @ai-sdk/openai-compatible
                api = "openai-completions"
                base_url = f"{base_path}/v1"

            if provider == "opencode" and model_id == "grok-build-0.1":
                compat = {**(compat or {}), "supportsReasoningEffort": False}

            if provider in ("opencode", "opencode-go") and model_id == "kimi-k2.6":
                # OpenCode Kimi K2.6 accepts Anthropic-style thinking objects
                # and rejects string thinking values or combined reasoning_effort.
                compat = {**(compat or {}), "thinkingFormat": "deepseek", "supportsReasoningEffort": False}

            # Fix known mismatches between models.dev npm data and actual OpenCode Go
            # endpoint behaviour. models.dev reports these models as @ai-sdk/anthropic,
            # but the OpenCode Go endpoints either don't accept Anthropic SDK auth
            # (MiniMax M2.7) or are served through the OpenAI-compatible
            # /v1/chat/completions path (Qwen 3.5/3.6). Switch them to
            # openai-completions so requests use Bearer auth.
            if provider == "opencode-go":
                if model_id == "minimax-m2.7":
                    api = "openai-completions"
                    base_url = f"{base_path}/v1"
                if model_id in ("qwen3.5-plus", "qwen3.6-plus"):
                    api = "openai-completions"
                    base_url = f"{base_path}/v1"
                    # Qwen/DashScope uses enable_thinking at the top level.
                    compat = {**(compat or {}), "thinkingFormat": "qwen"}

            if api == "openai-completions":
                compat = {**(compat or {}), "maxTokensField": "max_tokens"}
                if f"{provider}:{model_id}" in OPENCODE_LONG_CACHE_RETENTION_UNSUPPORTED_MODELS:
                    compat = {**compat, "supportsLongCacheRetention": False}

            thinking_level_map: Optional[Dict[str, Optional[str]]] = None
            if api == "google-generative-ai":
                thinking_level_map = get_google_thinking_level_map(model_id, model.get("reasoning_options") or [])
            elif provider == "opencode-go" and model_id == "deepseek-v4.1-flash":
                thinking_level_map = get_effort_thinking_level_map(model.get("reasoning_options") or [])

            _append(
                models,
                {
                    "id": model_id,
                    "name": model.get("name") or model_id,
                    "api": api,
                    "provider": provider,
                    "baseUrl": base_url,
                    "reasoning": model.get("reasoning") is True,
                    **({"thinkingLevelMap": thinking_level_map} if thinking_level_map else {}),
                    "input": ["text", "image"] if _supports_image(model) else ["text"],
                    "cost": _cost(model),
                    **({"compat": compat} if compat else {}),
                    **_limits(model),
                },
            )
            _record(reasoning_options_by_model, provider, model_id, model)

    # GitHub Copilot
    for model_id, model in _models(data, "github-copilot").items():
        if not _tool_callable(model) or not _not_deprecated(model):
            continue

        # Claude 4.x and 5.x models route to Anthropic Messages API
        is_copilot_claude = re.match(r"^claude-(haiku|sonnet|opus|fable)-[45]([.\-]|$)", model_id) is not None
        # GPT, Grok, OSWE, and MAI-Code models are only served through
        # the Copilot /responses endpoint.
        needs_responses_api = model_id.startswith(("gpt-", "grok-", "oswe", "mai-"))

        if is_copilot_claude:
            api = "anthropic-messages"
        elif needs_responses_api:
            api = "openai-responses"
        else:
            api = "openai-completions"

        _append(
            models,
            {
                "id": model_id,
                "name": model.get("name") or model_id,
                "api": api,
                "provider": "github-copilot",
                "baseUrl": "https://api.individual.githubcopilot.com",
                "reasoning": model.get("reasoning") is True,
                "input": ["text", "image"] if _supports_image(model) else ["text"],
                "cost": get_models_dev_cost(model.get("cost")),
                "contextWindow": (model.get("limit") or {}).get("context") or 128000,
                "maxTokens": (model.get("limit") or {}).get("output") or 8192,
                "headers": dict(COPILOT_STATIC_HEADERS),
                # compat only applies to openai-completions
                **(
                    {
                        "compat": {
                            "supportsStore": False,
                            "supportsDeveloperRole": False,
                            "supportsReasoningEffort": False,
                        }
                    }
                    if api == "openai-completions"
                    else {}
                ),
            },
        )
        _record(reasoning_options_by_model, "github-copilot", model_id, model)

    # MiniMax
    for key, provider, base_url in (
        ("minimax", "minimax", "https://api.minimax.io/anthropic"),
        ("minimax-cn", "minimax-cn", "https://api.minimaxi.com/anthropic"),
    ):
        for model_id, model in _models(data, key).items():
            if not _tool_callable(model):
                continue
            _append(
                models,
                {
                    "id": model_id,
                    "name": model.get("name") or model_id,
                    "api": "anthropic-messages",
                    "provider": provider,
                    # MiniMax's Anthropic-compatible API - SDK appends /v1/messages
                    "baseUrl": base_url,
                    "reasoning": model.get("reasoning") is True,
                    "input": ["text", "image"] if _supports_image(model) else ["text"],
                    "cost": _cost(model),
                    **_limits(model),
                },
            )
            _record(reasoning_options_by_model, provider, model_id, model)

    # Kimi For Coding
    kimi_models = _models(data, "kimi-code-plan-global")
    if kimi_models:
        has_canonical_model = "kimi-for-coding" in kimi_models
        for model_id, model in kimi_models.items():
            if not _tool_callable(model):
                continue
            # models.dev may expose versioned aliases (e.g. k2p5/k2p6/k2p7).
            # Normalize aliases to the canonical model id and drop duplicates when canonical exists.
            if model_id in KIMI_ALIASES and has_canonical_model:
                continue

            normalized_id = "kimi-for-coding" if model_id in KIMI_ALIASES else model_id
            normalized_name = "Kimi For Coding" if model_id in KIMI_ALIASES else (model.get("name") or normalized_id)
            is_kimi_k3 = normalized_id == "k3"
            allow_empty_signature = is_kimi_k3 or normalized_id == "kimi-for-coding"
            implied_cost = KIMI_CODING_IMPLIED_COSTS.get(normalized_id) or {}
            cost = model.get("cost") or {}

            _append(
                models,
                {
                    "id": normalized_id,
                    "name": normalized_name,
                    "api": "anthropic-messages",
                    "provider": "kimi-coding",
                    # Kimi For Coding's Anthropic-compatible API - SDK appends /v1/messages
                    "baseUrl": "https://api.kimi.com/coding",
                    "compat": {
                        **({"allowEmptySignature": True} if allow_empty_signature else {}),
                        "forceAdaptiveThinking": True,
                    },
                    "reasoning": is_kimi_k3 or model.get("reasoning") is True,
                    "input": ["text", "image"] if _supports_image(model) else ["text"],
                    "cost": {
                        "input": cost.get("input") or implied_cost.get("input") or 0,
                        "output": cost.get("output") or implied_cost.get("output") or 0,
                        "cacheRead": cost.get("cache_read") or implied_cost.get("cacheRead") or 0,
                        "cacheWrite": cost.get("cache_write") or implied_cost.get("cacheWrite") or 0,
                    },
                    **_limits(model),
                },
            )
            _record(reasoning_options_by_model, "kimi-coding", normalized_id, model)

    # Moonshot AI
    for key, provider, base_url in (
        ("moonshotai", "moonshotai", "https://api.moonshot.ai/v1"),
        ("moonshotai-cn", "moonshotai-cn", "https://api.moonshot.cn/v1"),
    ):
        for model_id, model in _models(data, key).items():
            if not _tool_callable(model):
                continue

            is_kimi_k3 = model_id == "kimi-k3"
            compat = dict(MOONSHOT_COMPAT)
            if is_kimi_k3:
                compat["requiresReasoningContentOnAssistantMessages"] = True
                compat["thinkingFormat"] = "openai"
                compat["supportsReasoningEffort"] = True
            cost = model.get("cost") or {}
            _append(
                models,
                {
                    "id": model_id,
                    "name": model.get("name") or model_id,
                    "api": "openai-completions",
                    "provider": provider,
                    "baseUrl": base_url,
                    "reasoning": is_kimi_k3 or model.get("reasoning") is True,
                    "input": ["text", "image"] if _supports_image(model) else ["text"],
                    "cost": {
                        "input": cost.get("input") or (KIMI_K3_COST["input"] if is_kimi_k3 else 0),
                        "output": cost.get("output") or (KIMI_K3_COST["output"] if is_kimi_k3 else 0),
                        "cacheRead": cost.get("cache_read") or (KIMI_K3_COST["cacheRead"] if is_kimi_k3 else 0),
                        "cacheWrite": cost.get("cache_write") or (KIMI_K3_COST["cacheWrite"] if is_kimi_k3 else 0),
                    },
                    **_limits(model),
                    "compat": compat,
                },
            )
            _record(reasoning_options_by_model, provider, model_id, model)

    # Xiaomi MiMo
    # Built-in `xiaomi` targets the API billing endpoint (single stable URL,
    # keys from platform.xiaomimimo.com). The three `xiaomi-token-plan-*`
    # providers cover prepaid Token Plan endpoints in cn / ams / sgp.
    for source, provider, base_url in (
        ("xiaomi", "xiaomi", "https://api.xiaomimimo.com/v1"),
        ("xiaomi-token-plan-cn", "xiaomi-token-plan-cn", "https://token-plan-cn.xiaomimimo.com/v1"),
        ("xiaomi-token-plan-ams", "xiaomi-token-plan-ams", "https://token-plan-ams.xiaomimimo.com/v1"),
        ("xiaomi-token-plan-sgp", "xiaomi-token-plan-sgp", "https://token-plan-sgp.xiaomimimo.com/v1"),
    ):
        for model_id, model in _models(data, source).items():
            if not _tool_callable(model) or not _not_deprecated(model):
                continue
            _append(
                models,
                {
                    "id": model_id,
                    "name": model.get("name") or model_id,
                    "api": "openai-completions",
                    "provider": provider,
                    "baseUrl": base_url,
                    "compat": dict(XIAOMI_COMPAT),
                    "reasoning": model.get("reasoning") is True,
                    "input": ["text", "image"] if _supports_image(model) else ["text"],
                    "cost": _cost(model),
                    **_limits(model),
                },
            )
            _record(reasoning_options_by_model, provider, model_id, model)

    # Alibaba Cloud Model Studio Token Plan models. International and China use
    # separate endpoints and API keys (sk-sp- prefix). The Individual provider
    # reuses the international source and endpoint with a narrower catalog.
    # models.dev keys are "alibaba-token-plan[-cn]"; pi exposes them as
    # "qwen-token-plan[-cn]" plus the Individual catalog view.
    qwen_token_plan_variants = (
        ("alibaba-token-plan", "qwen-token-plan", "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", None),
        (
            "alibaba-token-plan",
            "qwen-token-plan-individual",
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            QWEN_TOKEN_PLAN_INDIVIDUAL_MODEL_IDS,
        ),
        ("alibaba-token-plan-cn", "qwen-token-plan-cn", "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", None),
    )
    for source, provider, base_url, model_ids in qwen_token_plan_variants:
        emitted_model_ids: set = set()
        for model_id, model in _models(data, source).items():
            if not _tool_callable(model) or model_id in QWEN_TOKEN_PLAN_EXCLUDED_MODEL_IDS:
                continue
            if model_ids and model_id not in model_ids:
                continue
            thinking_level_map = get_effort_thinking_level_map(model.get("reasoning_options") or []) or (
                dict(QWEN_TOKEN_PLAN_FALLBACK_THINKING_LEVEL_MAP)
                if model_id in QWEN_TOKEN_PLAN_REASONING_EFFORT_FALLBACK_MODEL_IDS
                else None
            )

            _append(
                models,
                {
                    "id": model_id,
                    "name": model.get("name") or model_id,
                    "api": "openai-completions",
                    "provider": provider,
                    "baseUrl": base_url,
                    "compat": (
                        dict(QWEN_TOKEN_PLAN_COMPAT)
                        if thinking_level_map
                        else {**QWEN_TOKEN_PLAN_COMPAT, "supportsReasoningEffort": False}
                    ),
                    **({"thinkingLevelMap": thinking_level_map} if thinking_level_map else {}),
                    "reasoning": model.get("reasoning") is True,
                    "input": ["text", "image"] if _supports_image(model) else ["text"],
                    "cost": _cost(model),
                    **_limits(model),
                },
            )
            emitted_model_ids.add(model_id)

        if model_ids and strict:
            assert_exact_model_ids(provider, model_ids, emitted_model_ids)

    print(f"Loaded {len(models)} tool-capable models from models.dev")
    return models


def load_models_dev(strict: bool = False) -> ModelsDevCatalog:
    """Fetch the full models.dev catalog."""
    print("Fetching models from models.dev API...")
    return get_json("https://models.dev/api.json")


__all__ = [
    "get_models_dev_cost",
    "load_models_dev",
    "load_models_dev_classifier_models",
    "load_models_dev_data",
    "process_baseten_models",
    "process_fireworks_models",
    "process_google_models",
    "process_zai_models",
]

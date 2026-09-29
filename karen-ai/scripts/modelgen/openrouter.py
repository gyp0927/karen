"""OpenRouter listing -> chat/image/classifier catalog.

Port of pi-ai's `scripts/openrouter-catalog.ts` and
`scripts/openrouter-reasoning-options.ts`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .reasoning_options import get_effort_thinking_level_map
from .tables import round_cost

OpenRouterModelListItem = Mapping[str, Any]


def get_openrouter_thinking_level_map(reasoning: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Optional[str]]]:
    """Convert OpenRouter's reasoning metadata into Pi model capabilities."""
    if not reasoning:
        return None
    supported_efforts = reasoning.get("supported_efforts")
    mandatory = reasoning.get("mandatory") is True
    if not supported_efforts:
        return {"off": None} if mandatory else None

    # OpenRouter's supported_efforts uses the same effort values as models.dev
    # reasoning_options, so both sources can share the same conversion.
    level_map = get_effort_thinking_level_map([{"type": "effort", "values": supported_efforts}])
    if not level_map:
        return {"off": None} if mandatory else None
    return {**level_map, "off": None if mandatory else "none"}


def _modalities(values: Optional[Sequence[str]]) -> List[str]:
    seen: List[str] = []
    for value in values or ():
        if value in ("text", "image") and value not in seen:
            seen.append(value)
    return seen


def _cost(model: OpenRouterModelListItem) -> Dict[str, float]:
    """Convert pricing from $/token to $/million tokens."""
    pricing = model.get("pricing") or {}

    def rate(key: str) -> float:
        raw = pricing.get(key) or "0"
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    return {
        "input": round_cost(rate("prompt") * 1_000_000),
        "output": round_cost(rate("completion") * 1_000_000),
        "cacheRead": round_cost(rate("input_cache_read") * 1_000_000),
        "cacheWrite": round_cost(rate("input_cache_write") * 1_000_000),
    }


def build_openrouter_catalog(
    listed: Sequence[OpenRouterModelListItem],
    image_listed: Sequence[OpenRouterModelListItem],
    decision_listed: Sequence[OpenRouterModelListItem],
) -> Dict[str, List[Dict[str, Any]]]:
    """Build the OpenRouter catalog from the three upstream listings.

    The default listing omits image-only and decision models, so those come from
    the `output_modalities=image` and `output_modalities=decisions` listings. An
    upstream model may appear in several results; it then gets separate entries
    per operation.
    """
    chat: List[Dict[str, Any]] = []
    for model in listed:
        # Only include models that support tools.
        if "tools" not in (model.get("supported_parameters") or ()):
            continue
        input_modalities: List[str] = ["text"]
        if "image" in ((model.get("architecture") or {}).get("modality") or ""):
            input_modalities.append("image")

        model_id = model.get("id", "")
        thinking_level_map = get_openrouter_thinking_level_map(model.get("reasoning"))
        use_anthropic_messages = model_id.startswith("anthropic/") and not model_id.endswith(":batch")
        top_provider = model.get("top_provider") or {}
        chat.append(
            {
                "type": "chat",
                "id": model_id,
                "name": model.get("name", model_id),
                "api": "anthropic-messages" if use_anthropic_messages else "openai-completions",
                "baseUrl": "https://openrouter.ai/api" if use_anthropic_messages else "https://openrouter.ai/api/v1",
                "provider": "openrouter",
                "reasoning": "reasoning" in (model.get("supported_parameters") or ()),
                **({"thinkingLevelMap": thinking_level_map} if thinking_level_map else {}),
                "input": input_modalities,
                "cost": _cost(model),
                "contextWindow": top_provider.get("context_length") or model.get("context_length") or 4096,
                "maxTokens": top_provider.get("max_completion_tokens") or 4096,
            }
        )

    images: List[Dict[str, Any]] = []
    for model in image_listed:
        model_id = model.get("id", "")
        if any(entry["id"] == model_id for entry in images):
            continue
        architecture = model.get("architecture") or {}
        output = _modalities(architecture.get("output_modalities"))
        if "image" not in output:
            continue
        input_modalities = _modalities(architecture.get("input_modalities"))
        images.append(
            {
                "type": "image",
                "id": model_id,
                "name": model.get("name", model_id),
                "api": "openrouter-images",
                "provider": "openrouter",
                "baseUrl": "https://openrouter.ai/api/v1",
                "input": input_modalities or ["text"],
                "output": output,
                "cost": _cost(model),
            }
        )

    # Decision models such as TypeSafe's Jev are served through OpenRouter's
    # TypeSafe-compatible System One endpoint.
    classifiers: List[Dict[str, Any]] = []
    for model in decision_listed:
        model_id = model.get("id", "")
        if any(entry["id"] == model_id for entry in classifiers):
            continue
        architecture = model.get("architecture") or {}
        if "decisions" not in (architecture.get("output_modalities") or ()):
            continue
        input_modalities = _modalities(architecture.get("input_modalities"))
        top_provider = model.get("top_provider") or {}
        classifiers.append(
            {
                "type": "classifier",
                "id": model_id,
                "name": model.get("name", model_id),
                "api": "typesafe-system-one",
                "provider": "openrouter",
                "baseUrl": "https://openrouter.ai/api/v1",
                "input": input_modalities or ["text"],
                "cost": _cost(model),
                "contextWindow": top_provider.get("context_length") or model.get("context_length") or 4096,
            }
        )

    return {"chat": chat, "images": images, "classifiers": classifiers}


__all__ = ["build_openrouter_catalog", "get_openrouter_thinking_level_map"]

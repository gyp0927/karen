"""Upstream catalog fetchers.

Port of the `fetch*` helpers in pi-ai's `scripts/generate-models.ts`.

Every fetcher degrades to an empty result unless `--strict` was passed, which
turns a failed upstream into a hard error instead.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Mapping, Optional

import httpx

from .openrouter import build_openrouter_catalog
from .tables import (
    AI_GATEWAY_BASE_URL,
    AI_GATEWAY_MODELS_URL,
    NVIDIA_BASE_URL,
    normalize_nvidia_model_id,
    round_cost,
)

#: models.dev rejects requests without a browser-like UA.
USER_AGENT = "karen-ai-modelgen/1.0 (+https://github.com/karen-ai)"
TIMEOUT_SECONDS = 60.0


def _headers(extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if extra:
        headers.update(extra)
    return headers


def get_json(url: str, extra_headers: Optional[Mapping[str, str]] = None) -> Any:
    with httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
        response = client.get(url, headers=_headers(extra_headers))
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Nvidia NIM
# ---------------------------------------------------------------------------


def fetch_nvidia_nim_model_ids(strict: bool = False) -> Dict[str, str]:
    """Live Nvidia NIM ids, keyed by both the raw and normalized spelling."""
    try:
        print("Fetching models from NVIDIA NIM API...")
        data = get_json(f"{NVIDIA_BASE_URL}/models")
        model_ids: Dict[str, str] = {}
        for model in data.get("data") or ():
            model_id = model["id"]
            model_ids[model_id] = model_id
            model_ids[normalize_nvidia_model_id(model_id)] = model_id
        print(f"Fetched {len(data.get('data') or ())} model IDs from NVIDIA NIM")
        return model_ids
    except Exception as error:  # noqa: BLE001 - mirror pi-ai's catch-all
        print(f"Failed to fetch NVIDIA NIM models: {error}")
        if strict:
            raise
        return {}


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------


def fetch_openrouter_list(query: str) -> List[Dict[str, Any]]:
    data = get_json(f"https://openrouter.ai/api/v1/models{query}")
    return data.get("data") or []


def fetch_openrouter_models(strict: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    try:
        print("Fetching models from OpenRouter API...")
        listed = fetch_openrouter_list("")
        image_listed = fetch_openrouter_list("?output_modalities=image")
        decision_listed = fetch_openrouter_list("?output_modalities=decisions")
        catalog = build_openrouter_catalog(listed, image_listed, decision_listed)
        print(
            f"Fetched {len(catalog['chat'])} tool-capable, {len(catalog['images'])} image, "
            f"and {len(catalog['classifiers'])} classifier models from OpenRouter"
        )
        if strict and not catalog["images"]:
            raise RuntimeError("OpenRouter API returned no usable image models")
        return catalog
    except Exception as error:  # noqa: BLE001 - mirror pi-ai's catch-all
        print(f"Failed to fetch OpenRouter models: {error}")
        if strict:
            raise
        return {"chat": [], "images": [], "classifiers": []}


# ---------------------------------------------------------------------------
# Vercel AI Gateway
# ---------------------------------------------------------------------------


def _to_number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value) if value == value else 0.0
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def fetch_ai_gateway_models(strict: bool = False) -> List[Dict[str, Any]]:
    """Tool-capable models from the Vercel AI Gateway, priced per million tokens."""
    try:
        print("Fetching models from Vercel AI Gateway API...")
        data = get_json(f"{AI_GATEWAY_MODELS_URL}/models")
        models: List[Dict[str, Any]] = []
        for model in data.get("data") if isinstance(data.get("data"), list) else []:
            tags = model.get("tags") if isinstance(model.get("tags"), list) else []
            # Only include models that support tools.
            if "tool-use" not in tags:
                continue

            input_modalities = ["text"]
            if "vision" in tags:
                input_modalities.append("image")

            pricing = model.get("pricing") or {}
            models.append(
                {
                    "id": model["id"],
                    "name": model.get("name") or model["id"],
                    "api": "anthropic-messages",
                    "baseUrl": AI_GATEWAY_BASE_URL,
                    "provider": "vercel-ai-gateway",
                    "reasoning": "reasoning" in tags,
                    "input": input_modalities,
                    "compat": {"allowEmptySignature": True},
                    "cost": {
                        "input": round_cost(_to_number(pricing.get("input")) * 1_000_000),
                        "output": round_cost(_to_number(pricing.get("output")) * 1_000_000),
                        "cacheRead": round_cost(_to_number(pricing.get("input_cache_read")) * 1_000_000),
                        "cacheWrite": round_cost(_to_number(pricing.get("input_cache_write")) * 1_000_000),
                    },
                    "contextWindow": model.get("context_window") or 4096,
                    "maxTokens": model.get("max_tokens") or 4096,
                }
            )

        print(f"Fetched {len(models)} tool-capable models from Vercel AI Gateway")
        return models
    except Exception as error:  # noqa: BLE001 - mirror pi-ai's catch-all
        print(f"Failed to fetch Vercel AI Gateway models: {error}")
        if strict:
            raise
        return []


# ---------------------------------------------------------------------------
# Radius
# ---------------------------------------------------------------------------


def fetch_radius_models(strict: bool = False) -> List[Dict[str, Any]]:
    """Radius's unauthenticated public catalog."""
    try:
        print("Fetching models from Radius API...")
        from karen_ai.providers.radius_config import (
            DEFAULT_RADIUS_GATEWAY,
            get_radius_models_from_config,
            load_radius_gateway_config,
        )

        config = asyncio.run(load_radius_gateway_config(DEFAULT_RADIUS_GATEWAY))
        models = get_radius_models_from_config("radius", config)
        if not models:
            raise RuntimeError("Radius API returned no models")
        print(f"Fetched {len(models)} models from Radius")
        return [json.loads(model.model_dump_json(by_alias=True, exclude_none=True)) for model in models]
    except Exception as error:  # noqa: BLE001 - mirror pi-ai's catch-all
        print(f"Failed to fetch Radius models: {error}")
        if strict:
            raise
        return []


__all__ = [
    "fetch_ai_gateway_models",
    "fetch_nvidia_nim_model_ids",
    "fetch_openrouter_list",
    "fetch_openrouter_models",
    "fetch_radius_models",
    "get_json",
]

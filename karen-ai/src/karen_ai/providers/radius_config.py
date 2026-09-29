"""Radius gateway config, mirroring providers/radius-config.ts."""

from __future__ import annotations

from typing import Any, List, Optional

import httpx

from ..abort import AbortSignal
from ..auth.types import OAuthCredential
from ..types import Model

DEFAULT_RADIUS_GATEWAY = "https://radius.pi.dev"


def _is_gateway_model(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("id"), str)
        and isinstance(value.get("name"), str)
        and isinstance(value.get("reasoning"), bool)
        and isinstance(value.get("input"), list)
        and isinstance(value.get("cost"), dict)
        and isinstance(value.get("contextWindow"), (int, float))
        and isinstance(value.get("maxTokens"), (int, float))
    )


def _sanitize_gateway_config(config: Any) -> Optional[dict]:
    if not isinstance(config, dict):
        return None
    base_url = config.get("baseUrl")
    models = config.get("models")
    if not isinstance(base_url, str) or not isinstance(models, list):
        return None
    return {"baseUrl": base_url, "models": [dict(m) for m in models if _is_gateway_model(m)]}


def normalize_radius_gateway_url(value: str) -> str:
    with_scheme = value if value.lower().startswith(("http://", "https://")) else f"https://{value}"
    return with_scheme.rstrip("/")


def get_radius_credential_config(credential: Optional[OAuthCredential]) -> Optional[dict]:
    if credential is None:
        return None
    return _sanitize_gateway_config((credential.model_extra or {}).get("gatewayConfig"))


def get_radius_models_from_config(provider_id: str, config: dict) -> List[Model]:
    return [
        Model.model_validate({**model, "api": "pi-messages", "provider": provider_id, "baseUrl": config["baseUrl"]})
        for model in config["models"]
    ]


def get_radius_models(provider_id: str, credential: Optional[OAuthCredential]) -> List[Model]:
    config = get_radius_credential_config(credential)
    return get_radius_models_from_config(provider_id, config) if config else []


def _truncate_http_body(body: str) -> str:
    trimmed = body.strip()
    return f"{trimmed[:512]}…" if len(trimmed) > 512 else trimmed


async def load_radius_gateway_config(
    gateway: str,
    api_key: Optional[str] = None,
    signal: Optional[AbortSignal] = None,
) -> dict:
    headers = {"accept": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{gateway}/v1/config", headers=headers)
    if response.status_code >= 400:
        raise ValueError(
            f"Could not load Radius config from {gateway}: "
            f"{response.status_code}: {_truncate_http_body(response.text)}"
        )
    config = _sanitize_gateway_config(response.json())
    if not config:
        raise ValueError(f"Invalid Radius config from {gateway}")
    return config

"""Environment API-key discovery, mirroring pi-ai's `env-api-keys.ts`.

Used by the deprecated ambient dispatch surface in `karen_ai.compat`; provider
factories resolve credentials through the auth system instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from .utils.provider_env import get_provider_env_value

ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_OAUTH_TOKEN_ENV = "ANTHROPIC_OAUTH_TOKEN"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

#: Sentinel returned when ambient cloud credentials (not an API key) are configured.
AMBIENT_AUTH_MARKER = "<authenticated>"

API_KEY_ENV_VARS: Dict[str, str] = {
    "ant-ling": "ANT_LING_API_KEY",
    "qwen-token-plan": "QWEN_TOKEN_PLAN_API_KEY",
    "qwen-token-plan-cn": "QWEN_TOKEN_PLAN_CN_API_KEY",
    "qwen-token-plan-individual": "QWEN_TOKEN_PLAN_API_KEY",
    "openai": "OPENAI_API_KEY",
    "azure-openai-responses": "AZURE_OPENAI_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GEMINI_API_KEY",
    "google-vertex": "GOOGLE_CLOUD_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "xai": "XAI_API_KEY",
    "typesafe": "TYPESAFE_API_KEY",
    "radius": "RADIUS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "vercel-ai-gateway": "AI_GATEWAY_API_KEY",
    "zai": "ZAI_API_KEY",
    "zai-coding-cn": "ZAI_CODING_CN_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
    "moonshotai": "MOONSHOT_API_KEY",
    "moonshotai-cn": "MOONSHOT_API_KEY",
    "huggingface": "HF_TOKEN",
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "opencode": "OPENCODE_API_KEY",
    "opencode-go": "OPENCODE_API_KEY",
    "kimi-coding": "KIMI_API_KEY",
    "meta": "META_API_KEY",
    "cloudflare-workers-ai": "CLOUDFLARE_API_KEY",
    "cloudflare-ai-gateway": "CLOUDFLARE_API_KEY",
    "xiaomi": "XIAOMI_API_KEY",
    "xiaomi-token-plan-cn": "XIAOMI_TOKEN_PLAN_CN_API_KEY",
    "xiaomi-token-plan-ams": "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
    "xiaomi-token-plan-sgp": "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
}

_vertex_adc_credentials_exists: Optional[bool] = None


def get_api_key_env_vars(provider: str) -> Optional[List[str]]:
    """Environment variables that can carry an API key for `provider`."""
    if provider == "github-copilot":
        return ["COPILOT_GITHUB_TOKEN"]

    if provider == "anthropic":
        # ANTHROPIC_AUTH_TOKEN participates in env discovery/status, but
        # get_env_api_key() skips it: requests must send it as a bearer token.
        return [ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV]

    env_var = API_KEY_ENV_VARS.get(provider)
    return [env_var] if env_var else None


def find_env_keys(provider: str, env: Optional[Dict[str, str]] = None) -> Optional[List[str]]:
    """Configured API-key environment variables for `provider`, if any.

    Excludes ambient credential sources such as AWS profiles and Google ADC.
    """
    env_vars = get_api_key_env_vars(provider)
    if not env_vars:
        return None
    found = [name for name in env_vars if get_provider_env_value(name, env)]
    return found or None


def has_vertex_adc_credentials(env: Optional[Dict[str, str]] = None) -> bool:
    """Whether Google Application Default Credentials are present on disk."""
    global _vertex_adc_credentials_exists

    explicit_path = env.get("GOOGLE_APPLICATION_CREDENTIALS") if env else None
    if explicit_path:
        return os.path.exists(explicit_path)

    if _vertex_adc_credentials_exists is None:
        default_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        _vertex_adc_credentials_exists = default_path.is_file()
    return _vertex_adc_credentials_exists


def get_env_api_key(provider: str, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """API key for `provider` from the known environment variables.

    Never returns OAuth tokens, and reports ambient cloud credentials as the
    `"<authenticated>"` marker.
    """
    env_keys = find_env_keys(provider, env)
    if env_keys:
        api_key_env = (
            next((key for key in env_keys if key != ANTHROPIC_AUTH_TOKEN_ENV), None)
            if provider == "anthropic"
            else env_keys[0]
        )
        if api_key_env:
            return get_provider_env_value(api_key_env, env)

    # Vertex AI accepts an explicit API key or Application Default Credentials
    # (`gcloud auth application-default login`).
    if provider == "google-vertex":
        has_credentials = has_vertex_adc_credentials(env)
        has_project = bool(get_provider_env_value("GOOGLE_CLOUD_PROJECT", env) or get_provider_env_value("GCLOUD_PROJECT", env))
        has_location = bool(get_provider_env_value("GOOGLE_CLOUD_LOCATION", env))
        if has_credentials and has_project and has_location:
            return AMBIENT_AUTH_MARKER

    if provider == "amazon-bedrock":
        # AWS_PROFILE, IAM keys, Bedrock bearer token, ECS task roles, or IRSA.
        if (
            get_provider_env_value("AWS_PROFILE", env)
            or (
                get_provider_env_value("AWS_ACCESS_KEY_ID", env)
                and get_provider_env_value("AWS_SECRET_ACCESS_KEY", env)
            )
            or get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", env)
            or get_provider_env_value("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", env)
            or get_provider_env_value("AWS_CONTAINER_CREDENTIALS_FULL_URI", env)
            or get_provider_env_value("AWS_WEB_IDENTITY_TOKEN_FILE", env)
        ):
            return AMBIENT_AUTH_MARKER

    return None


__all__ = [
    "AMBIENT_AUTH_MARKER",
    "ANTHROPIC_API_KEY_ENV",
    "ANTHROPIC_AUTH_TOKEN_ENV",
    "ANTHROPIC_OAUTH_TOKEN_ENV",
    "API_KEY_ENV_VARS",
    "find_env_keys",
    "get_api_key_env_vars",
    "get_env_api_key",
    "has_vertex_adc_credentials",
]

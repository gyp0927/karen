"""Qwen Token Plan provider, mirroring pi-ai's providers/qwen-token-plan.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def qwen_token_plan_provider():
    return catalog_provider(
        id="qwen-token-plan",
        name="Qwen Token Plan",
        base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        key_name="Qwen Token Plan API key",
        env_vars=['QWEN_TOKEN_PLAN_API_KEY'],
        api=openai_completions_api(),
    )

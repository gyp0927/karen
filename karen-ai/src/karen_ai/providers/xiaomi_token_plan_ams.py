"""Xiaomi Token Plan AMS provider, mirroring pi-ai's providers/xiaomi-token-plan-ams.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def xiaomi_token_plan_ams_provider():
    return catalog_provider(
        id="xiaomi-token-plan-ams",
        name="Xiaomi Token Plan AMS",
        base_url="https://token-plan-ams.xiaomimimo.com/v1",
        key_name="Xiaomi Token Plan AMS API key",
        env_vars=['XIAOMI_TOKEN_PLAN_AMS_API_KEY'],
        api=openai_completions_api(),
    )

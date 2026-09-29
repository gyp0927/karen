"""Xiaomi Token Plan SGP provider, mirroring pi-ai's providers/xiaomi-token-plan-sgp.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def xiaomi_token_plan_sgp_provider():
    return catalog_provider(
        id="xiaomi-token-plan-sgp",
        name="Xiaomi Token Plan SGP",
        base_url="https://token-plan-sgp.xiaomimimo.com/v1",
        key_name="Xiaomi Token Plan SGP API key",
        env_vars=['XIAOMI_TOKEN_PLAN_SGP_API_KEY'],
        api=openai_completions_api(),
    )

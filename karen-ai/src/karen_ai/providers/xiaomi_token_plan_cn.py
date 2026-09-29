"""Xiaomi Token Plan CN provider, mirroring pi-ai's providers/xiaomi-token-plan-cn.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def xiaomi_token_plan_cn_provider():
    return catalog_provider(
        id="xiaomi-token-plan-cn",
        name="Xiaomi Token Plan CN",
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        key_name="Xiaomi Token Plan CN API key",
        env_vars=['XIAOMI_TOKEN_PLAN_CN_API_KEY'],
        api=openai_completions_api(),
    )

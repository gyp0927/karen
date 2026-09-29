"""Qwen Token Plan CN provider, mirroring pi-ai's providers/qwen-token-plan-cn.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def qwen_token_plan_cn_provider():
    return catalog_provider(
        id="qwen-token-plan-cn",
        name="Qwen Token Plan CN",
        base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        key_name="Qwen Token Plan CN API key",
        env_vars=['QWEN_TOKEN_PLAN_CN_API_KEY'],
        api=openai_completions_api(),
    )

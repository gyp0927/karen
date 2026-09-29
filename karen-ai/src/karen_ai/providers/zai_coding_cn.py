"""Z.AI Coding CN provider, mirroring pi-ai's providers/zai-coding-cn.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def zai_coding_cn_provider():
    return catalog_provider(
        id="zai-coding-cn",
        name="Z.AI Coding CN",
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
        key_name="Z.AI Coding CN API key",
        env_vars=['ZAI_CODING_CN_API_KEY'],
        api=openai_completions_api(),
    )

"""Hugging Face provider, mirroring pi-ai's providers/huggingface.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def huggingface_provider():
    return catalog_provider(
        id="huggingface",
        name="Hugging Face",
        base_url="https://router.huggingface.co/v1",
        key_name="Hugging Face token",
        env_vars=['HF_TOKEN'],
        api=openai_completions_api(),
    )

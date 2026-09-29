"""NVIDIA provider, mirroring pi-ai's providers/nvidia.ts."""

from __future__ import annotations

from ..api import openai_completions_api
from ._catalog import catalog_provider


def nvidia_provider():
    return catalog_provider(
        id="nvidia",
        name="NVIDIA",
        base_url="https://integrate.api.nvidia.com/v1",
        key_name="NVIDIA API key",
        env_vars=['NVIDIA_API_KEY'],
        api=openai_completions_api(),
    )

"""Azure OpenAI (Responses API) provider, mirroring azure-openai-responses.ts.

Models carry no base URL of their own; the Azure resource/deployment is resolved
per request from env vars or options by the api module.
"""

from __future__ import annotations

from ..api import azure_openai_responses_api
from ._catalog import catalog_provider


def azure_openai_responses_provider():
    return catalog_provider(
        id="azure-openai-responses",
        name="Azure OpenAI",
        key_name="Azure OpenAI API key",
        env_vars=["AZURE_OPENAI_API_KEY"],
        api=azure_openai_responses_api(),
    )

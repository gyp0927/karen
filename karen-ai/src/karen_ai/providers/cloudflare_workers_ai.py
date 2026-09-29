"""Cloudflare Workers AI provider, mirroring cloudflare-workers-ai.ts."""

from __future__ import annotations

from ..api import cloudflare_workers_ai_system_one, openai_completions_api
from ..auth.types import ProviderAuth
from ..model_catalog import flatten_chat_model_catalog, flatten_classifier_model_catalog
from ..models import CreateProviderOptions, create_provider
from .cloudflare_auth import cloudflare_workers_ai_auth
from .cloudflare_stream import cloudflare_classifier, cloudflare_streams


def cloudflare_workers_ai_provider():
    models = list(flatten_chat_model_catalog("cloudflare-workers-ai").values())
    models += list(flatten_classifier_model_catalog("cloudflare-workers-ai").values())
    return create_provider(
        CreateProviderOptions(
            id="cloudflare-workers-ai",
            name="Cloudflare Workers AI",
            auth=ProviderAuth(api_key=cloudflare_workers_ai_auth()),
            models=models,
            api=cloudflare_streams(openai_completions_api()),
            classifiers={
                "cloudflare-workers-ai-system-one": cloudflare_classifier(
                    cloudflare_workers_ai_system_one.classify
                )
            },
        )
    )

"""Cloudflare Workers AI System One classification.

Mirrors api/cloudflare-workers-ai-system-one.ts:
`POST {base}/run` with `{model, input}`. The REST API wraps the model output
in Cloudflare's API envelope and a run record:
`{success, result: {state: "Completed", result: {answers, usage}}}`.
https://developers.cloudflare.com/ai/models/typesafe/jev/
"""

from __future__ import annotations

from typing import Optional

from ..types import ClassifierContext, ClassifierModel, ClassifierResult, ProviderRequestOptions
from .system_one_shared import SystemOneTransport, classify_system_one, is_record

LABEL = "Cloudflare Workers AI"


def _cloudflare_error_message(errors) -> str:
    if isinstance(errors, list):
        messages = [
            error["message"]
            for error in errors
            if is_record(error) and isinstance(error.get("message"), str)
        ]
        if messages:
            return f"{LABEL} error: {'; '.join(messages)}"
    return f"{LABEL} request failed"


def _url(model: ClassifierModel) -> str:
    return f"{model.base_url.rstrip('/')}/run"


def _answers(body):
    if not is_record(body):
        raise ValueError(f"{LABEL} returned an unexpected response")
    if body.get("success") is False:
        raise ValueError(_cloudflare_error_message(body.get("errors")))
    run = body.get("result")
    if not is_record(run):
        raise ValueError(f"{LABEL} returned an unexpected response")
    if run.get("state") != "Completed":
        raise ValueError(f"{LABEL} run did not complete (state: {run.get('state')})")
    if not is_record(run.get("result")):
        raise ValueError(f"{LABEL} returned an unexpected response")
    return run["result"].get("answers")


TRANSPORT = SystemOneTransport(
    api="cloudflare-workers-ai-system-one",
    label=LABEL,
    url=_url,
    payload=lambda model, request: {"model": model.id, "input": request},
    answers=_answers,
)


async def classify(
    model: ClassifierModel,
    context: ClassifierContext,
    options: Optional[ProviderRequestOptions] = None,
) -> ClassifierResult:
    """Cloudflare Workers AI System One classification."""
    return await classify_system_one(TRANSPORT, model, context, options)

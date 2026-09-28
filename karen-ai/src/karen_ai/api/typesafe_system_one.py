"""TypeSafe System One classification, mirroring api/typesafe-system-one.ts.

TypeSafe's native System One protocol. OpenRouter serves the same protocol,
so both providers use this API with different base URLs.
"""

from __future__ import annotations

from typing import Optional

from ..types import ClassifierContext, ClassifierModel, ClassifierResult, ProviderRequestOptions
from .system_one_shared import SystemOneTransport, classify_system_one, is_record


def _url(model: ClassifierModel) -> str:
    return f"{model.base_url.rstrip('/')}/systemone"


def _answers(body):
    if not is_record(body):
        raise ValueError("System One API returned an unexpected response")
    return body.get("answers")


TRANSPORT = SystemOneTransport(
    api="typesafe-system-one",
    label="System One API",
    url=_url,
    payload=lambda model, request: {"model": model.id, **request},
    answers=_answers,
)


async def classify(
    model: ClassifierModel,
    context: ClassifierContext,
    options: Optional[ProviderRequestOptions] = None,
) -> ClassifierResult:
    """TypeSafe System One classification with public `bool` values mapped to wire-level `noul`."""
    return await classify_system_one(TRANSPORT, model, context, options)

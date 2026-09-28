"""Shared System One classification machinery, mirroring api/system-one-shared.ts.

TypeSafe System One serves the same protocol at TypeSafe's own endpoint, at
OpenRouter, and on Cloudflare Workers AI; transports differ only in URL,
request envelope, and response envelope.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import httpx

from ..types import (
    ClassifierAnswer,
    ClassifierBoolAnswer,
    ClassifierChoiceAnswer,
    ClassifierContext,
    ClassifierModel,
    ClassifierScoreAnswer,
    ProviderHeaders,
    ProviderRequestOptions,
    ProviderResponse,
    ClassifierResult,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.retry import ProviderHttpError, retry_provider_request


@dataclass
class SystemOneTransport:
    """Differences between services that serve System One models."""

    #: Classifier API implemented by this transport.
    api: str
    #: Service name used in error messages.
    label: str
    #: Absolute request URL for a model.
    url: Callable[[ClassifierModel], str]
    #: Wraps the System One request in the service's request envelope.
    payload: Callable[[ClassifierModel, Dict[str, Any]], Any]
    #: Extracts the System One `answers` object from the response envelope.
    answers: Callable[[Any], Any]


def is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _required_number(label: str, value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} returned an invalid {field}")
    return value


def _probabilities(label: str, value: Any, question_id: str) -> Dict[str, float]:
    if not is_record(value):
        raise ValueError(f"{label} returned invalid probabilities for {question_id}")
    return {
        key: _required_number(label, probability, f"probability for {question_id}.{key}")
        for key, probability in value.items()
    }


def _parse_answers(label: str, value: Any, context: ClassifierContext) -> Dict[str, ClassifierAnswer]:
    if not is_record(value):
        raise ValueError(f"{label} returned an unexpected response")
    answers: Dict[str, ClassifierAnswer] = {}
    for question_id, question in context.questions.items():
        answer = value.get(question_id)
        if not is_record(answer):
            raise ValueError(f"{label} did not return an answer for {question_id}")
        if question.type == "choice":
            if answer.get("type") != "choice" or not isinstance(answer.get("choice"), str):
                raise ValueError(f"{label} did not return a choice answer for {question_id}")
            answers[question_id] = ClassifierChoiceAnswer(
                choice=answer["choice"],
                probabilities=_probabilities(label, answer.get("probabilities"), question_id),
                confidence=_required_number(label, answer.get("confidence"), f"confidence for {question_id}"),
            )
        elif question.type == "score":
            if answer.get("type") != "score":
                raise ValueError(f"{label} did not return a score answer for {question_id}")
            answers[question_id] = ClassifierScoreAnswer(
                score=_required_number(label, answer.get("score"), f"score for {question_id}"),
                confidence=_required_number(label, answer.get("confidence"), f"confidence for {question_id}"),
            )
        else:
            if answer.get("type") != "noul":
                raise ValueError(f"{label} did not return a bool answer for {question_id}")
            answers[question_id] = ClassifierBoolAnswer(
                probability=_required_number(label, answer.get("noul"), f"probability for {question_id}"),
            )
    return answers


def _wire_request(context: ClassifierContext) -> Dict[str, Any]:
    """Maps public `bool` questions to TypeSafe's wire-level `noul` type."""
    questions: Dict[str, Any] = {}
    for question_id, question in context.questions.items():
        dumped = question.model_dump(by_alias=True, exclude_none=True)
        if question.type == "bool":
            dumped["type"] = "noul"
        questions[question_id] = dumped
    return {"state": context.state, "questions": questions}


def _request_headers(
    model: ClassifierModel,
    api_key: str,
    options_headers: Optional[ProviderHeaders],
) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    for overrides in (model.headers, options_headers):
        for name, value in (overrides or {}).items():
            if value is None:
                for existing in list(headers.keys()):
                    if existing.lower() == name.lower():
                        del headers[existing]
            else:
                headers[name] = value
    return headers


async def classify_system_one(
    transport: SystemOneTransport,
    model: ClassifierModel,
    context: ClassifierContext,
    options: Optional[ProviderRequestOptions] = None,
) -> ClassifierResult:
    """Runs one System One classification over the given transport. Never raises."""
    output = ClassifierResult(
        api=model.api,
        provider=model.provider,
        model=model.id,
        answers={},
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )

    try:
        if model.api != transport.api:
            raise ValueError(f"Unsupported classifier API: {model.api}")
        api_key = options.api_key if options else None
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")

        payload = transport.payload(model, _wire_request(context))
        if options and options.on_payload:
            transformed = options.on_payload(payload, model)
            if asyncio.iscoroutine(transformed):
                transformed = await transformed
            if transformed is not None:
                payload = transformed

        headers = _request_headers(model, api_key, options.headers if options else None)
        url = transport.url(model)

        timeout = httpx.Timeout((options.timeout_ms or 60_000) / 1000, connect=60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async def do_request() -> httpx.Response:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code >= 400:
                    raise ProviderHttpError(
                        f"{transport.label} returned {response.status_code}: {response.text}",
                        status=response.status_code,
                        headers=response.headers,
                    )
                return response

            response = await retry_provider_request(
                do_request,
                max_retries=options.max_retries if options and options.max_retries is not None else 2,
                max_retry_delay_ms=options.max_retry_delay_ms if options else None,
                signal=options.signal if options else None,
            )

            if options and options.on_response:
                maybe = options.on_response(
                    ProviderResponse(status=response.status_code, headers=dict(response.headers)), model
                )
                if asyncio.iscoroutine(maybe):
                    await maybe

            body = response.json()

        output.answers = _parse_answers(transport.label, transport.answers(body), context)
        return output
    except Exception as error:
        output.stop_reason = "aborted" if (options and options.signal and options.signal.aborted) else "error"
        output.error_message = format_provider_error(normalize_provider_error(error), f"{transport.label} error")
        return output

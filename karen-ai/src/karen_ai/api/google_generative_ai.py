"""Google Generative AI (Gemini Developer API) adapter.

Mirrors api/google-generative-ai.ts, but speaks the documented REST API
directly (POST {base_url}/models/{model}:streamGenerateContent?alt=sse)
instead of going through the @google/genai SDK.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterable, Dict, Optional

import httpx

from ..event_stream import AssistantMessageEventStream
from ..transcript import collapse_system_messages
from ..types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ProviderHeaders,
    ProviderResponse,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TranscriptContext,
    Usage,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.retry import ProviderHttpError, retry_provider_request
from ..utils.sse import iterate_sse_messages
from .google_shared import (
    GoogleThinkingConfig,
    build_google_config,
    build_google_stream_simple_thinking,
    convert_messages,
    process_google_stream,
)
from .request_options import coerce_options
from .simple_options import build_base_options

KAREN_USER_AGENT = "karen-ai/0.1.0"


class GoogleOptions(StreamOptions):
    tool_choice: Optional[str] = None  # "auto" | "none" | "any"
    thinking: Optional[GoogleThinkingConfig] = None

    model_config = {"arbitrary_types_allowed": True}


def _build_headers(
    model: Model,
    api_key: str,
    options_headers: Optional[ProviderHeaders],
) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "User-Agent": KAREN_USER_AGENT,
        "content-type": "application/json",
        "x-goog-api-key": api_key,
    }
    for name, value in (model.headers or {}).items():
        if value is not None:
            headers[name] = value
    # Merge options headers last so they can override defaults; None suppresses.
    for name, value in (options_headers or {}).items():
        if value is None:
            for existing in list(headers.keys()):
                if existing.lower() == name.lower():
                    del headers[existing]
        else:
            headers[name] = value
    return headers


def build_params(model: Model, context: TranscriptContext, options: Optional[GoogleOptions]) -> Dict[str, Any]:
    body: Dict[str, Any] = {"contents": convert_messages(model, context)}
    body.update(build_google_config(model, context, options))
    return body


async def _iterate_google_chunks(response: httpx.Response, signal=None) -> AsyncIterable[Dict[str, Any]]:
    async for sse in iterate_sse_messages(response.aiter_text(), signal):
        data = sse.data.strip()
        if not data:
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(chunk, dict):
            yield chunk


def stream(
    model: Model,
    context: TranscriptContext,
    options: Optional[GoogleOptions] = None,
) -> AssistantMessageEventStream:
    options = coerce_options(options, GoogleOptions)
    event_stream = AssistantMessageEventStream()
    normalized_context = collapse_system_messages(context)

    async def run() -> None:
        output = AssistantMessage(
            role="assistant",
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="pending",
            timestamp=int(time.time() * 1000),
        )

        try:
            api_key = options.api_key if options else None
            if not api_key:
                raise ValueError(f"No API key for provider: {model.provider}")

            params = build_params(model, normalized_context, options)
            if options and options.on_payload:
                next_params = options.on_payload(params, model)
                if asyncio.iscoroutine(next_params):
                    next_params = await next_params
                if next_params is not None:
                    params = next_params

            headers = _build_headers(model, api_key, options.headers if options else None)

            timeout = httpx.Timeout((options.timeout_ms or 600_000) / 1000, connect=60.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                url = f"{model.base_url.rstrip('/')}/models/{model.id}:streamGenerateContent?alt=sse"

                async def do_request() -> httpx.Response:
                    request = client.build_request("POST", url, json=params, headers=headers)
                    response = await client.send(request, stream=True)
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        await response.aclose()
                        raise ProviderHttpError(
                            f"Google Generative AI API error {response.status_code}: {body}",
                            status=response.status_code,
                            headers=response.headers,
                        )
                    return response

                response = await retry_provider_request(
                    do_request,
                    max_retries=options.max_retries if options else None,
                    max_retry_delay_ms=options.max_retry_delay_ms if options else None,
                    signal=options.signal if options else None,
                )

                try:
                    if options and options.on_response:
                        maybe = options.on_response(
                            ProviderResponse(status=response.status_code, headers=dict(response.headers)), model
                        )
                        if asyncio.iscoroutine(maybe):
                            await maybe

                    event_stream.push(StartEvent(partial=output))

                    await process_google_stream(
                        _iterate_google_chunks(response, options.signal if options else None),
                        output,
                        event_stream,
                        model,
                        options.on_provider_stream_event if options else None,
                    )

                    if options and options.signal and options.signal.aborted:
                        raise RuntimeError("Request was aborted")
                    if output.stop_reason == "pending":
                        raise RuntimeError("Google stream ended without a finish reason")
                    if output.stop_reason in ("aborted", "error"):
                        error_message = (
                            f"Provider stopped with: {output.raw_stop_reason}"
                            if output.raw_stop_reason
                            else "An unknown error occurred"
                        )
                        raise RuntimeError(error_message)

                    event_stream.push(DoneEvent(reason=output.stop_reason, message=output))  # type: ignore[arg-type]
                    event_stream.end()
                finally:
                    await response.aclose()

        except Exception as error:
            output.stop_reason = "aborted" if (options and options.signal and options.signal.aborted) else "error"
            output.error_message = format_provider_error(normalize_provider_error(error))
            event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))  # type: ignore[arg-type]
            event_stream.end()

    asyncio.get_running_loop().create_task(run())
    return event_stream


def stream_simple(
    model: Model,
    context: TranscriptContext,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    api_key = options.api_key if options else None
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, context, options, api_key)
    merged = GoogleOptions(**base.model_dump(exclude_none=True))
    merged.tool_choice = options.tool_choice if options else None
    merged.thinking = build_google_stream_simple_thinking(model, options)
    return stream(model, context, merged)

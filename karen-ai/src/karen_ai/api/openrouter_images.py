"""OpenRouter image generation, mirroring api/openrouter-images.ts.

Image generation over OpenRouter's chat completions endpoint (non-streaming,
`modalities: ["image", "text"]`).
"""

from __future__ import annotations

import asyncio
import base64
import re
import time
from typing import Any, Dict, Optional

import httpx

from ..types import (
    AssistantImages,
    ImageContent,
    ImageModel,
    ImagesContext,
    ProviderHeaders,
    ProviderRequestOptions,
    ProviderResponse,
    TextContent,
    Usage,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.retry import ProviderHttpError, retry_provider_request
from ..utils.sanitize import sanitize_surrogates

KAREN_USER_AGENT = "karen-ai/0.1.0"

_DATA_URL_PATTERN = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)


def _build_headers(
    model: ImageModel,
    api_key: str,
    options_headers: Optional[ProviderHeaders],
) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "User-Agent": KAREN_USER_AGENT,
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
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


def _build_params(model: ImageModel, context: ImagesContext) -> Dict[str, Any]:
    content = []
    for item in context.input:
        if item.type == "text":
            content.append({"type": "text", "text": sanitize_surrogates(item.text)})
        else:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"},
                }
            )
    return {
        "model": model.id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "modalities": ["image", "text"] if "text" in model.output else ["image"],
    }


def _parse_usage(raw_usage: Dict[str, Any], model: ImageModel) -> Usage:
    prompt_tokens = raw_usage.get("prompt_tokens") or 0
    prompt_details = raw_usage.get("prompt_tokens_details") or {}
    reported_cached = prompt_details.get("cached_tokens") or 0
    cache_write_tokens = prompt_details.get("cache_write_tokens") or 0
    cache_read_tokens = max(0, reported_cached - cache_write_tokens) if cache_write_tokens > 0 else reported_cached
    input_tokens = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)
    output_tokens = raw_usage.get("completion_tokens") or 0

    cost = model.cost
    usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read_tokens,
        cache_write=cache_write_tokens,
        total_tokens=input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
    )
    usage.cost.input = (cost.input / 1_000_000) * input_tokens
    usage.cost.output = (cost.output / 1_000_000) * output_tokens
    usage.cost.cache_read = (cost.cache_read / 1_000_000) * cache_read_tokens
    usage.cost.cache_write = (cost.cache_write / 1_000_000) * cache_write_tokens
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    return usage


async def generate_images(
    model: ImageModel,
    context: ImagesContext,
    options: Optional[ProviderRequestOptions] = None,
) -> AssistantImages:
    """Image generation over OpenRouter's chat completions endpoint. Never raises."""
    output = AssistantImages(
        api=model.api,
        provider=model.provider,
        model=model.id,
        output=[],
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )

    try:
        api_key = options.api_key if options else None
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")

        params = _build_params(model, context)
        if options and options.on_payload:
            next_params = options.on_payload(params, model)
            if asyncio.iscoroutine(next_params):
                next_params = await next_params
            if next_params is not None:
                params = next_params

        headers = _build_headers(model, api_key, options.headers if options else None)

        timeout = httpx.Timeout((options.timeout_ms or 600_000) / 1000, connect=60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            url = model.base_url.rstrip("/") + "/chat/completions"

            async def do_request() -> httpx.Response:
                response = await client.post(url, json=params, headers=headers)
                if response.status_code >= 400:
                    body = response.text
                    raise ProviderHttpError(
                        f"OpenRouter Images API error {response.status_code}: {body}",
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

            if options and options.on_response:
                maybe = options.on_response(
                    ProviderResponse(status=response.status_code, headers=dict(response.headers)), model
                )
                if asyncio.iscoroutine(maybe):
                    await maybe

            body = response.json()

        output.response_id = body.get("id")
        if body.get("usage"):
            output.usage = _parse_usage(body["usage"], model)

        choices = body.get("choices") or []
        choice = choices[0] if choices else None
        if choice:
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content:
                output.output.append(TextContent(text=content))

            for image in message.get("images") or []:
                image_url = image.get("image_url")
                if isinstance(image_url, dict):
                    image_url = image_url.get("url")
                if not isinstance(image_url, str) or not image_url.startswith("data:"):
                    continue
                matches = _DATA_URL_PATTERN.match(image_url)
                if not matches:
                    continue
                output.output.append(ImageContent(mime_type=matches.group(1), data=matches.group(2)))

        return output
    except Exception as error:
        output.stop_reason = "aborted" if (options and options.signal and options.signal.aborted) else "error"
        output.error_message = format_provider_error(normalize_provider_error(error))
        return output

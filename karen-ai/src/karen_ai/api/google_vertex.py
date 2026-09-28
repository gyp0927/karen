"""Google Vertex AI adapter, mirroring api/google-vertex.ts.

Speaks the documented Vertex REST API directly instead of the @google/genai
SDK. Two auth modes:
- Vertex API key (express mode): `x-goog-api-key` header against
  `https://aiplatform.googleapis.com`.
- Application Default Credentials (`gcloud auth application-default login` or
  GOOGLE_APPLICATION_CREDENTIALS): OAuth2 bearer token minted via the optional
  `google-auth` package, against `https://{location}-aiplatform.googleapis.com`.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncIterable, Dict, Optional, Tuple

import httpx

from ..event_stream import AssistantMessageEventStream
from ..transcript import collapse_system_messages
from ..types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ProviderEnv,
    ProviderHeaders,
    ProviderResponse,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TranscriptContext,
    Usage,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.provider_env import get_provider_env_value
from ..utils.retry import ProviderHttpError, retry_provider_request
from ..utils.sse import iterate_sse_messages
from .google_generative_ai import _iterate_google_chunks, build_params
from .google_shared import (
    GoogleThinkingConfig,
    build_google_stream_simple_thinking,
    process_google_stream,
)
from .simple_options import build_base_options

KAREN_USER_AGENT = "karen-ai/0.1.0"

API_VERSION = "v1"
GCP_VERTEX_CREDENTIALS_MARKER = "gcp-vertex-credentials"
DEFAULT_VERTEX_BASE_URL = "https://aiplatform.googleapis.com"

_ADC_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_adc_credentials: Any = None


class GoogleVertexOptions(StreamOptions):
    tool_choice: Optional[str] = None  # "auto" | "none" | "any"
    thinking: Optional[GoogleThinkingConfig] = None
    project: Optional[str] = None
    location: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Auth resolution
# ---------------------------------------------------------------------------


def _is_placeholder_api_key(api_key: str) -> bool:
    return re.match(r"^<[^>]+>$", api_key) is not None


def _resolve_api_key(options: Optional[GoogleVertexOptions]) -> Optional[str]:
    api_key = (options.api_key or "").strip() if options else ""
    if not api_key or api_key == GCP_VERTEX_CREDENTIALS_MARKER or _is_placeholder_api_key(api_key):
        return None
    return api_key


def _resolve_project(options: Optional[GoogleVertexOptions]) -> str:
    project = (
        (options.project if options else None)
        or get_provider_env_value("GOOGLE_CLOUD_PROJECT", options.env if options else None)
        or get_provider_env_value("GCLOUD_PROJECT", options.env if options else None)
    )
    if not project:
        raise ValueError(
            "Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options."
        )
    return project


def _resolve_location(options: Optional[GoogleVertexOptions]) -> str:
    location = (options.location if options else None) or get_provider_env_value(
        "GOOGLE_CLOUD_LOCATION", options.env if options else None
    )
    if not location:
        raise ValueError("Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.")
    return location


async def _get_adc_access_token() -> str:
    """Mint an access token from Application Default Credentials.

    GOOGLE_APPLICATION_CREDENTIALS is honored by google.auth.default itself.
    """
    global _adc_credentials
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError as exc:
        raise ValueError(
            "Vertex ADC auth requires the 'google-auth' package (pip install karen-ai[vertex]) "
            "or a Vertex API key."
        ) from exc

    if _adc_credentials is None:
        credentials, _ = await asyncio.to_thread(google.auth.default, scopes=_ADC_SCOPES)
        _adc_credentials = credentials

    credentials = _adc_credentials
    if not credentials.valid or credentials.expired:
        request = google.auth.transport.requests.Request()
        await asyncio.to_thread(credentials.refresh, request)
    return credentials.token


# ---------------------------------------------------------------------------
# URL building
# ---------------------------------------------------------------------------


def _base_url_includes_api_version(base_url: str) -> bool:
    path = re.sub(r"^https?://[^/]+", "", base_url)
    if not path:
        return re.search(r"(?:^|/)v\d+(?:beta\d*)?(?:/|$)", base_url) is not None
    return any(re.match(r"^v\d+(?:beta\d*)?$", part) for part in path.split("/"))


def _resolve_custom_base_url(base_url: str) -> Optional[str]:
    trimmed = (base_url or "").strip()
    if not trimmed or "{location}" in trimmed:
        return None
    return trimmed


def _versioned_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if _base_url_includes_api_version(base):
        return base
    return f"{base}/{API_VERSION}"


def _build_api_key_url(model: Model) -> str:
    base = _versioned_base(_resolve_custom_base_url(model.base_url) or DEFAULT_VERTEX_BASE_URL)
    return f"{base}/publishers/google/models/{model.id}:streamGenerateContent?alt=sse"


def _build_adc_url(model: Model, project: str, location: str) -> str:
    custom = _resolve_custom_base_url(model.base_url)
    base = _versioned_base(custom) if custom else f"https://{location}-aiplatform.googleapis.com/{API_VERSION}"
    return (
        f"{base}/projects/{project}/locations/{location}"
        f"/publishers/google/models/{model.id}:streamGenerateContent?alt=sse"
    )


def _build_headers(
    model: Model,
    auth_header: Tuple[str, str],
    options_headers: Optional[ProviderHeaders],
) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "User-Agent": KAREN_USER_AGENT,
        "content-type": "application/json",
        auth_header[0]: auth_header[1],
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


# ---------------------------------------------------------------------------
# stream / stream_simple
# ---------------------------------------------------------------------------


def stream(
    model: Model,
    context: TranscriptContext,
    options: Optional[GoogleVertexOptions] = None,
) -> AssistantMessageEventStream:
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
            api_key = _resolve_api_key(options)
            if api_key:
                url = _build_api_key_url(model)
                auth_header = ("x-goog-api-key", api_key)
            else:
                project = _resolve_project(options)
                location = _resolve_location(options)
                url = _build_adc_url(model, project, location)
                auth_header = ("authorization", f"Bearer {await _get_adc_access_token()}")

            params = build_params(model, normalized_context, options)
            if options and options.on_payload:
                next_params = options.on_payload(params, model)
                if asyncio.iscoroutine(next_params):
                    next_params = await next_params
                if next_params is not None:
                    params = next_params

            headers = _build_headers(model, auth_header, options.headers if options else None)

            timeout = httpx.Timeout((options.timeout_ms or 600_000) / 1000, connect=60.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async def do_request() -> httpx.Response:
                    request = client.build_request("POST", url, json=params, headers=headers)
                    response = await client.send(request, stream=True)
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        await response.aclose()
                        raise ProviderHttpError(
                            f"Google Vertex API error {response.status_code}: {body}",
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
                        raise RuntimeError("Google Vertex stream ended without a finish reason")
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
    base = build_base_options(model, context, options, None)
    merged = GoogleVertexOptions(**base.model_dump(exclude_none=True))
    merged.tool_choice = options.tool_choice if options else None
    merged.thinking = build_google_stream_simple_thinking(model, options)
    return stream(model, context, merged)

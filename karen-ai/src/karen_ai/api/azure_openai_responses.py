"""Azure OpenAI Responses adapter, mirroring api/azure-openai-responses.ts.

Speaks Azure's GA v1 Responses API directly:
POST {base_url}/responses?api-version={version} with `api-key` auth, where
base_url is normalized to https://{resource}.openai.azure.com/openai/v1 and
the deployment name travels in the body's `model` field.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import httpx

from ..event_stream import AssistantMessageEventStream
from ..models import clamp_thinking_level
from ..transcript import get_declared_tools, resolve_transcript, resolve_transcript_tools
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
from .constrained_sampling import create_grammar_tool_input_properties
from .openai_prompt_cache import clamp_openai_prompt_cache_key
from .openai_responses import _iterate_responses_events
from .openai_responses_shared import (
    ConvertResponsesMessagesOptions,
    ConvertResponsesToolsOptions,
    OpenAIResponsesStreamOptions,
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from .simple_options import build_base_options

KAREN_USER_AGENT = "karen-ai/0.1.0"

DEFAULT_AZURE_API_VERSION = "v1"
AZURE_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode", "azure-openai-responses"}
# OpenAI Responses rejects max_output_tokens below 16:
# https://github.com/earendil-works/pi/issues/6265
OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16


class AzureOpenAIResponsesOptions(StreamOptions):
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None  # "auto" | "detailed" | "concise"
    tool_choice: Optional[Any] = None
    azure_api_version: Optional[str] = None
    azure_resource_name: Optional[str] = None
    azure_base_url: Optional[str] = None
    azure_deployment_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Azure config resolution
# ---------------------------------------------------------------------------


def parse_deployment_name_map(value: Optional[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not value:
        return mapping
    for entry in value.split(","):
        trimmed = entry.strip()
        if not trimmed:
            continue
        parts = trimmed.split("=", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            continue
        mapping[parts[0].strip()] = parts[1].strip()
    return mapping


def resolve_deployment_name(model: Model, options: Optional[AzureOpenAIResponsesOptions]) -> str:
    if options and options.azure_deployment_name:
        return options.azure_deployment_name
    mapped = parse_deployment_name_map(
        get_provider_env_value("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", options.env if options else None)
    ).get(model.id)
    return mapped or model.id


def normalize_azure_base_url(base_url: str) -> str:
    from urllib.parse import urlparse, urlunparse

    trimmed = base_url.strip().rstrip("/")
    parsed = urlparse(trimmed)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid Azure OpenAI base URL: {base_url}")

    is_azure_host = (
        parsed.hostname.endswith(".openai.azure.com")
        or parsed.hostname.endswith(".cognitiveservices.azure.com")
        or parsed.hostname.endswith(".ai.azure.com")
    )
    normalized_path = parsed.path.rstrip("/")

    # Ensure Azure hosts have /openai/v1 as base path so /responses and
    # ?api-version=v1 resolve correctly.
    if is_azure_host and normalized_path in ("", "/", "/openai", "/openai/v1/responses"):
        parsed = parsed._replace(path="/openai/v1", query="")

    return urlunparse(parsed).rstrip("/")


def _build_default_base_url(resource_name: str) -> str:
    return f"https://{resource_name}.openai.azure.com/openai/v1"


def resolve_azure_config(model: Model, options: Optional[AzureOpenAIResponsesOptions]) -> Dict[str, str]:
    api_version = (
        (options.azure_api_version if options else None)
        or get_provider_env_value("AZURE_OPENAI_API_VERSION", options.env if options else None)
        or DEFAULT_AZURE_API_VERSION
    )

    env = options.env if options else None
    base_url = (
        (options.azure_base_url or "").strip()
        if options and options.azure_base_url
        else (get_provider_env_value("AZURE_OPENAI_BASE_URL", env) or "").strip()
    ) or None
    resource_name = (options.azure_resource_name if options else None) or get_provider_env_value(
        "AZURE_OPENAI_RESOURCE_NAME", env
    )

    resolved_base_url = base_url
    if not resolved_base_url and resource_name:
        resolved_base_url = _build_default_base_url(resource_name)
    if not resolved_base_url and model.base_url:
        resolved_base_url = model.base_url
    if not resolved_base_url:
        raise ValueError(
            "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, "
            "or pass azure_base_url, azure_resource_name, or model.base_url."
        )

    return {"base_url": normalize_azure_base_url(resolved_base_url), "api_version": api_version}


# ---------------------------------------------------------------------------
# Headers / params
# ---------------------------------------------------------------------------


def _build_headers(
    model: Model,
    api_key: str,
    options_headers: Optional[ProviderHeaders],
) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "User-Agent": KAREN_USER_AGENT,
        "content-type": "application/json",
        "api-key": api_key,
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


def _compat_flag(model: Model, name: str, default: bool) -> bool:
    value = getattr(model.compat, name, None)
    return default if value is None else bool(value)


def _build_params(
    model: Model,
    context: TranscriptContext,
    options: Optional[AzureOpenAIResponsesOptions],
    deployment_name: str,
    grammar_tool_input_properties: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            get_declared_tools(context.messages), _compat_flag(model, "supports_openai_grammar_tools", False)
        )
    supports_additional_tools = _compat_flag(model, "supports_additional_tools", False)
    supports_tool_search = _compat_flag(model, "supports_tool_search", False)
    supports_strict_mode = _compat_flag(model, "supports_strict_mode", True)
    supports_grammar = _compat_flag(model, "supports_openai_grammar_tools", False)

    transcript_tools = resolve_transcript_tools(context.messages, supports_additional_tools or supports_tool_search)
    messages = convert_responses_messages(
        model,
        context,
        AZURE_TOOL_CALL_PROVIDERS,
        ConvertResponsesMessagesOptions(
            grammar_tool_input_properties=grammar_tool_input_properties,
            supports_mid_convo_system_messages=_compat_flag(model, "supports_mid_convo_system_messages", False),
            supports_additional_tools=supports_additional_tools,
            supports_tool_search=supports_tool_search,
            tool_options=ConvertResponsesToolsOptions(
                supports_strict_mode=supports_strict_mode,
                supports_openai_grammar_tools=supports_grammar,
            ),
        ),
    )

    params: Dict[str, Any] = {
        "model": deployment_name,
        "input": messages,
        "stream": True,
        "store": False,
    }
    cache_key = clamp_openai_prompt_cache_key(options.session_id if options else None)
    if cache_key is not None:
        params["prompt_cache_key"] = cache_key

    if options and options.max_tokens:
        params["max_output_tokens"] = max(options.max_tokens, OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)

    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    if transcript_tools.request_tools:
        params["tools"] = convert_responses_tools(
            transcript_tools.request_tools,
            ConvertResponsesToolsOptions(
                supports_strict_mode=supports_strict_mode,
                supports_openai_grammar_tools=supports_grammar,
            ),
        )

    if options and options.tool_choice is not None:
        params["tool_choice"] = options.tool_choice

    if model.reasoning:
        reasoning_effort = options.reasoning_effort if options else None
        reasoning_summary = options.reasoning_summary if options else None
        level_map = model.thinking_level_map or {}
        if reasoning_effort or reasoning_summary:
            effort = (level_map.get(reasoning_effort) or reasoning_effort) if reasoning_effort else "medium"
            params["reasoning"] = {"effort": effort, "summary": reasoning_summary or "auto"}
            params["include"] = ["reasoning.encrypted_content"]
        elif not ("off" in level_map and level_map["off"] is None):
            params["reasoning"] = {"effort": level_map.get("off") or "none"}

    # Last so custom keys override the named request fields. Per-request keys
    # override model defaults.
    if model.sampling_params:
        params.update(model.sampling_params)
    if options and options.sampling_params:
        params.update(options.sampling_params)

    return params


# ---------------------------------------------------------------------------
# stream / stream_simple
# ---------------------------------------------------------------------------


def stream(
    model: Model,
    context: TranscriptContext,
    options: Optional[AzureOpenAIResponsesOptions] = None,
) -> AssistantMessageEventStream:
    event_stream = AssistantMessageEventStream()
    normalized_context = resolve_transcript(
        context, _compat_flag(model, "supports_mid_convo_system_messages", False)
    )

    async def run() -> None:
        deployment_name = resolve_deployment_name(model, options)
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

            grammar_tool_input_properties = create_grammar_tool_input_properties(
                get_declared_tools(normalized_context.messages),
                _compat_flag(model, "supports_openai_grammar_tools", False),
            )
            params = _build_params(model, normalized_context, options, deployment_name, grammar_tool_input_properties)
            if options and options.on_payload:
                next_params = options.on_payload(params, model)
                if asyncio.iscoroutine(next_params):
                    next_params = await next_params
                if next_params is not None:
                    params = next_params

            config = resolve_azure_config(model, options)
            url = f"{config['base_url']}/responses?api-version={config['api_version']}"
            headers = _build_headers(model, api_key, options.headers if options else None)

            timeout = httpx.Timeout((options.timeout_ms or 600_000) / 1000, connect=60.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async def do_request() -> httpx.Response:
                    request = client.build_request("POST", url, json=params, headers=headers)
                    response = await client.send(request, stream=True)
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        await response.aclose()
                        raise ProviderHttpError(
                            f"Azure OpenAI API error {response.status_code}: {body}",
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

                    await process_responses_stream(
                        _iterate_responses_events(response, options.signal if options else None),
                        output,
                        event_stream,
                        model,
                        OpenAIResponsesStreamOptions(
                            on_provider_stream_event=options.on_provider_stream_event if options else None,
                            grammar_tool_input_properties=grammar_tool_input_properties,
                        ),
                    )

                    if options and options.signal and options.signal.aborted:
                        raise RuntimeError("Request was aborted")
                    if output.stop_reason == "pending":
                        raise RuntimeError("Azure OpenAI Responses stream ended without a stop reason")
                    if output.stop_reason in ("aborted", "error"):
                        raise RuntimeError(output.error_message or "An unknown error occurred")

                    event_stream.push(DoneEvent(reason=output.stop_reason, message=output))  # type: ignore[arg-type]
                    event_stream.end()
                finally:
                    await response.aclose()

        except Exception as error:
            output.stop_reason = "aborted" if (options and options.signal and options.signal.aborted) else "error"
            output.error_message = format_provider_error(normalize_provider_error(error), "Azure OpenAI API error")
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
    clamped_reasoning = clamp_thinking_level(model, options.reasoning) if options and options.reasoning else None
    reasoning_effort = None if clamped_reasoning == "off" else clamped_reasoning

    merged = AzureOpenAIResponsesOptions(**base.model_dump(exclude_none=True))
    merged.tool_choice = options.tool_choice if options else None
    merged.reasoning_effort = reasoning_effort
    return stream(model, context, merged)

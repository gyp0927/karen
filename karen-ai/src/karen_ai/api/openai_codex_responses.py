"""OpenAI Codex Responses API, mirroring pi-ai's api/openai-codex-responses.ts.

ChatGPT-backend Responses variant: JWT account auth, `store: false`, encrypted
reasoning, zstd-compressed request bodies when a zstd codec is available, and
usage-limit aware retries.

Transport note: pi-ai additionally speaks a cached WebSocket transport
(`transport="websocket" | "auto"` prefers it). karen-ai currently implements
the SSE transport only; `auto`/`websocket` transparently use SSE, matching
pi-ai's own fallback behavior when the socket cannot be established.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from typing import Any, Dict, List, Optional

import httpx

from ..event_stream import AssistantMessageEventStream
from ..models import clamp_thinking_level
from ..transcript import (
    get_declared_tools,
    get_initial_system_message,
    resolve_transcript,
    resolve_transcript_tools,
)
from ..types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ProviderResponse,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    ThinkingLevel,
    ToolCall,
    TranscriptContext,
    Usage,
)
from ..utils.diagnostics import format_thrown_value
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.headers import headers_to_record, provider_headers_to_record
from ..utils.pi_user_agent import get_pi_user_agent
from ..utils.text import get_system_message_text
from .constrained_sampling import create_grammar_tool_input_properties
from .openai_prompt_cache import clamp_openai_prompt_cache_key
from .openai_responses_shared import (
    ConvertResponsesMessagesOptions,
    ConvertResponsesToolsOptions,
    OpenAIResponsesStreamOptions,
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from .simple_options import build_base_options

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
DEFAULT_MAX_RETRIES = 0
BASE_DELAY_MS = 1000
DEFAULT_MAX_RETRY_DELAY_MS = 60_000
REQUEST_COMPRESSION_ZSTD_LEVEL = 3
CODEX_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode"}

CODEX_RESPONSE_STATUSES = {"completed", "failed", "in_progress", "incomplete", "queued", "cancelled"}


class OpenAICodexResponsesOptions(StreamOptions):
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None
    text_verbosity: Optional[str] = None
    tool_choice: Optional[Any] = None
    service_tier: Optional[str] = None


class CodexApiError(Exception):
    def __init__(self, message: str, code: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.payload = payload


class CodexProtocolError(Exception):
    def __init__(self, message: str, payload: Any = None) -> None:
        super().__init__(message)
        self.payload = payload


def _assert_successful_output(output: AssistantMessage) -> None:
    if output.stop_reason == "pending":
        raise ValueError("Codex stream ended without a stop reason")
    if output.stop_reason in ("error", "aborted"):
        raise ValueError(output.error_message or "An unknown error occurred")


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


def _is_terminal_rate_limit_error(error_text: str) -> bool:
    return bool(
        re.search(
            r"GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance"
            r"|insufficient_quota|out of budget|quota exceeded|billing",
            error_text,
            re.IGNORECASE,
        )
    )


def _is_retryable_error(status: int, error_text: str) -> bool:
    if status == 429 and _is_terminal_rate_limit_error(error_text):
        return False
    if status in (429, 500, 502, 503, 504):
        return True
    return bool(
        re.search(r"rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused", error_text, re.IGNORECASE)
    )


def _get_retry_after_delay_ms(headers: httpx.Headers) -> Optional[float]:
    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            return max(0.0, float(retry_after_ms))
        except ValueError:
            pass
    retry_after = headers.get("retry-after")
    if not retry_after:
        return None
    try:
        return max(0.0, float(retry_after) * 1000)
    except ValueError:
        pass
    from email.utils import parsedate_to_datetime

    try:
        return max(0.0, parsedate_to_datetime(retry_after).timestamp() * 1000 - time.time() * 1000)
    except (TypeError, ValueError):
        return None


class RetryDelayExceededError(Exception):
    pass


def _validate_retry_delay_ms(delay_ms: float, options: Optional[StreamOptions]) -> float:
    max_retry_delay_ms = (
        options.max_retry_delay_ms if options and options.max_retry_delay_ms is not None else DEFAULT_MAX_RETRY_DELAY_MS
    )
    if max_retry_delay_ms > 0 and delay_ms > max_retry_delay_ms:
        raise RetryDelayExceededError(
            f"Server requested {int(delay_ms / 1000 + 0.5)}s retry delay (max: {int(max_retry_delay_ms / 1000)}s)"
        )
    return delay_ms


async def _sleep_ms(ms: float, signal=None) -> None:
    if signal is not None and signal.aborted:
        raise ValueError("Request was aborted")
    await asyncio.sleep(ms / 1000)
    if signal is not None and signal.aborted:
        raise ValueError("Request was aborted")


def _compress_request_body_zstd(body: bytes) -> Optional[bytes]:
    """zstd-compressed body, or None when no codec is available (pi-ai does the
    same in browser builds; the Codex backend accepts both)."""
    try:
        from compression import zstd  # Python 3.14+

        return zstd.compress(body, level=REQUEST_COMPRESSION_ZSTD_LEVEL)
    except ImportError:
        pass
    try:
        import zstandard

        return zstandard.ZstdCompressor(level=REQUEST_COMPRESSION_ZSTD_LEVEL).compress(body)
    except ImportError:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Auth & headers
# ---------------------------------------------------------------------------


def extract_account_id(token: str) -> str:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token")
        payload = json.loads(base64.urlsafe_b64decode(_b64pad(parts[1])))
        account_id = payload.get(JWT_CLAIM_PATH, {}).get("chatgpt_account_id")
        if not account_id:
            raise ValueError("No account ID in token")
        return account_id
    except Exception:
        raise ValueError("Failed to extract accountId from token") from None


def _b64pad(value: str) -> str:
    return value + "=" * (-len(value) % 4)


def _build_base_codex_headers(
    init_headers: Optional[Dict[str, str]],
    additional_headers: Optional[Dict[str, Optional[str]]],
    account_id: str,
    token: str,
) -> Dict[str, str]:
    headers: Dict[str, str] = dict(init_headers or {})
    for key, value in (additional_headers or {}).items():
        lower = key.lower()
        for existing in list(headers.keys()):
            if existing.lower() == lower:
                del headers[existing]
        if value is not None:
            headers[key] = value
    headers["Authorization"] = f"Bearer {token}"
    headers["chatgpt-account-id"] = account_id
    headers["originator"] = "pi"
    headers["User-Agent"] = get_pi_user_agent()
    return headers


def _build_sse_headers(
    init_headers: Optional[Dict[str, str]],
    additional_headers: Optional[Dict[str, Optional[str]]],
    account_id: str,
    token: str,
    session_id: Optional[str],
) -> Dict[str, str]:
    headers = _build_base_codex_headers(init_headers, additional_headers, account_id, token)
    headers["OpenAI-Beta"] = "responses=experimental"
    headers["accept"] = "text/event-stream"
    headers["content-type"] = "application/json"
    if session_id:
        headers["session-id"] = session_id
        headers["x-client-request-id"] = session_id
    return headers


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def _build_request_body(
    model: Model,
    context: TranscriptContext,
    options: Optional[OpenAICodexResponsesOptions],
    cache_session_id: Optional[str],
    grammar_tool_input_properties: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    compat = model.compat
    supports_strict_mode = getattr(compat, "supports_strict_mode", None)
    supports_strict_mode = True if supports_strict_mode is None else supports_strict_mode
    supports_grammar = bool(getattr(compat, "supports_openai_grammar_tools", None))
    supports_additional_tools = bool(getattr(compat, "supports_additional_tools", None))
    supports_tool_search = bool(getattr(compat, "supports_tool_search", None))
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            get_declared_tools(context.messages), supports_grammar
        )

    transcript_tools = resolve_transcript_tools(context.messages, supports_additional_tools or supports_tool_search)
    messages = convert_responses_messages(
        model,
        context,
        CODEX_TOOL_CALL_PROVIDERS,
        ConvertResponsesMessagesOptions(
            include_system_prompt=False,
            grammar_tool_input_properties=grammar_tool_input_properties,
            supports_mid_convo_system_messages=bool(getattr(compat, "supports_mid_convo_system_messages", None)),
            supports_additional_tools=supports_additional_tools,
            supports_tool_search=supports_tool_search,
            tool_options=ConvertResponsesToolsOptions(
                strict=None,
                supports_strict_mode=supports_strict_mode,
                supports_openai_grammar_tools=supports_grammar,
            ),
        ),
    )

    initial_system_message = get_initial_system_message(context.messages)
    instructions = get_system_message_text(initial_system_message) if initial_system_message else ""
    body: Dict[str, Any] = {
        "model": model.id,
        "store": False,
        "stream": True,
        "instructions": instructions or "You are a helpful assistant.",
        "input": messages,
        "text": {"verbosity": (options.text_verbosity if options else None) or "low"},
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": cache_session_id,
        "tool_choice": (options.tool_choice if options else None) or "auto",
        "parallel_tool_calls": True,
    }

    if options and options.temperature is not None:
        body["temperature"] = options.temperature
    if options and options.service_tier is not None:
        body["service_tier"] = options.service_tier
    if transcript_tools.request_tools:
        body["tools"] = convert_responses_tools(
            transcript_tools.request_tools,
            ConvertResponsesToolsOptions(
                strict=None,
                supports_strict_mode=supports_strict_mode,
                supports_openai_grammar_tools=supports_grammar,
            ),
        )

    if options and options.reasoning_effort is not None:
        thinking_map = model.thinking_level_map
        if options.reasoning_effort == "none":
            effort = "none" if thinking_map is None or thinking_map.off is None else thinking_map.off
        else:
            mapped = getattr(thinking_map, options.reasoning_effort, None) if thinking_map else None
            effort = mapped if mapped is not None else options.reasoning_effort
        if effort is not None:
            body["reasoning"] = {
                "effort": effort,
                "summary": (options.reasoning_summary if options else None) or "auto",
            }
    elif model.reasoning and (model.thinking_level_map is None or model.thinking_level_map.off is not None):
        body["reasoning"] = {"effort": model.thinking_level_map.off if model.thinking_level_map else "none"}

    return body


def _get_service_tier_cost_multiplier(model: Model, service_tier: Optional[str]) -> float:
    if service_tier == "flex":
        return 0.5
    if service_tier == "priority":
        return 2.5 if model.id == "gpt-5.5" else 2.0
    return 1.0


def _apply_service_tier_pricing(usage: Usage, service_tier: Optional[str], model: Model) -> None:
    multiplier = _get_service_tier_cost_multiplier(model, service_tier)
    if multiplier == 1:
        return
    usage.cost.input *= multiplier
    usage.cost.output *= multiplier
    usage.cost.cache_read *= multiplier
    usage.cost.cache_write *= multiplier
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write


def _resolve_codex_service_tier(
    response_service_tier: Optional[str], request_service_tier: Optional[str]
) -> Optional[str]:
    if response_service_tier == "default" and request_service_tier in ("flex", "priority"):
        return request_service_tier
    return response_service_tier if response_service_tier is not None else request_service_tier


def resolve_codex_url(base_url: Optional[str]) -> str:
    raw = base_url if base_url and base_url.strip() else DEFAULT_CODEX_BASE_URL
    normalized = raw.rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


# ---------------------------------------------------------------------------
# Response processing
# ---------------------------------------------------------------------------


def _normalize_codex_status(status: Any) -> Optional[str]:
    return status if isinstance(status, str) and status in CODEX_RESPONSE_STATUSES else None


def _extract_codex_event_error(event: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    nested = event.get("error") if isinstance(event.get("error"), dict) else None
    code = event.get("code") if isinstance(event.get("code"), str) else (nested or {}).get("code")
    message = event.get("message") if isinstance(event.get("message"), str) else (nested or {}).get("message")
    return code, message


async def _map_codex_events(events, output: AssistantMessage, model: Model, on_provider_stream_event):
    async for event in events:
        if on_provider_stream_event is not None:
            maybe = on_provider_stream_event(event, model)
            if asyncio.iscoroutine(maybe):
                await maybe
        event_type = event.get("type")
        if not isinstance(event_type, str):
            continue

        if event_type == "error":
            code, message = _extract_codex_event_error(event)
            raise CodexApiError(f"Codex error: {message or code or json.dumps(event)}", code=code, payload=event)

        if event_type == "response.failed":
            response = event.get("response") or {}
            error = response.get("error") or {}
            raise CodexApiError(error.get("message") or "Codex response failed", code=error.get("code"), payload=event)

        if event_type in ("response.done", "response.completed", "response.incomplete"):
            response = event.get("response")
            if isinstance(response, dict):
                if isinstance(response.get("end_turn"), bool):
                    output.end_turn = response["end_turn"]
                response = {**response, "status": _normalize_codex_status(response.get("status"))}
                event = {**event, "type": "response.completed", "response": response}
            else:
                event = {**event, "type": "response.completed"}
            yield event
            return

        yield event


async def _parse_sse(response: httpx.Response, signal=None):
    buffer = ""
    async for chunk in response.aiter_text():
        if signal is not None and signal.aborted:
            raise ValueError("Request was aborted")
        buffer += chunk
        buffer = buffer.replace("\r\n", "\n")
        idx = buffer.find("\n\n")
        while idx != -1:
            block = buffer[:idx]
            buffer = buffer[idx + 2 :]
            data_lines = [line[5:].strip() for line in block.split("\n") if line.startswith("data:")]
            if data_lines:
                data = "\n".join(data_lines).strip()
                if data and data != "[DONE]":
                    try:
                        yield json.loads(data)
                    except ValueError as cause:
                        raise CodexProtocolError(
                            f"Invalid Codex SSE JSON: {format_thrown_value(cause)}", payload=data
                        ) from cause
            idx = buffer.find("\n\n")
    if buffer.strip():
        data_lines = [line[5:].strip() for line in buffer.split("\n") if line.startswith("data:")]
        data = "\n".join(data_lines).strip()
        if data and data != "[DONE]":
            try:
                yield json.loads(data)
            except ValueError as cause:
                raise CodexProtocolError(
                    f"Invalid Codex SSE JSON: {format_thrown_value(cause)}", payload=data
                ) from cause


async def _process_stream(
    response: httpx.Response,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    grammar_tool_input_properties: Dict[str, str],
    options: Optional[OpenAICodexResponsesOptions],
) -> None:
    await process_responses_stream(
        _map_codex_events(
            _parse_sse(response, options.signal if options else None),
            output,
            model,
            options.on_provider_stream_event if options else None,
        ),
        output,
        stream,
        model,
        OpenAIResponsesStreamOptions(
            service_tier=options.service_tier if options else None,
            grammar_tool_input_properties=grammar_tool_input_properties,
            resolve_service_tier=_resolve_codex_service_tier,
            apply_service_tier_pricing=lambda usage, tier: _apply_service_tier_pricing(usage, tier, model),
        ),
    )


def _parse_error_response(status: int, status_text: str, raw: str) -> tuple[str, Optional[str]]:
    message = raw or status_text or "Request failed"
    friendly_message: Optional[str] = None
    try:
        parsed = json.loads(raw)
        error = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(error, dict):
            code = error.get("code") or error.get("type") or ""
            if re.search(r"usage_limit_reached|usage_not_included|rate_limit_exceeded", code, re.IGNORECASE) or status == 429:
                plan = f" ({error['plan_type'].lower()} plan)" if error.get("plan_type") else ""
                resets_at = error.get("resets_at")
                mins = (
                    max(0, round((resets_at * 1000 - time.time() * 1000) / 60000))
                    if isinstance(resets_at, (int, float))
                    else None
                )
                when = f" Try again in ~{mins} min." if mins is not None else ""
                friendly_message = f"You have hit your ChatGPT usage limit{plan}.{when}".strip()
            message = error.get("message") or friendly_message or message
    except (ValueError, TypeError):
        pass
    return message, friendly_message


# ---------------------------------------------------------------------------
# Main stream function
# ---------------------------------------------------------------------------


def stream(
    model: Model,
    context: TranscriptContext,
    options: Optional[OpenAICodexResponsesOptions] = None,
) -> AssistantMessageEventStream:
    event_stream = AssistantMessageEventStream()
    compat = model.compat
    normalized_context = resolve_transcript(context, getattr(compat, "supports_mid_convo_system_messages", None))

    async def run() -> None:
        output = AssistantMessage(
            role="assistant",
            content=[],
            api="openai-codex-responses",
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

            account_id = extract_account_id(api_key)
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                get_declared_tools(normalized_context.messages),
                bool(getattr(compat, "supports_openai_grammar_tools", None)),
            )
            cache_session_id = (
                None if (options and options.cache_retention == "none") else (options.session_id if options else None)
            )
            codex_session_id = clamp_openai_prompt_cache_key(cache_session_id)
            body = _build_request_body(model, normalized_context, options, codex_session_id, grammar_tool_input_properties)
            if options and options.on_payload:
                next_body = options.on_payload(body, model)
                if asyncio.iscoroutine(next_body):
                    next_body = await next_body
                if next_body is not None:
                    body = next_body

            headers = _build_sse_headers(
                model.headers, options.headers if options else None, account_id, api_key, codex_session_id
            )
            body_json = json.dumps(body, separators=(",", ":"))
            compressed = _compress_request_body_zstd(body_json.encode("utf-8"))
            if compressed is not None:
                headers["content-encoding"] = "zstd"
            content: Any = compressed if compressed is not None else body_json

            # Fetch with retry logic for rate limits and transient errors.
            response: Optional[httpx.Response] = None
            last_error: Optional[BaseException] = None
            max_retries = options.max_retries if options and options.max_retries is not None else DEFAULT_MAX_RETRIES

            timeout_ms = options.timeout_ms if options and options.timeout_ms else None
            timeout = httpx.Timeout((timeout_ms or 600_000) / 1000, connect=60.0)
            client = httpx.AsyncClient(timeout=timeout)
            try:
                for attempt in range(max_retries + 1):
                    if options and options.signal and options.signal.aborted:
                        raise ValueError("Request was aborted")
                    try:
                        request = client.build_request(
                            "POST", resolve_codex_url(model.base_url), content=content, headers=headers
                        )
                        response = await client.send(request, stream=True)
                        if options and options.on_response:
                            maybe = options.on_response(
                                ProviderResponse(
                                    status=response.status_code, headers=headers_to_record(response.headers)
                                ),
                                model,
                            )
                            if asyncio.iscoroutine(maybe):
                                await maybe

                        if response.status_code < 400:
                            break

                        error_text = (await response.aread()).decode("utf-8", "replace")
                        await response.aclose()
                        if attempt < max_retries and _is_retryable_error(response.status_code, error_text):
                            retry_after = _get_retry_after_delay_ms(response.headers)
                            delay_ms = (
                                BASE_DELAY_MS * 2**attempt
                                if retry_after is None
                                else _validate_retry_delay_ms(retry_after, options)
                            )
                            await _sleep_ms(delay_ms, options.signal if options else None)
                            continue

                        message, friendly = _parse_error_response(
                            response.status_code, response.reason_phrase, error_text
                        )
                        raise ValueError(friendly or message)
                    except ValueError as error:
                        if str(error) == "Request was aborted":
                            raise
                        last_error = error
                        if (
                            attempt < max_retries
                            and not isinstance(error, RetryDelayExceededError)
                            and "usage limit" not in str(error)
                        ):
                            await _sleep_ms(BASE_DELAY_MS * 2**attempt, options.signal if options else None)
                            continue
                        raise
                    except (httpx.HTTPError, CodexApiError) as error:
                        last_error = error
                        if attempt < max_retries:
                            await _sleep_ms(BASE_DELAY_MS * 2**attempt, options.signal if options else None)
                            continue
                        raise

                if response is None or response.status_code >= 400:
                    raise last_error or ValueError("Failed after retries")

                try:
                    event_stream.push(StartEvent(partial=output))
                    await _process_stream(response, output, event_stream, model, grammar_tool_input_properties, options)
                finally:
                    await response.aclose()

                if options and options.signal and options.signal.aborted:
                    raise ValueError("Request was aborted")

                _assert_successful_output(output)
                event_stream.push(DoneEvent(reason=output.stop_reason, message=output))  # type: ignore[arg-type]
            finally:
                await client.aclose()
        except Exception as error:
            for block in output.content:
                if isinstance(block, ToolCall):
                    block.partial_json = None
            output.stop_reason = "aborted" if (options and options.signal and options.signal.aborted) else "error"
            output.error_message = format_provider_error(normalize_provider_error(error))
            event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))  # type: ignore[arg-type]
        finally:
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
    clamped = clamp_thinking_level(model, options.reasoning) if options and options.reasoning else None
    reasoning_effort = None if clamped == "off" else clamped

    return stream(
        model,
        context,
        OpenAICodexResponsesOptions(
            **base.model_dump(),
            tool_choice=options.tool_choice if options else None,
            reasoning_effort=reasoning_effort,
        ),
    )

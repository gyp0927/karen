"""Retry helpers, mirroring pi-ai's utils/provider-retry.ts and utils/retry.ts.

Two layers, exactly like pi-ai:

- :func:`retry_provider_request` reproduces the retry behavior of the
  OpenAI/Anthropic SDKs while making the backoff sleep interruptible:
  408/409/429/5xx and connection errors retry with exponential backoff;
  `retry-after` / `retry-after-ms` headers are honored but capped (60s by
  default).
- :func:`retry_assistant_call` wraps a whole assistant-producing call
  (considered failed only when its message comes back with
  ``stopReason: "error"``) in a caller-owned :class:`RetryPolicy`, classifying
  transient provider/transport errors via
  :func:`is_retryable_assistant_error`. Callers use it to restart a failed
  turn instead of failing the run.
"""

from __future__ import annotations

import dataclasses
import email.utils
import random
import re
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar

import httpx

from ..abort import AbortSignal, abortable_sleep
from ..errors import AbortError
from ..types import AssistantMessage, KarenBase

T = TypeVar("T")

DEFAULT_MAX_RETRY_DELAY_MS = 60_000


class ProviderHttpError(Exception):
    """HTTP error from a provider request, carrying status/headers for retry policy."""

    def __init__(self, message: str, *, status: Optional[int] = None, headers: Optional[httpx.Headers] = None):
        super().__init__(message)
        self.status = status
        self.headers = headers


def _header(headers: Optional[httpx.Headers], name: str) -> Optional[str]:
    if headers is None:
        return None
    return headers.get(name)


def is_retryable_error(error: BaseException) -> bool:
    if isinstance(error, ProviderHttpError):
        should_retry = _header(error.headers, "x-should-retry")
        if should_retry == "true":
            return True
        if should_retry == "false":
            return False
        if error.status is None:
            return True
        return error.status in (408, 409, 429) or error.status >= 500
    # Network-level failures (connect/read timeouts, connection resets).
    if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError, httpx.PoolTimeout)):
        return True
    return False


def _validate_server_retry_delay_ms(delay_ms: float, max_retry_delay_ms: Optional[int], provider_message: str) -> float:
    max_delay_ms = DEFAULT_MAX_RETRY_DELAY_MS if max_retry_delay_ms is None else max_retry_delay_ms
    if max_delay_ms > 0 and delay_ms > max_delay_ms:
        raise ProviderHttpError(
            f"Server requested {delay_ms / 1000:.0f}s retry delay (max: {max_delay_ms / 1000:.0f}s). {provider_message}"
        )
    return delay_ms


def _retry_delay_ms(error: BaseException, retry_index: int, max_retry_delay_ms: Optional[int]) -> float:
    headers = error.headers if isinstance(error, ProviderHttpError) else None

    retry_after_ms = _header(headers, "retry-after-ms")
    if retry_after_ms:
        try:
            return _validate_server_retry_delay_ms(float(retry_after_ms), max_retry_delay_ms, str(error))
        except ValueError:
            pass

    retry_after = _header(headers, "retry-after")
    if retry_after:
        try:
            delay_ms = float(retry_after) * 1000
        except ValueError:
            parsed = email.utils.parsedate_to_datetime(retry_after)
            delay_ms = parsed.timestamp() * 1000 - time.time() * 1000
        return _validate_server_retry_delay_ms(max(0.0, delay_ms), max_retry_delay_ms, str(error))

    exponential_delay = min(0.5 * 2**retry_index, 8) * 1000
    return exponential_delay * (1 - random.random() * 0.25)


async def retry_provider_request(
    request: Callable[[], Awaitable[T]],
    *,
    max_retries: Optional[int] = None,
    max_retry_delay_ms: Optional[int] = None,
    signal: Optional[AbortSignal] = None,
) -> T:
    retries = 0 if max_retries is None else max_retries
    retries_remaining = retries

    while True:
        try:
            return await request()
        except Exception as error:
            if signal is not None and signal.aborted:
                raise AbortError() from error
            if retries_remaining <= 0 or not is_retryable_error(error):
                raise
            retry_index = retries - retries_remaining
            retries_remaining -= 1
            await abortable_sleep(_retry_delay_ms(error, retry_index, max_retry_delay_ms) / 1000, signal)


# ---------------------------------------------------------------------------
# Assistant-call retry (pi-ai's utils/retry.ts)
# ---------------------------------------------------------------------------

_MAX_SAFE_INTEGER = 2**53 - 1


def _build_provider_error_pattern(patterns: list) -> "re.Pattern[str]":
    return re.compile("|".join(patterns), re.IGNORECASE)


_NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN = _build_provider_error_pattern(
    [
        # OpenCode Go/free-tier limits returned as 429 JSON error types by OpenCode's
        # Zen API. These are subscription/account limits, not transient throttles.
        "GoUsageLimitError",
        "FreeUsageLimitError",
        # OpenCode Go subscription-limit text asks users to enable available-balance
        # usage after rolling/weekly/monthly limits are reached.
        "Monthly usage limit reached",
        "available balance",
        # Generic quota/budget/billing exhaustion. `insufficient_quota` is OpenAI's
        # quota/billing error code; the other strings cover common gateway wording.
        "insufficient_quota",
        "out of budget",
        "quota exceeded",
        "billing",
    ]
)

_RETRYABLE_PROVIDER_ERROR_PATTERN = _build_provider_error_pattern(
    [
        # Generic provider load, HTTP status, and server-side transient failures.
        "overloaded",
        "currently experiencing high demand",
        "rate.?limit",
        "too many requests",
        "429",
        "500",
        "502",
        "503",
        "504",
        "520",
        "524",
        "service.?unavailable",
        "server.?error",
        "internal.?error",
        # Wrapper/provider text for transient upstream failures, including OpenRouter
        # "Provider returned error" responses.
        "provider.?returned.?error",
        "exceeded request buffer limit while retrying upstream",
        # Network, proxy, and fetch transport failures. This includes OpenAI Codex
        # raw-fetch failures such as "upstream connect", "connection refused", and
        # "reset before headers", plus OpenRouter connection drops.
        "network.?error",
        "connection.?error",
        "connection.?refused",
        "connection.?lost",
        "other side closed",
        "fetch failed",
        "getaddrinfo",
        "ENOTFOUND",
        "EAI_AGAIN",
        "upstream.?connect",
        "reset before headers",
        "socket hang up",
        "socket connection was closed",
        "timed? out",
        "timeout",
        "terminated",
        # WebSocket transports can report close/error text instead of HTTP/fetch text.
        "websocket.?closed",
        "websocket.?error",
        # Premature stream endings from SDKs and transports. Anthropic can throw
        # "stream ended without ..." and "Anthropic stream ended before message_stop";
        # Bedrock/Smithy can throw an HTTP/2 no-response error.
        "ended without",
        "stream ended before message_stop",
        "stream ended before a terminal response event",
        "http2 request did not get a response",
        # Provider-requested retry delay cap failures should flow through the outer
        # retry policy so callers can surface/abort the backoff.
        "retry delay",
        # Explicit retry guidance emitted mid-stream by OpenAI Responses and Bedrock
        # stream exceptions.
        "you can retry your request",
        "try your request again",
        "please retry your request",
        # gRPC based providers (e.g. NVIDIA NIM)
        "ResourceExhausted",
    ]
)


class RetryPolicy(KarenBase):
    """Assistant-call retry policy (pi-ai's `RetryPolicy`).

    Bounded attempts with exponential backoff (``base_delay_ms * 2^(attempt-1)``);
    ``max_agent_delay_ms`` caps each computed delay and defaults to 60 seconds.
    """

    enabled: bool
    #: Max retry attempts (0 = no retries). The initial call never counts as a retry.
    max_retries: int
    #: Base delay in ms. Per-attempt delay is `baseDelayMs * 2^(attempt-1)`.
    base_delay_ms: int
    #: Optional cap for agent-level retry delays in ms. Defaults to 60 seconds.
    max_agent_delay_ms: Optional[int] = None


#: Default cap for agent-level retry delays in ms.
DEFAULT_MAX_AGENT_RETRY_DELAY_MS = 60_000


def retry_delay_ms(policy: Any, attempt: int) -> int:
    """Backoff for one retry attempt (pi's `retryDelayMs`), in milliseconds."""
    delay = policy.base_delay_ms * 2 ** max(0, attempt - 1)
    safe_delay = delay if delay <= _MAX_SAFE_INTEGER else _MAX_SAFE_INTEGER
    cap = (
        policy.max_agent_delay_ms
        if policy.max_agent_delay_ms is not None
        else DEFAULT_MAX_AGENT_RETRY_DELAY_MS
    )
    return min(safe_delay, cap)


@dataclasses.dataclass
class RetryCallbacks:
    """Optional callbacks emitted by :func:`retry_assistant_call` around each retry.

    Each callback may be sync or async (pi allows ``void | Promise<void>``).
    """

    #: Emitted before the backoff sleep of each retry attempt (1-indexed).
    on_retry_scheduled: Optional[Callable[[int, int, int, str], Any]] = None
    #: Emitted after the backoff sleep, immediately before the retried call starts.
    on_retry_attempt_start: Optional[Callable[[], Any]] = None
    #: Emitted once when the loop ends: success if a later call completed normally.
    on_retry_finished: Optional[Callable[[bool, int, Optional[str]], Any]] = None


async def _invoke(callbacks: Optional[RetryCallbacks], name: str, *args: Any) -> None:
    """Fire one optional callback, awaiting it when it returns an awaitable."""
    callback = getattr(callbacks, name, None) if callbacks is not None else None
    if callback is None:
        return
    result = callback(*args)
    if hasattr(result, "__await__"):
        await result


async def retry_assistant_call(
    produce: Callable[[], Awaitable[AssistantMessage]],
    policy: Optional[RetryPolicy] = None,
    *,
    signal: Optional[AbortSignal] = None,
    callbacks: Optional[RetryCallbacks] = None,
) -> AssistantMessage:
    """Run one assistant-producing call with bounded retry on transient errors.

    Behavior (pi's `retryAssistantCall`):

    - A successful response is returned immediately. Aborts are terminal and are
      never retried, but reported as unsuccessful if they happen after a retry was
      scheduled. Aborts during the backoff sleep are normalized to an aborted
      ``AssistantMessage`` too, so callers do not need to care when cancellation
      happened.
    - A non-retryable error (per :func:`is_retryable_assistant_error`, including
      quota/billing exhaustion) is returned immediately so deterministic errors
      fail fast.
    - Otherwise retries up to ``policy.max_retries`` times with exponential
      backoff, emitting ``on_retry_scheduled`` before each sleep,
      ``on_retry_attempt_start`` after each sleep before the retried call starts,
      and ``on_retry_finished`` once at the end (whether the loop ends in success,
      exhausted retries, or an aborted backoff).

    When ``policy`` is None or disabled, the first response is returned unchanged.
    """
    max_attempts = policy.max_retries if policy is not None and policy.enabled else 0

    attempt = 0
    last_retry: Optional[tuple] = None  # (attempt, error_message)
    while True:
        response = await produce()

        # Abort: terminal but not successful. Never retry an aborted message.
        if response.stop_reason == "aborted":
            if last_retry is not None:
                await _invoke(callbacks, "on_retry_finished", False, last_retry[0])
            return response

        # Success: non-error, non-abort responses return as-is.
        if response.stop_reason != "error":
            if last_retry is not None:
                await _invoke(callbacks, "on_retry_finished", True, last_retry[0])
            return response

        # Non-retryable, or budget exhausted: return the final error message.
        if attempt >= max_attempts or not is_retryable_assistant_error(response):
            if last_retry is not None:
                await _invoke(
                    callbacks, "on_retry_finished", False, last_retry[0], response.error_message
                )
            return response

        attempt += 1
        last_retry = (attempt, response.error_message or "Unknown error")
        delay_ms = retry_delay_ms(policy, attempt)
        await _invoke(
            callbacks,
            "on_retry_scheduled",
            attempt,
            max_attempts,
            delay_ms,
            last_retry[1],
        )

        # Normalize aborts during retry backoff to the same AssistantMessage shape as
        # provider stream aborts, so callers do not need to care when cancellation happened.
        try:
            await abortable_sleep(delay_ms / 1000, signal)
        except BaseException as error:  # noqa: BLE001 - re-raised unless it is the abort
            await _invoke(callbacks, "on_retry_finished", False, attempt, last_retry[1])
            if isinstance(error, AbortError):
                return response.model_copy(update={"stop_reason": "aborted", "error_message": None})
            raise
        await _invoke(callbacks, "on_retry_attempt_start")


def is_retryable_assistant_error(message: AssistantMessage) -> bool:
    """Classify whether a failed assistant message looks like a transient provider
    or transport error, so callers can decide if the last assistant turn should
    be restarted (pi's `isRetryableAssistantError`).

    This does not implement retry policy. Callers should first handle context
    overflow separately, then apply their own retry budget, backoff, and reporting
    before restarting the assistant turn.
    """
    stop_reason = getattr(message, "stop_reason", None)
    error_message = getattr(message, "error_message", None)
    if stop_reason != "error" or not error_message:
        return False
    if _NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN.search(error_message):
        return False
    return _RETRYABLE_PROVIDER_ERROR_PATTERN.search(error_message) is not None

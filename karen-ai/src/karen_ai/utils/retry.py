"""Provider request retry, mirroring pi-ai's utils/provider-retry.ts.

Reproduces the retry behavior of the OpenAI/Anthropic SDKs while making the
backoff sleep interruptible: 408/409/429/5xx and connection errors retry with
exponential backoff; `retry-after` / `retry-after-ms` headers are honored but
capped (60s by default).
"""

from __future__ import annotations

import email.utils
import random
import time
from typing import Awaitable, Callable, Optional, TypeVar

import httpx

from ..abort import AbortSignal, abortable_sleep
from ..errors import AbortError

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

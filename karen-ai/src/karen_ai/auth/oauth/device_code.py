"""RFC 8628 device-code polling, mirroring auth/oauth/device-code.ts."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Literal, Optional, TypeVar

from ...abort import AbortSignal, abortable_sleep
from ...errors import AbortError

T = TypeVar("T")

CANCEL_MESSAGE = "Login cancelled"
TIMEOUT_MESSAGE = "Device flow timed out"
SLOW_DOWN_TIMEOUT_MESSAGE = (
    "Device flow timed out after one or more slow_down responses. "
    "This is often caused by clock drift in WSL or VM environments. "
    "Please sync or restart the VM clock and try again."
)
MINIMUM_INTERVAL_MS = 1000
# RFC 8628 section 3.2: if the authorization server omits `interval`, the client must use 5 seconds.
DEFAULT_POLL_INTERVAL_SECONDS = 5
# RFC 8628 section 3.5: `slow_down` means the polling interval must increase by 5 seconds.
SLOW_DOWN_INTERVAL_INCREMENT_MS = 5000


@dataclass
class DeviceCodePollResult(Generic[T]):
    """One poll attempt's outcome; construct via the poll_* helpers."""

    status: Literal["pending", "slow_down", "failed", "complete"]
    value: Optional[T] = None
    message: Optional[str] = None
    interval_seconds: Optional[float] = None


def poll_complete(value: T) -> DeviceCodePollResult[T]:
    return DeviceCodePollResult(status="complete", value=value)


def poll_pending() -> DeviceCodePollResult[T]:
    return DeviceCodePollResult(status="pending")


def poll_slow_down(interval_seconds: Optional[float] = None) -> DeviceCodePollResult[T]:
    return DeviceCodePollResult(status="slow_down", interval_seconds=interval_seconds)


def poll_failed(message: str) -> DeviceCodePollResult[T]:
    return DeviceCodePollResult(status="failed", message=message)


async def _sleep_ms(ms: float, signal: AbortSignal) -> None:
    try:
        await abortable_sleep(ms / 1000, signal)
    except AbortError as error:
        raise AbortError(CANCEL_MESSAGE) from error


async def poll_oauth_device_code_flow(
    *,
    poll: Callable[[], Awaitable[DeviceCodePollResult[T]]],
    signal: AbortSignal,
    interval_seconds: Optional[float] = None,
    expires_in_seconds: Optional[float] = None,
    wait_before_first_poll: bool = False,
) -> T:
    """Poll a device-code token endpoint until complete, failed, or timed out."""
    start = time.monotonic()
    deadline = start + expires_in_seconds if expires_in_seconds is not None else math.inf
    interval_ms = max(
        MINIMUM_INTERVAL_MS,
        math.floor((interval_seconds if interval_seconds is not None else DEFAULT_POLL_INTERVAL_SECONDS) * 1000),
    )

    slow_down_responses = 0
    if wait_before_first_poll:
        remaining_ms = (deadline - time.monotonic()) * 1000
        if remaining_ms > 0:
            await _sleep_ms(min(interval_ms, remaining_ms), signal)

    while time.monotonic() < deadline:
        if signal.aborted:
            raise AbortError(CANCEL_MESSAGE)

        result = await poll()
        if result.status == "complete":
            return result.value  # type: ignore[return-value]
        if result.status == "failed":
            raise ValueError(result.message or "Device flow failed")
        if result.status == "slow_down":
            slow_down_responses += 1
            # Use the server-provided interval when given (GitHub reports the new
            # required minimum in `interval`); trusting only a client-tracked value
            # risks polling early forever under WSL/VM clock drift. Otherwise apply
            # RFC 8628 section 3.5: increase by 5 seconds.
            if result.interval_seconds is not None and result.interval_seconds > 0:
                interval_ms = max(MINIMUM_INTERVAL_MS, math.floor(result.interval_seconds * 1000))
            else:
                interval_ms = max(MINIMUM_INTERVAL_MS, interval_ms + SLOW_DOWN_INTERVAL_INCREMENT_MS)

        remaining_ms = (deadline - time.monotonic()) * 1000
        if remaining_ms <= 0:
            break

        await _sleep_ms(min(interval_ms, remaining_ms), signal)

    raise ValueError(SLOW_DOWN_TIMEOUT_MESSAGE if slow_down_responses > 0 else TIMEOUT_MESSAGE)

"""Port of pi-ai's `utils/retry.ts` assistant-call layer.

The provider-request layer (`retry_provider_request`) is exercised through the
adapter tests; this suite covers the classifier, the backoff computation and
`retry_assistant_call`.
"""

import asyncio

import pytest

from karen_ai import AbortController
from karen_ai.types import AssistantMessage
from karen_ai.utils.retry import (
    DEFAULT_MAX_AGENT_RETRY_DELAY_MS,
    RetryCallbacks,
    RetryPolicy,
    is_retryable_assistant_error,
    retry_assistant_call,
    retry_delay_ms,
)


def error_message(text: str, stop_reason: str = "error") -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api="openai-completions",
        provider="test-provider",
        model="test-model",
        stop_reason=stop_reason,
        error_message=text if stop_reason == "error" else None,
        timestamp=1,
    )


def ok_message(text: str = "hello") -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api="openai-completions",
        provider="test-provider",
        model="test-model",
        stop_reason="stop",
        timestamp=2,
    )


def policy(**overrides) -> RetryPolicy:
    values = {"enabled": True, "max_retries": 3, "base_delay_ms": 1, "max_agent_delay_ms": 10}
    values.update(overrides)
    return RetryPolicy(**values)


# ---------------------------------------------------------------------------
# retryDelayMs
# ---------------------------------------------------------------------------


def test_retry_delay_doubles_each_attempt():
    settings = RetryPolicy(enabled=True, max_retries=5, base_delay_ms=2000)
    assert [retry_delay_ms(settings, attempt) for attempt in range(1, 5)] == [
        2000,
        4000,
        8000,
        16000,
    ]


def test_retry_delay_is_capped_by_max_agent_delay():
    settings = RetryPolicy(enabled=True, max_retries=5, base_delay_ms=2000, max_agent_delay_ms=5000)
    assert retry_delay_ms(settings, 4) == 5000


def test_retry_delay_defaults_the_cap_to_sixty_seconds():
    settings = RetryPolicy(enabled=True, max_retries=5, base_delay_ms=1000)
    assert retry_delay_ms(settings, 20) == DEFAULT_MAX_AGENT_RETRY_DELAY_MS


def test_retry_delay_clamps_overflowing_delays_to_the_safe_integer_range():
    settings = RetryPolicy(enabled=True, max_retries=5, base_delay_ms=2, max_agent_delay_ms=2**60)
    assert retry_delay_ms(settings, 100) == 2**53 - 1


def test_retry_delay_treats_attempt_zero_as_the_first_attempt():
    settings = RetryPolicy(enabled=True, max_retries=5, base_delay_ms=250)
    assert retry_delay_ms(settings, 0) == 250


# ---------------------------------------------------------------------------
# isRetryableAssistantError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Error 503 Service Unavailable",
        "overloaded_error: Overloaded",
        "Rate limit reached for requests",
        "Too many requests, please slow down",
        '{"error":{"type":"server_error"}}',
        "Provider returned error",
        "fetch failed",
        "socket hang up",
        "The operation timed out",
        "stream ended before message_stop",
        "ResourceExhausted: quota refresh",
        "Upstream connect error or disconnect/reset before headers",
        "You can retry your request",
    ],
)
def test_transient_errors_are_retryable(text):
    assert is_retryable_assistant_error(error_message(text)) is True


@pytest.mark.parametrize(
    "text",
    [
        "GoUsageLimitError: usage limit reached",
        "FreeUsageLimitError",
        "Monthly usage limit reached",
        "Insufficient available balance",
        "insufficient_quota",
        "You are out of budget",
        "quota exceeded",
        "Billing hard limit has been reached",
        "invalid api key",
    ],
)
def test_account_and_billing_limits_are_not_retryable(text):
    assert is_retryable_assistant_error(error_message(text)) is False


def test_non_error_and_message_less_failures_are_not_retryable():
    assert is_retryable_assistant_error(ok_message()) is False
    assert is_retryable_assistant_error(error_message("", stop_reason="error")) is False
    assert is_retryable_assistant_error(error_message("503", stop_reason="aborted")) is False


# ---------------------------------------------------------------------------
# retryAssistantCall
# ---------------------------------------------------------------------------


async def test_returns_the_first_response_when_it_succeeds():
    calls = []
    finished = []

    async def produce():
        calls.append(1)
        return ok_message()

    response = await retry_assistant_call(
        produce, policy(), callbacks=RetryCallbacks(on_retry_finished=lambda *a: finished.append(a))
    )
    assert response.stop_reason == "stop"
    assert calls == [1]
    assert finished == []  # no retry was ever scheduled


async def test_retries_transient_errors_until_success():
    responses = [error_message("Error 503"), error_message("rate limit exceeded"), ok_message()]
    scheduled = []
    started = []
    finished = []

    async def produce():
        return responses.pop(0)

    response = await retry_assistant_call(
        produce,
        policy(max_retries=3, base_delay_ms=1, max_agent_delay_ms=1),
        callbacks=RetryCallbacks(
            on_retry_scheduled=lambda *args: scheduled.append(args),
            on_retry_attempt_start=lambda: started.append(1),
            on_retry_finished=lambda *args: finished.append(args),
        ),
    )
    assert response.stop_reason == "stop"
    assert scheduled == [(1, 3, 1, "Error 503"), (2, 3, 1, "rate limit exceeded")]
    assert started == [1, 1]
    assert finished == [(True, 2)]


async def test_non_retryable_errors_return_immediately():
    calls = []

    async def produce():
        calls.append(1)
        return error_message("insufficient_quota")

    response = await retry_assistant_call(produce, policy())
    assert response.error_message == "insufficient_quota"
    assert calls == [1]


async def test_budget_exhaustion_returns_the_last_error():
    calls = []
    finished = []

    async def produce():
        calls.append(1)
        return error_message(f"Error 500 #{len(calls)}")

    response = await retry_assistant_call(
        produce,
        policy(max_retries=2, base_delay_ms=1, max_agent_delay_ms=1),
        callbacks=RetryCallbacks(on_retry_finished=lambda *args: finished.append(args)),
    )
    assert response.error_message == "Error 500 #3"
    assert len(calls) == 3  # the initial call plus max_retries
    assert finished == [(False, 2, "Error 500 #3")]


async def test_disabled_policy_returns_the_first_response():
    calls = []
    finished = []

    async def produce():
        calls.append(1)
        return error_message("Error 503")

    response = await retry_assistant_call(
        produce,
        policy(enabled=False),
        callbacks=RetryCallbacks(on_retry_finished=lambda *args: finished.append(args)),
    )
    assert response.stop_reason == "error"
    assert calls == [1]
    assert finished == []


async def test_aborted_responses_are_never_retried():
    controller = AbortController()
    controller.abort()
    calls = []

    async def produce():
        calls.append(1)
        return error_message("", stop_reason="aborted")

    response = await retry_assistant_call(produce, policy(), signal=controller.signal)
    assert response.stop_reason == "aborted"
    assert calls == [1]


async def test_abort_during_backoff_returns_an_aborted_message():
    controller = AbortController()
    finished = []
    calls = []

    async def produce():
        calls.append(1)
        return error_message("Error 503")

    async def abort_soon():
        await asyncio.sleep(0.01)
        controller.abort()

    aborter = asyncio.ensure_future(abort_soon())
    response = await retry_assistant_call(
        produce,
        policy(max_retries=3, base_delay_ms=30_000, max_agent_delay_ms=30_000),
        signal=controller.signal,
        callbacks=RetryCallbacks(on_retry_finished=lambda *args: finished.append(args)),
    )
    await aborter
    assert response.stop_reason == "aborted"
    assert response.error_message is None  # pi strips the error message on a backoff abort
    assert calls == [1]  # the sleep was interrupted instead of waiting 30s
    assert finished == [(False, 1, "Error 503")]


async def test_async_callbacks_are_awaited():
    order = []

    async def on_scheduled(*args):
        await asyncio.sleep(0)
        order.append(("scheduled", args[0]))

    async def on_start():
        order.append(("start",))

    async def on_finished(*args):
        order.append(("finished", args[0]))

    responses = [error_message("Error 503"), ok_message()]

    async def produce():
        return responses.pop(0)

    await retry_assistant_call(
        produce,
        policy(base_delay_ms=1, max_agent_delay_ms=1),
        callbacks=RetryCallbacks(
            on_retry_scheduled=on_scheduled, on_retry_attempt_start=on_start, on_retry_finished=on_finished
        ),
    )
    assert order == [("scheduled", 1), ("start",), ("finished", True)]


async def test_without_a_policy_the_first_response_is_returned_unchanged():
    calls = []

    async def produce():
        calls.append(1)
        return error_message("Error 503")

    response = await retry_assistant_call(produce)
    assert response.error_message == "Error 503"
    assert calls == [1]

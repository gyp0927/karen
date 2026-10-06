"""Port of pi-ai's `test/overflow.test.ts` — context-overflow detection.

The sibling `context-overflow.test.ts` is a live-API suite (it overflows real
provider endpoints) and is not ported; karen verifies live behavior through the
examples instead.
"""

from karen_ai.types import AssistantMessage, Usage
from karen_ai.utils.overflow import is_context_overflow, is_recoverable_length


def create_error_message(error_message: str, provider: str = "ollama") -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api="openai-completions",
        provider=provider,
        model="qwen3.5:35b",
        stop_reason="error",
        error_message=error_message,
        timestamp=1,
    )


def create_length_stop_message(
    input: int,
    cache_read: int,
    output: int,
    cache_write: int = 0,
    api: str = "openai-completions",
    provider: str = "test-provider",
    model: str = "test-model",
) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=api,
        provider=provider,
        model=model,
        usage=Usage(
            input=input,
            output=output,
            cache_read=cache_read,
            cache_write=cache_write,
            total_tokens=input + cache_read + cache_write + output,
        ),
        stop_reason="length",
        timestamp=1,
    )


def test_detects_explicit_ollama_prompt_too_long_errors():
    message = create_error_message("400 `prompt too long; exceeded max context length by 100918 tokens`")
    assert is_context_overflow(message, 32768) is True


def test_detects_zai_prompt_too_long_errors():
    # Regression for pi#9805.
    message = create_error_message('400 {"code":"1261","message":"Prompt too long"}', "zai")
    assert is_context_overflow(message, 1048576) is True


def test_detects_together_ai_context_length_errors():
    message = create_error_message(
        "400 The input (516368 tokens) is longer than the model's context length (262144 tokens)."
    )
    assert is_context_overflow(message, 262144) is True


def test_detects_litellm_wrapped_openai_maximum_context_length_errors():
    message = create_error_message(
        "Error: 503 litellm.ServiceUnavailableError: litellm.MidStreamFallbackError: "
        "litellm.APIConnectionError: APIConnectionError: OpenAIException - Requested token "
        "count exceeds the model's maximum context length of 131072 tokens."
    )
    assert is_context_overflow(message, 131072) is True


def test_detects_openai_compatible_parenthesized_maximum_context_length_errors():
    message = create_error_message("Error: 400 Input length (265330) exceeds model's maximum context length (262144).")
    assert is_context_overflow(message, 262144) is True


def test_detects_openrouter_poolside_maximum_allowed_input_length_errors():
    message = create_error_message(
        "Provider returned error: Input length 131393 exceeds the maximum allowed input length of 131040 tokens."
    )
    assert is_context_overflow(message, 131072) is True


def test_detects_ds4_configured_context_size_errors():
    message = create_error_message("400 Prompt has 256468 tokens, but the configured context size is 256000 tokens")
    assert is_context_overflow(message, 256000) is True

    comma_message = create_error_message(
        "Prompt has 5,958,968 tokens, but the configured context size is 256,000 tokens"
    )
    assert is_context_overflow(comma_message, 256000) is True


def test_does_not_treat_generic_non_overflow_ollama_errors_as_overflow():
    message = create_error_message("500 `model runner crashed unexpectedly`")
    assert is_context_overflow(message, 32768) is False


def test_only_treats_bodyless_400_and_413_as_overflow_for_cerebras():
    # Regression for pi#9482.
    for error_message in ["400 status code (no body)", "413 status code (no body)"]:
        assert is_context_overflow(create_error_message(error_message, "cerebras"), 131072) is True
        assert is_context_overflow(create_error_message(error_message, "opencode-go"), 1000000) is False


def test_does_not_treat_bedrock_throttling_too_many_tokens_as_overflow():
    # Bedrock returns this for HTTP 429 rate limiting, NOT context overflow.
    # formatBedrockError uses a human-readable prefix for ThrottlingException.
    message = create_error_message("Throttling error: Too many tokens, please wait before trying again.")
    assert is_context_overflow(message, 200000) is False


def test_does_not_treat_bedrock_service_unavailable_as_overflow():
    message = create_error_message("Service unavailable: The service is temporarily unavailable.")
    assert is_context_overflow(message, 200000) is False


def test_does_not_treat_generic_rate_limit_errors_as_overflow():
    message = create_error_message("Rate limit exceeded, please retry after 30 seconds.")
    assert is_context_overflow(message, 200000) is False


def test_does_not_treat_http_429_style_errors_as_overflow():
    message = create_error_message("Too many requests. Please slow down.")
    assert is_context_overflow(message, 200000) is False


def test_detects_xiaomi_style_overflow_length_stop_zero_output_filled_context():
    message = create_length_stop_message(
        input=58, cache_read=1048512, output=0, provider="xiaomi", model="mimo-v2.5-pro"
    )
    assert is_context_overflow(message, 1048576) is True


def test_treats_length_stop_below_desired_output_limit_as_recoverable():
    message = create_length_stop_message(
        input=3,
        cache_read=253584,
        cache_write=25554,
        output=16,
        api="openai-responses",
        provider="openai",
        model="gpt-5.6-sol",
    )
    assert is_recoverable_length(message, 128000) is True


def test_does_not_recover_length_stop_that_reached_desired_output_limit():
    message = create_length_stop_message(input=4062, cache_read=0, output=1024)
    assert is_recoverable_length(message, 1024) is False


def test_treats_zero_output_length_stops_as_recoverable_without_context_metadata():
    message = create_length_stop_message(input=100, cache_read=0, output=0)
    assert is_recoverable_length(message, 128000) is True


def test_does_not_treat_normal_length_stops_with_output_as_context_overflow():
    message = create_length_stop_message(input=1000, cache_read=0, output=4096)
    assert is_context_overflow(message, 200000) is False


def test_does_not_treat_zero_output_length_stops_far_below_context_as_context_overflow():
    message = create_length_stop_message(input=100, cache_read=0, output=0)
    assert is_context_overflow(message, 200000) is False


def test_silent_overflow_stop_with_usage_above_context_window():
    message = AssistantMessage(
        content=[],
        api="openai-completions",
        provider="zai",
        model="glm-5.2",
        usage=Usage(input=900000, cache_read=200000, output=500, total_tokens=1100500),
        stop_reason="stop",
        timestamp=1,
    )
    assert is_context_overflow(message, 1048576) is True
    # without the context window, silent overflow is undetectable
    assert is_context_overflow(message) is False

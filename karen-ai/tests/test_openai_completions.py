"""Tests for the OpenAI Chat Completions adapter against a mocked HTTP layer."""

import asyncio
import json

import httpx
import pytest
import respx

from karen_ai import Context, Tool, UserMessage, normalize_context
from karen_ai.api import openai_completions
from karen_ai.providers import faux_model

BASE_URL = "https://api.test/v1"


def _model(**overrides):
    model = faux_model(provider="testprov", api="openai-completions")
    model.base_url = BASE_URL
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


def _sse(*events: str) -> httpx.Response:
    body = "".join(f"data: {event}\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def _chunk(*, id="chatcmpl-1", content=None, tool_calls=None, finish=None, usage=None, reasoning=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    chunk = {"id": id, "object": "chat.completion.chunk", "model": "faux-1", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        chunk["usage"] = usage
    return json.dumps(chunk)


def test_text_stream():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/chat/completions").mock(
                return_value=_sse(
                    _chunk(content="Hello"),
                    _chunk(content=", world"),
                    _chunk(finish="stop", usage={"prompt_tokens": 10, "completion_tokens": 5}),
                )
            )
            stream = openai_completions.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                openai_completions.OpenAICompletionsOptions(api_key="sk-test"),
            )
            events = [e async for e in stream]
            message = await stream.result()

        text = "".join(e.delta for e in events if e.type == "text_delta")
        assert text == "Hello, world"
        assert message.stop_reason == "stop"
        assert message.usage.input == 10
        assert message.usage.output == 5
        assert message.response_id == "chatcmpl-1"

    asyncio.run(main())


def test_tool_call_stream():
    async def main():
        model = _model()
        with respx.mock:
            route = respx.post(f"{BASE_URL}/chat/completions").mock(
                return_value=_sse(
                    _chunk(tool_calls=[{"index": 0, "id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"q": "hel'}}]),
                    _chunk(tool_calls=[{"index": 0, "function": {"arguments": 'lo world"}'}}]),
                    _chunk(finish="tool_calls", usage={"prompt_tokens": 7, "completion_tokens": 3}),
                )
            )
            stream = openai_completions.stream_simple(
                model,
                normalize_context(
                    Context(
                        tools=[Tool(name="search", description="s", parameters={"type": "object"})],
                        messages=[UserMessage(content="hi", timestamp=1)],
                    )
                ),
                __import__("karen_ai").SimpleStreamOptions(api_key="sk-test"),
            )
            events = [e async for e in stream]
            message = await stream.result()

        assert message.stop_reason == "toolUse"
        tool_call = message.content[0]
        assert tool_call.type == "toolCall"
        assert tool_call.id == "call_1"
        assert tool_call.name == "search"
        assert tool_call.arguments == {"q": "hello world"}

    asyncio.run(main())


def test_stream_simple_requires_auth():
    model = _model()
    with pytest.raises(ValueError, match="No API key"):
        openai_completions.stream_simple(
            model,
            normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
            None,
        )


def test_reasoning_stream_maps_to_thinking_block():
    async def main():
        model = _model()
        model.reasoning = True
        with respx.mock:
            respx.post(f"{BASE_URL}/chat/completions").mock(
                return_value=_sse(
                    _chunk(reasoning="thinking "),
                    _chunk(reasoning="hard"),
                    _chunk(content="answer"),
                    _chunk(finish="stop"),
                )
            )
            stream = openai_completions.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                openai_completions.OpenAICompletionsOptions(api_key="sk-test"),
            )
            [e async for e in stream]
            message = await stream.result()

        thinking = [b for b in message.content if b.type == "thinking"]
        text = [b for b in message.content if b.type == "text"]
        assert thinking and thinking[0].thinking == "thinking hard"
        assert thinking[0].thinking_signature == "reasoning_content"
        assert text and text[0].text == "answer"

    asyncio.run(main())


def test_cache_tokens_split_from_input():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/chat/completions").mock(
                return_value=_sse(
                    _chunk(
                        finish="stop",
                        usage={
                            "prompt_tokens": 100,
                            "completion_tokens": 10,
                            "prompt_tokens_details": {"cached_tokens": 40},
                        },
                    ),
                )
            )
            stream = openai_completions.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                openai_completions.OpenAICompletionsOptions(api_key="sk-test"),
            )
            [e async for e in stream]
            message = await stream.result()

        assert message.usage.cache_read == 40
        assert message.usage.input == 60
        assert message.usage.total_tokens == 110

    asyncio.run(main())


def test_http_error_terminates_stream_with_error_event():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/chat/completions").mock(
                return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
            )
            stream = openai_completions.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                openai_completions.OpenAICompletionsOptions(api_key="sk-bad", max_retries=0),
            )
            events = [e async for e in stream]
            message = await stream.result()

        assert events[-1].type == "error"
        assert message.stop_reason == "error"
        assert "401" in message.error_message

    asyncio.run(main())


def test_request_payload_shape():
    """Params sent to the endpoint follow Chat Completions conventions."""
    async def main():
        model = _model()
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            captured["headers"] = request.headers
            return _sse(_chunk(content="ok", finish="stop"))

        with respx.mock:
            respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=side_effect)
            stream = openai_completions.stream_simple(
                model,
                normalize_context(
                    Context(
                        system_prompt="Be brief.",
                        tools=[Tool(name="search", description="s", parameters={"type": "object"})],
                        messages=[UserMessage(content="hi", timestamp=1)],
                    )
                ),
                __import__("karen_ai").SimpleStreamOptions(api_key="sk-test", temperature=0.5, max_tokens=100),
            )
            [e async for e in stream]
            await stream.result()

        payload = captured["payload"]
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        assert payload["temperature"] == 0.5
        # testprov is a "non-standard" provider (unknown URL): max_completion_tokens default
        assert payload.get("max_completion_tokens") == 100 or payload.get("max_tokens") == 100
        assert payload["messages"][0] == {"role": "system", "content": "Be brief."}
        assert payload["messages"][1] == {"role": "user", "content": "hi"}
        assert payload["tools"][0]["function"]["name"] == "search"
        assert captured["headers"]["authorization"] == "Bearer sk-test"

    asyncio.run(main())


def test_deepseek_thinking_format_params():
    async def main():
        model = _model()
        model.provider = "deepseek"
        model.base_url = "https://api.deepseek.com"
        model.reasoning = True
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return _sse(_chunk(finish="stop"))

        with respx.mock:
            respx.post("https://api.deepseek.com/chat/completions").mock(side_effect=side_effect)
            stream = openai_completions.stream_simple(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                __import__("karen_ai").SimpleStreamOptions(api_key="sk-test", reasoning="medium"),
            )
            [e async for e in stream]
            await stream.result()

        payload = captured["payload"]
        assert payload["thinking"] == {"type": "enabled"}
        # DeepSeek uses the legacy max_tokens field.
        assert "max_completion_tokens" not in payload

    asyncio.run(main())

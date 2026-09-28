"""Tests for the Mistral Conversations adapter against a mocked HTTP layer."""

import asyncio
import json

import httpx
import pytest
import respx

import karen_ai
from karen_ai import (
    AssistantMessage,
    Context,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    normalize_context,
)
from karen_ai.api import mistral_conversations
from karen_ai.api.mistral_conversations import (
    MistralToolCallIdNormalizer,
    derive_mistral_tool_call_id,
    to_mistral_wire_payload,
)
from karen_ai.providers import faux_model

BASE_URL = "https://api.mistral.test"


def _model(**overrides):
    model = faux_model(provider="mistral", api="mistral-conversations")
    model.base_url = BASE_URL
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


def _sse(*payloads) -> httpx.Response:
    body = "".join(f"data: {p if isinstance(p, str) else json.dumps(p)}\n\n" for p in payloads)
    body += "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def _chunk(*, id="cmpl-1", content=None, tool_calls=None, finish=None, usage=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    chunk = {"id": id, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def test_derive_mistral_tool_call_id():
    # 9-char alphanumeric IDs pass through.
    assert derive_mistral_tool_call_id("abc123XYZ", 0) == "abc123XYZ"
    # Longer/foreign IDs become deterministic 9-char alphanumeric hashes.
    derived = derive_mistral_tool_call_id("toolu_01ABCdef", 0)
    assert len(derived) == 9
    assert derived.isalnum()
    assert derive_mistral_tool_call_id("toolu_01ABCdef", 0) == derived
    # Later attempts differ (collision handling).
    assert derive_mistral_tool_call_id("toolu_01ABCdef", 1) != derived


def test_normalizer_is_stable_and_collision_free():
    normalize = MistralToolCallIdNormalizer()
    first = normalize("call-one")
    assert normalize("call-one") == first
    second = normalize("call-two")
    assert first != second


def test_text_stream():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/chat/completions").mock(
                return_value=_sse(
                    _chunk(content="Hello"),
                    _chunk(content="!"),
                    _chunk(
                        finish="stop",
                        usage={"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
                    ),
                )
            )
            stream = mistral_conversations.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                mistral_conversations.MistralOptions(api_key="m-test"),
            )
            events = [e async for e in stream]
            message = await stream.result()

        text = "".join(e.delta for e in events if e.type == "text_delta")
        assert text == "Hello!"
        assert message.stop_reason == "stop"
        assert message.usage.input == 12
        assert message.usage.output == 4
        assert message.response_id == "cmpl-1"

    asyncio.run(main())


def test_thinking_chunks_and_tool_calls():
    async def main():
        model = _model()
        model.reasoning = True
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/chat/completions").mock(
                return_value=_sse(
                    _chunk(content=[{"type": "thinking", "thinking": [{"text": "pondering"}]}]),
                    _chunk(
                        tool_calls=[
                            {"id": "abc123XYZ", "index": 0, "function": {"name": "search", "arguments": '{"q": "pa'}}
                        ]
                    ),
                    _chunk(tool_calls=[{"index": 0, "function": {"name": "search", "arguments": 'ris"}'}}]),
                    _chunk(finish="tool_calls", usage={"prompt_tokens": 7, "completion_tokens": 3}),
                )
            )
            stream = mistral_conversations.stream_simple(
                model,
                normalize_context(
                    Context(
                        tools=[Tool(name="search", description="s", parameters={"type": "object"})],
                        messages=[UserMessage(content="weather?", timestamp=1)],
                    )
                ),
                karen_ai.SimpleStreamOptions(api_key="m-test", reasoning="high"),
            )
            [e async for e in stream]
            message = await stream.result()

        assert message.stop_reason == "toolUse"
        thinking = message.content[0]
        assert thinking.type == "thinking"
        assert thinking.thinking == "pondering"
        tool_call = message.content[1]
        assert tool_call.type == "toolCall"
        assert tool_call.id == "abc123XYZ"
        assert tool_call.arguments == {"q": "paris"}

    asyncio.run(main())


def test_cached_prompt_tokens_split_from_input():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/chat/completions").mock(
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
            stream = mistral_conversations.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                mistral_conversations.MistralOptions(api_key="m-test"),
            )
            [e async for e in stream]
            message = await stream.result()

        assert message.usage.cache_read == 40
        assert message.usage.input == 60

    asyncio.run(main())


def test_request_payload_shape_and_wire_remap():
    async def main():
        model = _model()
        model.reasoning = True
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            captured["headers"] = request.headers
            return _sse(_chunk(content="ok", finish="stop"))

        with respx.mock:
            respx.post(f"{BASE_URL}/v1/chat/completions").mock(side_effect=side_effect)
            stream = mistral_conversations.stream_simple(
                model,
                normalize_context(
                    Context(
                        system_prompt="Be brief.",
                        tools=[Tool(name="search", description="s", parameters={"type": "object"})],
                        messages=[UserMessage(content="hi", timestamp=1)],
                    )
                ),
                karen_ai.SimpleStreamOptions(
                    api_key="m-test", reasoning="high", max_tokens=50, session_id="sess-1"
                ),
            )
            [e async for e in stream]
            await stream.result()

        payload = captured["payload"]
        assert payload["model"] == "faux-1"
        assert payload["stream"] is True
        # camelCase payload keys are remapped to snake_case on the wire.
        assert payload["max_tokens"] == 50
        assert "maxTokens" not in payload
        # Prompt-mode reasoning for non-effort Mistral models.
        assert payload["prompt_mode"] == "reasoning"
        # Session affinity for prompt caching.
        assert payload["prompt_cache_key"] == "sess-1"
        assert captured["headers"]["x-affinity"] == "sess-1"
        assert payload["messages"][0] == {"role": "system", "content": "Be brief."}
        assert payload["messages"][1] == {"role": "user", "content": "hi"}
        assert payload["tools"][0]["function"]["strict"] is False
        assert captured["headers"]["authorization"] == "Bearer m-test"

    asyncio.run(main())


def test_replay_normalizes_tool_call_ids():
    async def main():
        model = _model()
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return _sse(_chunk(content="ok", finish="stop"))

        history = [
            UserMessage(content="hi", timestamp=1),
            AssistantMessage(
                api="anthropic-messages",
                provider="anthropic",
                model="claude-sonnet",
                content=[ToolCall(id="toolu_01LongForeignId", name="search", arguments={"q": "x"})],
                timestamp=2,
            ),
            ToolResultMessage(
                tool_call_id="toolu_01LongForeignId",
                tool_name="search",
                content=[TextContent(text="result")],
                timestamp=3,
            ),
        ]
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/chat/completions").mock(side_effect=side_effect)
            stream = mistral_conversations.stream(
                model,
                normalize_context(Context(messages=history)),
                mistral_conversations.MistralOptions(api_key="m-test"),
            )
            [e async for e in stream]
            await stream.result()

        assistant = next(m for m in captured["payload"]["messages"] if m["role"] == "assistant")
        tool = next(m for m in captured["payload"]["messages"] if m["role"] == "tool")
        normalized_id = assistant["tool_calls"][0]["id"]
        assert len(normalized_id) == 9
        assert normalized_id.isalnum()
        # The tool result references the same normalized id.
        assert tool["tool_call_id"] == normalized_id
        assert tool["content"][0] == {"type": "text", "text": "result"}

    asyncio.run(main())


def test_thinking_replay_as_thinking_chunk():
    async def main():
        model = _model()
        model.reasoning = True
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return _sse(_chunk(content="ok", finish="stop"))

        history = [
            UserMessage(content="hi", timestamp=1),
            AssistantMessage(
                api="mistral-conversations",
                provider="mistral",
                model="faux-1",
                content=[
                    ThinkingContent(thinking="reasoned"),
                    TextContent(text="answer"),
                ],
                timestamp=2,
            ),
        ]
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/chat/completions").mock(side_effect=side_effect)
            stream = mistral_conversations.stream(
                model,
                normalize_context(Context(messages=history)),
                mistral_conversations.MistralOptions(api_key="m-test"),
            )
            [e async for e in stream]
            await stream.result()

        assistant = next(m for m in captured["payload"]["messages"] if m["role"] == "assistant")
        assert assistant["content"] == [
            {"type": "thinking", "thinking": [{"type": "text", "text": "reasoned"}]},
            {"type": "text", "text": "answer"},
        ]

    asyncio.run(main())


def test_wire_payload_remaps_response_format():
    payload = {
        "model": "m",
        "stream": True,
        "messages": [],
        "responseFormat": {
            "type": "json_schema",
            "jsonSchema": {"name": "x", "schemaDefinition": {"type": "object"}},
        },
    }
    wire = to_mistral_wire_payload(payload)
    assert "responseFormat" not in wire
    assert wire["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert "schemaDefinition" not in wire["response_format"]["json_schema"]


def test_http_error_format_includes_status_and_body():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/chat/completions").mock(
                return_value=httpx.Response(429, text="rate limited")
            )
            stream = mistral_conversations.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                mistral_conversations.MistralOptions(api_key="m-test"),
            )
            events = [e async for e in stream]
            message = await stream.result()

        assert events[-1].type == "error"
        assert message.stop_reason == "error"
        assert message.error_message == "Mistral API error (429): rate limited"

    asyncio.run(main())


def test_stream_simple_requires_auth():
    model = _model()
    with pytest.raises(ValueError, match="No API key"):
        mistral_conversations.stream_simple(
            model,
            normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
            None,
        )

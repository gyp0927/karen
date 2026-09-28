"""Tests for the Anthropic Messages adapter against a mocked HTTP layer."""

import asyncio
import json

import httpx
import pytest
import respx

from karen_ai import Context, Tool, UserMessage, normalize_context
from karen_ai.api import anthropic_messages
from karen_ai.providers import faux_model

BASE_URL = "https://api.anthropic.test"


def _model(**overrides):
    model = faux_model(provider="anthropic", api="anthropic-messages")
    model.base_url = BASE_URL
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


def _sse(*events: str) -> httpx.Response:
    body = "".join(events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def _event(name: str, payload: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


MESSAGE_START = _event(
    "message_start",
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "faux-1",
            "usage": {"input_tokens": 12, "output_tokens": 1},
        },
    },
)


def test_text_stream():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/messages").mock(
                return_value=_sse(
                    MESSAGE_START,
                    _event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                    _event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
                    _event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "!"}}),
                    _event("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    _event(
                        "message_delta",
                        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}},
                    ),
                    _event("message_stop", {"type": "message_stop"}),
                )
            )
            stream = anthropic_messages.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                anthropic_messages.AnthropicOptions(api_key="sk-ant-test"),
            )
            events = [e async for e in stream]
            message = await stream.result()

        text = "".join(e.delta for e in events if e.type == "text_delta")
        assert text == "Hello!"
        assert message.stop_reason == "stop"
        assert message.usage.input == 12
        assert message.usage.output == 4
        assert message.response_id == "msg_1"

    asyncio.run(main())


def test_thinking_and_tool_use_stream():
    async def main():
        model = _model()
        model.reasoning = True
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/messages").mock(
                return_value=_sse(
                    MESSAGE_START,
                    _event(
                        "content_block_start",
                        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
                    ),
                    _event(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "let me think"}},
                    ),
                    _event(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig123"}},
                    ),
                    _event("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    _event(
                        "content_block_start",
                        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}},
                    ),
                    _event(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"q": "paris"}'}},
                    ),
                    _event("content_block_stop", {"type": "content_block_stop", "index": 1}),
                    _event(
                        "message_delta",
                        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}},
                    ),
                    _event("message_stop", {"type": "message_stop"}),
                )
            )
            stream = anthropic_messages.stream_simple(
                model,
                normalize_context(
                    Context(
                        tools=[Tool(name="search", description="s", parameters={"type": "object"})],
                        messages=[UserMessage(content="weather?", timestamp=1)],
                    )
                ),
                __import__("karen_ai").SimpleStreamOptions(api_key="sk-ant-test", reasoning="low"),
            )
            events = [e async for e in stream]
            message = await stream.result()

        assert message.stop_reason == "toolUse"
        thinking = message.content[0]
        assert thinking.type == "thinking"
        assert thinking.thinking == "let me think"
        assert thinking.thinking_signature == "sig123"
        tool_call = message.content[1]
        assert tool_call.type == "toolCall"
        assert tool_call.name == "search"
        assert tool_call.arguments == {"q": "paris"}

    asyncio.run(main())


def test_request_payload_shape():
    async def main():
        model = _model()
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            captured["headers"] = request.headers
            return _sse(
                MESSAGE_START,
                _event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                _event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
                _event("content_block_stop", {"type": "content_block_stop", "index": 0}),
                _event("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
                _event("message_stop", {"type": "message_stop"}),
            )

        with respx.mock:
            respx.post(f"{BASE_URL}/v1/messages").mock(side_effect=side_effect)
            stream = anthropic_messages.stream(
                model,
                normalize_context(
                    Context(
                        system_prompt="Be nice.",
                        tools=[Tool(name="search", description="s", parameters={"type": "object"})],
                        messages=[UserMessage(content="hi", timestamp=1)],
                    )
                ),
                anthropic_messages.AnthropicOptions(api_key="sk-ant-test", max_tokens=50),
            )
            [e async for e in stream]
            await stream.result()

        payload = captured["payload"]
        assert payload["model"] == "faux-1"
        assert payload["stream"] is True
        assert payload["max_tokens"] == 50
        assert payload["system"] == [
            {"type": "text", "text": "Be nice.", "cache_control": {"type": "ephemeral"}}
        ]
        # Cache control lands on the last conversation message (pi-ai behavior).
        assert payload["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}
        ]
        assert payload["tools"][0]["name"] == "search"
        assert captured["headers"]["x-api-key"] == "sk-ant-test"
        assert captured["headers"]["anthropic-version"] == "2023-06-01"

    asyncio.run(main())


def test_stream_simple_requires_auth():
    model = _model()
    with pytest.raises(ValueError, match="No API key"):
        anthropic_messages.stream_simple(
            model,
            normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
            None,
        )


def test_oauth_token_uses_bearer_and_claude_code_identity():
    async def main():
        model = _model()
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["headers"] = request.headers
            captured["payload"] = json.loads(request.content)
            return _sse(
                MESSAGE_START,
                _event("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
                _event("message_stop", {"type": "message_stop"}),
            )

        with respx.mock:
            respx.post(f"{BASE_URL}/v1/messages").mock(side_effect=side_effect)
            stream = anthropic_messages.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                anthropic_messages.AnthropicOptions(api_key="sk-ant-oat01-token"),
            )
            [e async for e in stream]
            await stream.result()

        assert captured["headers"]["authorization"] == "Bearer sk-ant-oat01-token"
        assert "x-api-key" not in captured["headers"]
        assert captured["headers"]["x-app"] == "cli"
        assert "oauth-2025-04-20" in captured["headers"]["anthropic-beta"]
        # Claude Code identity preamble is forced for OAuth tokens.
        assert captured["payload"]["system"][0]["text"].startswith("You are Claude Code")

    asyncio.run(main())


def test_http_error_terminates_stream_with_error_event():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/messages").mock(
                return_value=httpx.Response(400, json={"error": {"message": "invalid request"}})
            )
            stream = anthropic_messages.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                anthropic_messages.AnthropicOptions(api_key="sk-ant-test", max_retries=0),
            )
            events = [e async for e in stream]
            message = await stream.result()

        assert events[-1].type == "error"
        assert message.stop_reason == "error"
        assert "400" in message.error_message

    asyncio.run(main())

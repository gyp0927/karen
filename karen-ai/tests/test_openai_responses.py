"""Tests for the OpenAI Responses adapter against a mocked HTTP layer."""

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
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    normalize_context,
)
from karen_ai.api import openai_responses
from karen_ai.api.openai_responses_shared import (
    encode_text_signature_v1,
    parse_text_signature,
)
from karen_ai.providers import faux_model
from karen_ai.types import GrammarSampling, OpenAIResponsesCompat

BASE_URL = "https://api.openai.test/v1"


def _model(**overrides):
    model = faux_model(provider="openai", api="openai-responses")
    model.base_url = BASE_URL
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


def _sse(*payloads: dict) -> httpx.Response:
    body = "".join(f"data: {json.dumps(p)}\n\n" for p in payloads)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def _completed(**response_overrides):
    response = {
        "id": "resp_1",
        "status": "completed",
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    response.update(response_overrides)
    return {"type": "response.completed", "response": response}


def test_text_signature_roundtrip():
    sig = encode_text_signature_v1("msg_1")
    assert parse_text_signature(sig) == {"id": "msg_1"}
    sig = encode_text_signature_v1("msg_1", "final_answer")
    assert parse_text_signature(sig) == {"id": "msg_1", "phase": "final_answer"}
    assert parse_text_signature("plain-id") == {"id": "plain-id"}
    assert parse_text_signature(None) is None


def test_text_stream():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/responses").mock(
                return_value=_sse(
                    {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}},
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": [], "status": "in_progress"},
                    },
                    {"type": "response.output_text.delta", "output_index": 0, "delta": "Hello"},
                    {"type": "response.output_text.delta", "output_index": 0, "delta": "!"},
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": "msg_1",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
                            "status": "completed",
                        },
                    },
                    _completed(
                        usage={
                            "input_tokens": 12,
                            "output_tokens": 4,
                            "total_tokens": 16,
                            "input_tokens_details": {"cached_tokens": 2},
                            "output_tokens_details": {"reasoning_tokens": 0},
                        }
                    ),
                )
            )
            stream = openai_responses.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                openai_responses.OpenAIResponsesOptions(api_key="sk-test"),
            )
            events = [e async for e in stream]
            message = await stream.result()

        text = "".join(e.delta for e in events if e.type == "text_delta")
        assert text == "Hello!"
        assert message.stop_reason == "stop"
        assert message.usage.input == 10  # 12 minus 2 cached
        assert message.usage.cache_read == 2
        assert message.usage.output == 4
        assert message.response_id == "resp_1"
        text_block = message.content[0]
        assert text_block.text_signature == encode_text_signature_v1("msg_1")

    asyncio.run(main())


def test_reasoning_and_function_call_stream():
    async def main():
        model = _model()
        model.reasoning = True
        with respx.mock:
            respx.post(f"{BASE_URL}/responses").mock(
                return_value=_sse(
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {"type": "reasoning", "id": "rs_1", "summary": []},
                    },
                    {"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "let me think"},
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "reasoning",
                            "id": "rs_1",
                            "summary": [{"type": "summary_text", "text": "let me think"}],
                            "encrypted_content": "encrypted-blob",
                        },
                    },
                    {
                        "type": "response.output_item.added",
                        "output_index": 1,
                        "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "search", "arguments": ""},
                    },
                    {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": '{"q": "par'},
                    {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": 'is"}'},
                    {
                        "type": "response.output_item.done",
                        "output_index": 1,
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "search",
                            "arguments": '{"q": "paris"}',
                        },
                    },
                    _completed(),
                )
            )
            stream = openai_responses.stream_simple(
                model,
                normalize_context(
                    Context(
                        tools=[Tool(name="search", description="s", parameters={"type": "object"})],
                        messages=[UserMessage(content="weather?", timestamp=1)],
                    )
                ),
                karen_ai.SimpleStreamOptions(api_key="sk-test", reasoning="low"),
            )
            [e async for e in stream]
            message = await stream.result()

        assert message.stop_reason == "toolUse"
        thinking = message.content[0]
        assert thinking.type == "thinking"
        assert thinking.thinking == "let me think"
        # The reasoning item is persisted as a JSON signature for stateless replay.
        stored = json.loads(thinking.thinking_signature)
        assert stored["id"] == "rs_1"
        assert stored["encrypted_content"] == "encrypted-blob"
        tool_call = message.content[1]
        assert tool_call.type == "toolCall"
        assert tool_call.id == "call_1|fc_1"
        assert tool_call.arguments == {"q": "paris"}

    asyncio.run(main())


def test_custom_tool_call_stream_with_grammar_tool():
    async def main():
        model = _model()
        model.compat = OpenAIResponsesCompat(supports_openai_grammar_tools=True)
        grammar_tool = Tool(
            name="run",
            description="run code",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            constrained_sampling=GrammarSampling(variants={"openai_regex": ".+"}),
        )
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return _sse(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "run", "input": ""},
                },
                {"type": "response.custom_tool_call_input.delta", "output_index": 0, "delta": "print("},
                {"type": "response.custom_tool_call_input.delta", "output_index": 0, "delta": "1)"},
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "run", "input": "print(1)"},
                },
                _completed(),
            )

        with respx.mock:
            respx.post(f"{BASE_URL}/responses").mock(side_effect=side_effect)
            stream = openai_responses.stream_simple(
                model,
                normalize_context(
                    Context(tools=[grammar_tool], messages=[UserMessage(content="run it", timestamp=1)])
                ),
                karen_ai.SimpleStreamOptions(api_key="sk-test"),
            )
            events = [e async for e in stream]
            message = await stream.result()

        tool_call = message.content[0]
        assert tool_call.type == "toolCall"
        assert tool_call.name == "run"
        # The raw grammar input is exposed as the single string argument.
        assert tool_call.arguments == {"code": "print(1)"}
        # Streamed deltas are JSON fragments of the synthetic arguments object.
        deltas = "".join(e.delta for e in events if e.type == "toolcall_delta")
        assert json.loads(deltas) == {"code": "print(1)"}
        # The tool is sent as a custom grammar tool.
        assert captured["payload"]["tools"] == [
            {
                "type": "custom",
                "name": "run",
                "description": "run code",
                "format": {"type": "grammar", "syntax": "regex", "definition": ".+"},
            }
        ]

    asyncio.run(main())


def test_request_payload_shape():
    async def main():
        model = _model()
        model.reasoning = True
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            captured["headers"] = request.headers
            return _sse(
                _completed(),
            )

        with respx.mock:
            respx.post(f"{BASE_URL}/responses").mock(side_effect=side_effect)
            stream = openai_responses.stream_simple(
                model,
                normalize_context(
                    Context(
                        system_prompt="Be brief.",
                        tools=[Tool(name="search", description="s", parameters={"type": "object"})],
                        messages=[UserMessage(content="hi", timestamp=1)],
                    )
                ),
                karen_ai.SimpleStreamOptions(api_key="sk-test", reasoning="high", max_tokens=4, session_id="sess-1"),
            )
            [e async for e in stream]
            await stream.result()

        payload = captured["payload"]
        assert payload["model"] == "faux-1"
        assert payload["stream"] is True
        assert payload["store"] is False
        assert payload["prompt_cache_key"] == "sess-1"
        # max_output_tokens is clamped to the API minimum of 16.
        assert payload["max_output_tokens"] == 16
        assert payload["reasoning"] == {"effort": "high", "summary": "auto"}
        assert payload["include"] == ["reasoning.encrypted_content"]
        # Reasoning models get the developer role for the system prompt.
        assert payload["input"][0] == {"role": "developer", "content": "Be brief."}
        assert payload["input"][1] == {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["name"] == "search"
        assert "strict" not in payload["tools"][0]
        assert captured["headers"]["authorization"] == "Bearer sk-test"
        # OpenAI-format session affinity headers.
        assert captured["headers"]["session_id"] == "sess-1"
        assert captured["headers"]["x-client-request-id"] == "sess-1"

    asyncio.run(main())


def test_cross_provider_tool_call_id_normalization():
    async def main():
        model = _model()
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return _sse(_completed())

        history = [
            UserMessage(content="hi", timestamp=1),
            AssistantMessage(
                api="anthropic-messages",
                provider="anthropic",
                model="claude-sonnet",
                content=[
                    TextContent(text="checking"),
                    ToolCall(id="toolu_1|msg part", name="search", arguments={"q": "x"}),
                ],
                timestamp=2,
            ),
            ToolResultMessage(
                tool_call_id="toolu_1|msg part",
                tool_name="search",
                content=[TextContent(text="result")],
                timestamp=3,
            ),
        ]
        with respx.mock:
            respx.post(f"{BASE_URL}/responses").mock(side_effect=side_effect)
            stream = openai_responses.stream(
                model,
                normalize_context(Context(messages=history)),
                openai_responses.OpenAIResponsesOptions(api_key="sk-test"),
            )
            [e async for e in stream]
            await stream.result()

        function_calls = [item for item in captured["payload"]["input"] if item.get("type") == "function_call"]
        assert len(function_calls) == 1
        call = function_calls[0]
        assert call["call_id"] == "toolu_1"
        # Foreign (cross-provider) item ids become deterministic fc_ hashes.
        assert call["id"].startswith("fc_")
        assert call["name"] == "search"
        outputs = [item for item in captured["payload"]["input"] if item.get("type") == "function_call_output"]
        assert outputs == [{"type": "function_call_output", "call_id": "toolu_1", "output": "result"}]

    asyncio.run(main())


def test_same_provider_different_model_drops_item_id():
    async def main():
        model = _model()
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return _sse(_completed())

        history = [
            UserMessage(content="hi", timestamp=1),
            AssistantMessage(
                api="openai-responses",
                provider="openai",
                model="gpt-other",
                content=[ToolCall(id="call_1|fc_abc", name="search", arguments={"q": "x"})],
                timestamp=2,
            ),
            ToolResultMessage(
                tool_call_id="call_1|fc_abc",
                tool_name="search",
                content=[TextContent(text="result")],
                timestamp=3,
            ),
        ]
        with respx.mock:
            respx.post(f"{BASE_URL}/responses").mock(side_effect=side_effect)
            stream = openai_responses.stream(
                model,
                normalize_context(Context(messages=history)),
                openai_responses.OpenAIResponsesOptions(api_key="sk-test"),
            )
            [e async for e in stream]
            await stream.result()

        function_calls = [item for item in captured["payload"]["input"] if item.get("type") == "function_call"]
        # fc_ item id is dropped to avoid OpenAI's fc_xxx/rs_xxx pairing validation.
        assert "id" not in function_calls[0]
        assert function_calls[0]["call_id"] == "call_1"

    asyncio.run(main())


def test_incomplete_max_output_tokens_maps_to_length():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/responses").mock(
                return_value=_sse(
                    {
                        "type": "response.incomplete",
                        "response": {
                            "id": "resp_1",
                            "status": "incomplete",
                            "incomplete_details": {"reason": "max_output_tokens"},
                            "usage": {"input_tokens": 3, "output_tokens": 16, "total_tokens": 19},
                        },
                    },
                )
            )
            stream = openai_responses.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                openai_responses.OpenAIResponsesOptions(api_key="sk-test"),
            )
            [e async for e in stream]
            message = await stream.result()

        assert message.stop_reason == "length"
        assert message.raw_stop_reason == "incomplete.max_output_tokens"

    asyncio.run(main())


def test_response_failed_terminates_stream_with_error_event():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/responses").mock(
                return_value=_sse(
                    {
                        "type": "response.failed",
                        "response": {
                            "id": "resp_1",
                            "status": "failed",
                            "error": {"code": "server_error", "message": "boom"},
                        },
                    },
                )
            )
            stream = openai_responses.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                openai_responses.OpenAIResponsesOptions(api_key="sk-test"),
            )
            events = [e async for e in stream]
            message = await stream.result()

        assert events[-1].type == "error"
        assert message.stop_reason == "error"
        assert "server_error: boom" in message.error_message

    asyncio.run(main())


def test_http_error_terminates_stream_with_error_event():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/responses").mock(
                return_value=httpx.Response(400, json={"error": {"message": "bad request"}})
            )
            stream = openai_responses.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                openai_responses.OpenAIResponsesOptions(api_key="sk-test", max_retries=0),
            )
            events = [e async for e in stream]
            message = await stream.result()

        assert events[-1].type == "error"
        assert message.stop_reason == "error"
        assert "400" in message.error_message

    asyncio.run(main())


def test_stream_simple_requires_auth():
    model = _model()
    with pytest.raises(ValueError, match="No API key"):
        openai_responses.stream_simple(
            model,
            normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
            None,
        )

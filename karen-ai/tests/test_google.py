"""Tests for the Google Generative AI / Vertex adapters against mocked HTTP."""

import asyncio
import base64
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
from karen_ai.api import google_generative_ai, google_vertex
from karen_ai.api.google_shared import (
    GoogleThinkingConfig,
    convert_messages,
    requires_tool_call_id,
    supports_google_strict_tool_sampling,
    uses_google_thinking_level,
)
from karen_ai.providers import faux_model

BASE_URL = "https://generativelanguage.test/v1beta"


def _model(**overrides):
    model = faux_model(provider="google", api="google-generative-ai")
    model.base_url = BASE_URL
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


def _sse(*payloads: dict) -> httpx.Response:
    body = "".join(f"data: {json.dumps(p)}\n\n" for p in payloads)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def _text_chunk(text: str, *, thought=False, signature=None, finish=None, usage=None):
    part = {"text": text}
    if thought:
        part["thought"] = True
    if signature:
        part["thoughtSignature"] = signature
    chunk = {"candidates": [{"content": {"role": "model", "parts": [part]}}]}
    if finish:
        chunk["candidates"][0]["finishReason"] = finish
    if usage:
        chunk["usageMetadata"] = usage
    return chunk


def test_thinking_level_detection():
    assert uses_google_thinking_level(_model(id="gemini-3-flash-preview"))
    assert uses_google_thinking_level(_model(id="gemini-3.1-pro-preview"))
    assert uses_google_thinking_level(_model(id="gemma-4-27b"))
    assert uses_google_thinking_level(_model(id="gemma4-27b"))
    assert not uses_google_thinking_level(_model(id="gemini-2.5-flash"))
    assert requires_tool_call_id("gemini-3-pro")
    assert requires_tool_call_id("claude-sonnet-4-5")
    assert not requires_tool_call_id("gemini-2.5-flash")
    assert supports_google_strict_tool_sampling("gemini-3-pro")
    assert not supports_google_strict_tool_sampling("gemini-2.5-flash")


def test_text_stream():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/models/faux-1:streamGenerateContent").mock(
                return_value=_sse(
                    _text_chunk("Hello", signature=_b64sig()),
                    _text_chunk("!"),
                    _text_chunk(
                        "",
                        finish="STOP",
                        usage={
                            "promptTokenCount": 12,
                            "candidatesTokenCount": 4,
                            "cachedContentTokenCount": 2,
                            "totalTokenCount": 16,
                        },
                    ),
                )
            )
            stream = google_generative_ai.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                google_generative_ai.GoogleOptions(api_key="g-test"),
            )
            events = [e async for e in stream]
            message = await stream.result()

        text = "".join(e.delta for e in events if e.type == "text_delta")
        assert text == "Hello!"
        assert message.stop_reason == "stop"
        assert message.usage.input == 10  # 12 minus 2 cached
        assert message.usage.cache_read == 2
        assert message.usage.output == 4
        # Thought signature retained from the first delta of the block.
        assert message.content[0].text_signature == _b64sig()

    asyncio.run(main())


def test_thinking_and_function_call_stream():
    async def main():
        model = _model(id="gemini-3-flash-preview")
        model.reasoning = True
        with respx.mock:
            respx.post(f"{BASE_URL}/models/gemini-3-flash-preview:streamGenerateContent").mock(
                return_value=_sse(
                    _text_chunk("thinking hard", thought=True),
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"functionCall": {"name": "search", "args": {"q": "paris"}, "id": "call_1"}}],
                                }
                            }
                        ]
                    },
                    _text_chunk(
                        "",
                        finish="STOP",
                        usage={"promptTokenCount": 7, "candidatesTokenCount": 3, "thoughtsTokenCount": 5, "totalTokenCount": 15},
                    ),
                )
            )
            stream = google_generative_ai.stream_simple(
                model,
                normalize_context(
                    Context(
                        tools=[Tool(name="search", description="s", parameters={"type": "object"})],
                        messages=[UserMessage(content="weather?", timestamp=1)],
                    )
                ),
                karen_ai.SimpleStreamOptions(api_key="g-test", reasoning="low"),
            )
            [e async for e in stream]
            message = await stream.result()

        assert message.stop_reason == "toolUse"
        thinking = message.content[0]
        assert thinking.type == "thinking"
        assert thinking.thinking == "thinking hard"
        tool_call = message.content[1]
        assert tool_call.type == "toolCall"
        assert tool_call.id == "call_1"
        assert tool_call.arguments == {"q": "paris"}
        # output = candidates + thoughts tokens
        assert message.usage.output == 8
        assert message.usage.reasoning == 5

    asyncio.run(main())


def test_function_call_without_id_gets_synthetic_unique_id():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/models/faux-1:streamGenerateContent").mock(
                return_value=_sse(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [
                                        {"functionCall": {"name": "a", "args": {}}},
                                        {"functionCall": {"name": "b", "args": {}}},
                                    ],
                                },
                                "finishReason": "STOP",
                            }
                        ]
                    },
                )
            )
            stream = google_generative_ai.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                google_generative_ai.GoogleOptions(api_key="g-test"),
            )
            [e async for e in stream]
            message = await stream.result()

        ids = [b.id for b in message.content if b.type == "toolCall"]
        assert len(ids) == 2
        assert ids[0] != ids[1]
        assert ids[0].startswith("a_")
        assert message.stop_reason == "toolUse"

    asyncio.run(main())


def test_request_payload_shape():
    async def main():
        model = _model(id="gemini-3-flash-preview")
        model.reasoning = True
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            captured["headers"] = request.headers
            captured["url"] = str(request.url)
            return _sse(_text_chunk("ok", finish="STOP"))

        with respx.mock:
            respx.post(f"{BASE_URL}/models/gemini-3-flash-preview:streamGenerateContent").mock(
                side_effect=side_effect
            )
            stream = google_generative_ai.stream_simple(
                model,
                normalize_context(
                    Context(
                        system_prompt="Be brief.",
                        tools=[Tool(name="search", description="s", parameters={"type": "object"})],
                        messages=[UserMessage(content="hi", timestamp=1)],
                    )
                ),
                karen_ai.SimpleStreamOptions(api_key="g-test", reasoning="high", temperature=0.5, max_tokens=100),
            )
            [e async for e in stream]
            await stream.result()

        assert "alt=sse" in captured["url"]
        payload = captured["payload"]
        # System prompt is sent as systemInstruction, not a content turn.
        assert payload["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
        assert payload["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
        assert payload["generationConfig"]["temperature"] == 0.5
        assert payload["generationConfig"]["maxOutputTokens"] == 100
        # Gemini 3 models use the discrete thinkingLevel control.
        assert payload["generationConfig"]["thinkingConfig"] == {
            "includeThoughts": True,
            "thinkingLevel": "HIGH",
        }
        assert payload["tools"][0]["functionDeclarations"][0]["name"] == "search"
        assert "parametersJsonSchema" in payload["tools"][0]["functionDeclarations"][0]
        assert captured["headers"]["x-goog-api-key"] == "g-test"

    asyncio.run(main())


def test_budget_thinking_config_for_25_models():
    async def main():
        model = _model(id="gemini-2.5-flash")
        model.reasoning = True
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return _sse(_text_chunk("ok", finish="STOP"))

        with respx.mock:
            respx.post(f"{BASE_URL}/models/gemini-2.5-flash:streamGenerateContent").mock(side_effect=side_effect)
            stream = google_generative_ai.stream_simple(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                karen_ai.SimpleStreamOptions(api_key="g-test", reasoning="medium"),
            )
            [e async for e in stream]
            await stream.result()

        # Budget-based models get thinkingBudget, not thinkingLevel.
        assert captured["payload"]["generationConfig"]["thinkingConfig"] == {
            "includeThoughts": True,
            "thinkingBudget": 8192,
        }

    asyncio.run(main())


def test_convert_messages_keeps_signed_thinking_for_same_model():
    model = _model(id="gemini-3-flash-preview")
    signature = _b64sig()
    context = normalize_context(
        Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                AssistantMessage(
                    api="google-generative-ai",
                    provider="google",
                    model="gemini-3-flash-preview",
                    content=[
                        ThinkingContent(thinking="reasoning", thinking_signature=signature),
                        TextContent(text="answer", text_signature=signature),
                        ToolCall(id="call_1", name="search", arguments={"q": "x"}, thought_signature=signature),
                    ],
                    timestamp=2,
                ),
                ToolResultMessage(
                    tool_call_id="call_1",
                    tool_name="search",
                    content=[TextContent(text="result")],
                    timestamp=3,
                ),
            ]
        )
    )
    contents = convert_messages(model, context)
    model_turn = contents[1]
    assert model_turn["role"] == "model"
    assert model_turn["parts"][0] == {"thought": True, "text": "reasoning", "thoughtSignature": signature}
    assert model_turn["parts"][1] == {"text": "answer", "thoughtSignature": signature}
    assert model_turn["parts"][2]["thoughtSignature"] == signature
    # Gemini 3 replays the tool call id.
    assert model_turn["parts"][2]["functionCall"]["id"] == "call_1"
    # Function response is a user turn with the tool call id echoed.
    assert contents[2]["role"] == "user"
    assert contents[2]["parts"][0]["functionResponse"]["id"] == "call_1"
    assert contents[2]["parts"][0]["functionResponse"]["response"] == {"output": "result"}


def test_convert_messages_drops_signature_for_other_model():
    model = _model(id="gemini-3-flash-preview")
    other_signature = _b64sig()
    context = normalize_context(
        Context(
            messages=[
                UserMessage(content="hi", timestamp=1),
                AssistantMessage(
                    api="google-generative-ai",
                    provider="google",
                    model="gemini-2.5-flash",  # different model: signature unusable
                    content=[TextContent(text="answer", text_signature=other_signature)],
                    timestamp=2,
                ),
            ]
        )
    )
    contents = convert_messages(model, context)
    assert contents[1]["parts"] == [{"text": "answer"}]


def test_error_finish_reason_terminates_stream():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/models/faux-1:streamGenerateContent").mock(
                return_value=_sse(_text_chunk("partial", finish="SAFETY"))
            )
            stream = google_generative_ai.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                google_generative_ai.GoogleOptions(api_key="g-test"),
            )
            events = [e async for e in stream]
            message = await stream.result()

        assert events[-1].type == "error"
        assert message.stop_reason == "error"
        assert "SAFETY" in message.error_message

    asyncio.run(main())


def test_stream_simple_requires_auth():
    model = _model()
    with pytest.raises(ValueError, match="No API key"):
        google_generative_ai.stream_simple(
            model,
            normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
            None,
        )


# ---------------------------------------------------------------------------
# Vertex
# ---------------------------------------------------------------------------


def _vertex_model(**overrides):
    model = faux_model(provider="google-vertex", api="google-vertex")
    model.base_url = "https://{location}-aiplatform.test"
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


def test_vertex_api_key_mode_url_and_headers():
    async def main():
        model = _vertex_model()
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = request.headers
            return _sse(_text_chunk("ok", finish="STOP"))

        with respx.mock:
            respx.post(url__startswith="https://aiplatform.googleapis.com/").mock(side_effect=side_effect)
            stream = google_vertex.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                google_vertex.GoogleVertexOptions(api_key="vertex-key"),
            )
            [e async for e in stream]
            await stream.result()

        assert captured["url"].startswith("https://aiplatform.googleapis.com/v1/publishers/google/models/faux-1")
        assert captured["headers"]["x-goog-api-key"] == "vertex-key"

    asyncio.run(main())


def test_vertex_adc_url_construction():
    model = _vertex_model()
    url = google_vertex._build_adc_url(model, "my-project", "europe-west1")
    assert url == (
        "https://europe-west1-aiplatform.googleapis.com/v1/projects/my-project"
        "/locations/europe-west1/publishers/google/models/faux-1:streamGenerateContent?alt=sse"
    )
    # Custom base URL with a version segment is used as-is.
    model.base_url = "https://proxy.example.com/vertex/v1beta1"
    url = google_vertex._build_adc_url(model, "p", "loc")
    assert url.startswith("https://proxy.example.com/vertex/v1beta1/projects/p/")
    # Custom base URL without a version segment gets /v1 appended.
    model.base_url = "https://proxy.example.com"
    url = google_vertex._build_adc_url(model, "p", "loc")
    assert url.startswith("https://proxy.example.com/v1/projects/p/")


def test_vertex_credential_marker_and_placeholder_fall_back_to_adc():
    assert google_vertex._resolve_api_key(google_vertex.GoogleVertexOptions(api_key="gcp-vertex-credentials")) is None
    assert google_vertex._resolve_api_key(google_vertex.GoogleVertexOptions(api_key="<paste-key>")) is None
    assert google_vertex._resolve_api_key(google_vertex.GoogleVertexOptions(api_key="real-key")) == "real-key"


def _b64sig() -> str:
    # Valid base64 with length % 4 == 0, as required by Google's TYPE_BYTES fields.
    return base64.b64encode(b"test-signature-blob").decode()

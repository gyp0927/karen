"""OpenAI Codex Responses API tests against a mocked ChatGPT backend."""

import asyncio
import base64
import json

import httpx
import respx

from karen_ai import Context, UserMessage, normalize_context
from karen_ai.api import openai_codex_responses as codex
from karen_ai.api.openai_codex_responses import OpenAICodexResponsesOptions, extract_account_id, resolve_codex_url
from karen_ai.providers import faux_model

BASE_URL = "https://chatgpt.test/backend-api"
RESPONSES_URL = f"{BASE_URL}/codex/responses"


def _jwt(account_id: str = "acct-123") -> str:
    def part(payload: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")

    return f"{part({'alg': 'none'})}.{part({'https://api.openai.com/auth': {'chatgpt_account_id': account_id}})}.sig"


def _model(**overrides):
    model = faux_model(provider="openai-codex", api="openai-codex-responses")
    model.base_url = BASE_URL
    model.reasoning = True
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


def _sse(*events: dict) -> httpx.Response:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def _completed(**response_overrides):
    response = {
        "status": "completed",
        "usage": {
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16,
            "input_tokens_details": {"cached_tokens": 2},
        },
        "output": [],
    }
    response.update(response_overrides)
    return {"type": "response.completed", "response": response}


def _context():
    return normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)]))


def test_extract_account_id():
    assert extract_account_id(_jwt("acct-xyz")) == "acct-xyz"


def test_resolve_codex_url():
    assert resolve_codex_url(None) == "https://chatgpt.com/backend-api/codex/responses"
    assert resolve_codex_url("https://x.test/codex") == "https://x.test/codex/responses"
    assert resolve_codex_url("https://x.test/codex/responses") == "https://x.test/codex/responses"
    assert resolve_codex_url("https://x.test/") == "https://x.test/codex/responses"


def _request_body(request) -> dict:
    """Decodes a captured request body, transparently handling zstd compression."""
    content = request.content
    if request.headers.get("content-encoding") == "zstd":
        import zstandard

        content = zstandard.ZstdDecompressor().decompress(content)
    return json.loads(content)


def test_text_stream_with_usage_and_headers():
    async def main():
        model = _model()
        with respx.mock:
            route = respx.post(RESPONSES_URL).mock(
                return_value=_sse(
                    {"type": "response.created", "response": {"id": "resp-1"}},
                    {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg-1"}},
                    {"type": "response.output_text.delta", "output_index": 0, "delta": "Hello"},
                    {"type": "response.output_text.delta", "output_index": 0, "delta": "!"},
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {"type": "message", "id": "msg-1", "content": [{"type": "output_text", "text": "Hello!"}]},
                    },
                    _completed(),
                )
            )
            stream = codex.stream(model, _context(), OpenAICodexResponsesOptions(transport="sse", api_key=_jwt(), session_id="sess-9"))
            events = [e async for e in stream]
            message = await stream.result()

        assert message.stop_reason == "stop"
        assert "".join(e.delta for e in events if e.type == "text_delta") == "Hello!"
        # Cached tokens are split out of input.
        assert message.usage.input == 10
        assert message.usage.cache_read == 2
        assert message.usage.output == 4

        request = route.calls[0].request
        assert request.headers["authorization"].startswith("Bearer ")
        assert request.headers["chatgpt-account-id"] == "acct-123"
        assert request.headers["originator"] == "pi"
        assert request.headers["session-id"] == "sess-9"
        assert request.headers["openai-beta"] == "responses=experimental"
        assert request.headers["accept"] == "text/event-stream"

        assert request.headers["content-encoding"] == "zstd"
        body = _request_body(request)
        assert body["model"] == model.id
        assert body["store"] is False
        assert body["stream"] is True
        assert body["instructions"] == "You are a helpful assistant."
        assert body["include"] == ["reasoning.encrypted_content"]
        assert body["prompt_cache_key"] == "sess-9"
        assert body["tool_choice"] == "auto"
        # reasoning=True model defaults to effort "none" without an explicit effort.
        assert body["reasoning"] == {"effort": "none"}

    asyncio.run(main())


def test_usage_limit_friendly_message():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(RESPONSES_URL).mock(
                return_value=httpx.Response(
                    429,
                    json={
                        "error": {
                            "code": "usage_limit_reached",
                            "message": "limit hit",
                            "plan_type": "PLUS",
                            "resets_at": 1893456000,
                        }
                    },
                )
            )
            stream = codex.stream(model, _context(), OpenAICodexResponsesOptions(transport="sse", api_key=_jwt()))
            message = await stream.result()

        assert message.stop_reason == "error"
        assert "usage limit" in (message.error_message or "").lower()
        assert "plus plan" in (message.error_message or "").lower()

    asyncio.run(main())


def test_retry_on_transient_500():
    async def main():
        model = _model()
        with respx.mock:
            route = respx.post(RESPONSES_URL).mock(
                side_effect=[
                    httpx.Response(500, text="upstream connect error"),
                    httpx.Response(200, headers={"content-type": "text/event-stream"}, text=""),
                ]
            )
            stream = codex.stream(
                model, _context(), OpenAICodexResponsesOptions(transport="sse", api_key=_jwt(), max_retries=2)
            )
            message = await stream.result()

        assert route.call_count == 2
        # Second response carried no terminal event.
        assert message.stop_reason == "error"
        assert "terminal response event" in (message.error_message or "")

    asyncio.run(main())


def test_response_failed_event():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(RESPONSES_URL).mock(
                return_value=_sse(
                    {"type": "response.failed", "response": {"error": {"code": "boom", "message": "it broke"}}}
                )
            )
            stream = codex.stream(model, _context(), OpenAICodexResponsesOptions(transport="sse", api_key=_jwt()))
            message = await stream.result()

        assert message.stop_reason == "error"
        assert "it broke" in (message.error_message or "")

    asyncio.run(main())


def test_invalid_jay_dubya_tee():
    async def main():
        stream = codex.stream(_model(), _context(), OpenAICodexResponsesOptions(transport="sse", api_key="not-a-jwt"))
        message = await stream.result()
        assert message.stop_reason == "error"
        assert "accountId" in (message.error_message or "")

    asyncio.run(main())

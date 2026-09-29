"""pi-messages API tests against a mocked SSE backend."""

import asyncio
import json

import httpx
import respx

from karen_ai import Context, UserMessage, normalize_context
from karen_ai.api import pi_messages
from karen_ai.api.pi_messages import PiMessagesOptions
from karen_ai.providers import faux_model

BASE_URL = "https://radius.test/v1"


def _model():
    model = faux_model(provider="radius", api="pi-messages")
    model.base_url = BASE_URL
    return model


def _sse(*events: dict) -> httpx.Response:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def _usage():
    return {
        "input": 3,
        "output": 5,
        "cacheRead": 1,
        "cacheWrite": 0,
        "totalTokens": 9,
        "cost": {"input": 0.1, "output": 0.2, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.3},
    }


def _context():
    return normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)]))


def test_full_event_flow():
    async def main():
        model = _model()
        with respx.mock:
            route = respx.post(f"{BASE_URL}/messages").mock(
                return_value=_sse(
                    {"type": "start"},
                    {"type": "text_start", "contentIndex": 0},
                    {"type": "text_delta", "contentIndex": 0, "delta": "Hel"},
                    {"type": "text_delta", "contentIndex": 0, "delta": "lo"},
                    {"type": "text_end", "contentIndex": 0, "content": "Hello", "contentSignature": "sig-1"},
                    {"type": "done", "reason": "stop", "usage": _usage(), "responseId": "resp-1"},
                )
            )
            stream = pi_messages.stream(model, _context(), PiMessagesOptions(api_key="rk-test"))
            events = [e async for e in stream]
            message = await stream.result()

        assert [e.type for e in events] == ["start", "text_start", "text_delta", "text_delta", "text_end", "done"]
        assert message.stop_reason == "stop"
        assert message.content[0].text == "Hello"
        assert message.content[0].text_signature == "sig-1"
        assert message.usage.output == 5
        assert message.usage.cache_read == 1
        assert message.usage.cost.total == 0.3
        assert message.response_id == "resp-1"

        request = route.calls[0].request
        assert request.headers["authorization"] == "Bearer rk-test"
        payload = json.loads(request.content)
        assert payload["model"] == model.id
        assert payload["context"]["messages"][0]["role"] == "user"
        assert payload["options"]["toolChoice"] is None or "toolChoice" in payload["options"]

    asyncio.run(main())


def test_toolcall_flow():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/messages").mock(
                return_value=_sse(
                    {"type": "start"},
                    {"type": "toolcall_start", "contentIndex": 0, "id": "call-1", "toolName": "search"},
                    {"type": "toolcall_delta", "contentIndex": 0, "delta": '{"q": "we'},
                    {"type": "toolcall_delta", "contentIndex": 0, "delta": 'ather"}'},
                    {
                        "type": "toolcall_end",
                        "contentIndex": 0,
                        "toolCall": {"type": "toolCall", "id": "call-1", "name": "search", "arguments": {"q": "weather"}},
                    },
                    {"type": "done", "reason": "toolUse", "usage": _usage()},
                )
            )
            stream = pi_messages.stream(model, _context(), PiMessagesOptions(api_key="rk-test"))
            events = [e async for e in stream]
            message = await stream.result()

        assert message.stop_reason == "toolUse"
        tool_call = next(e for e in events if e.type == "toolcall_end").tool_call
        assert tool_call.id == "call-1"
        assert tool_call.name == "search"
        assert tool_call.arguments == {"q": "weather"}

    asyncio.run(main())


def test_error_response_diagnostic():
    async def main():
        model = _model()
        body = json.dumps({"error": {"message": "bad gateway model", "code": "model_error"}})
        with respx.mock:
            respx.post(f"{BASE_URL}/messages").mock(
                return_value=httpx.Response(502, json={"error": {"message": "bad gateway model", "code": "model_error"}})
            )
            stream = pi_messages.stream(model, _context(), PiMessagesOptions(api_key="rk-test"))
            message = await stream.result()

        assert message.stop_reason == "error"
        assert "bad gateway model" in (message.error_message or "")
        assert message.diagnostics and message.diagnostics[0].type == "pi_messages_response_failure"
        assert message.diagnostics[0].details["status"] == 502
        _ = body

    asyncio.run(main())


def test_terminal_error_event():
    async def main():
        model = _model()
        with respx.mock:
            respx.post(f"{BASE_URL}/messages").mock(
                return_value=_sse(
                    {"type": "start"},
                    {"type": "error", "reason": "error", "usage": _usage(), "errorMessage": "backend exploded"},
                )
            )
            stream = pi_messages.stream(model, _context(), PiMessagesOptions(api_key="rk-test"))
            message = await stream.result()

        assert message.stop_reason == "error"
        assert message.error_message == "backend exploded"

    asyncio.run(main())


def test_missing_api_key():
    async def main():
        stream = pi_messages.stream(_model(), _context(), PiMessagesOptions())
        message = await stream.result()
        assert message.stop_reason == "error"
        assert "No API key" in (message.error_message or "")

    asyncio.run(main())


def test_cache_retention_env_opt_in():
    async def main():
        model = _model()
        with respx.mock:
            route = respx.post(f"{BASE_URL}/messages").mock(
                return_value=_sse({"type": "done", "reason": "stop", "usage": _usage()})
            )
            stream = pi_messages.stream(
                model, _context(), PiMessagesOptions(api_key="rk-test", env={"PI_CACHE_RETENTION": "long"})
            )
            await stream.result()

        payload = json.loads(route.calls[0].request.content)
        assert payload["options"]["cacheRetention"] == "long"

    asyncio.run(main())

"""Codex WebSocket transport tests: parsing, session caching, and SSE fallback."""

import asyncio
import base64
import json

import httpx
import pytest
import respx
from websockets.asyncio.server import serve
from websockets.protocol import State

from karen_ai import Context, UserMessage, normalize_context
from karen_ai.abort import AbortController
from karen_ai.api import codex_websocket as ws
from karen_ai.api import openai_codex_responses as codex
from karen_ai.api.codex_errors import CodexProtocolError
from karen_ai.api.openai_codex_responses import OpenAICodexResponsesOptions
from karen_ai.providers import faux_model
from karen_ai.session_resources import cleanup_session_resources


def _jwt(account_id: str = "acct-123") -> str:
    def part(payload: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")

    return f"{part({'alg': 'none'})}.{part({'https://api.openai.com/auth': {'chatgpt_account_id': account_id}})}.sig"


class FakeSocket:
    """Duck-typed `websockets` connection for parse-level tests."""

    def __init__(self, messages, error=None, state=State.OPEN):
        self.messages = list(messages)
        self.error = error
        self.state = state
        self.closed = []

    async def __aiter__(self):
        for message in self.messages:
            yield message
        if self.error is not None:
            raise self.error

    def close(self, code=1000, reason=""):
        self.closed.append((code, reason))


class FakeCloseError(Exception):
    def __init__(self, code, reason):
        super().__init__(f"closed {code} {reason}")
        self.code = code
        self.reason = reason


def _events(*types: str) -> list[str]:
    return [json.dumps({"type": event}) for event in types]


def _context(text: str = "hi"):
    return normalize_context(Context(messages=[UserMessage(content=text, timestamp=1)]))


def test_parse_websocket_yields_events_until_completion():
    async def main():
        socket = FakeSocket(
            _events("response.created", "response.output_text.delta", "response.completed", "response.output_text.delta")
        )
        events = [event async for event in ws.parse_websocket(socket)]

        assert [event["type"] for event in events] == [
            "response.created",
            "response.output_text.delta",
            "response.completed",
        ]

    asyncio.run(main())


def test_parse_websocket_reports_early_close():
    async def main():
        socket = FakeSocket(_events("response.created"), error=FakeCloseError(1006, "abnormal"))
        with pytest.raises(ws.WebSocketCloseError) as caught:
            [event async for event in ws.parse_websocket(socket)]

        assert caught.value.code == 1006
        assert caught.value.reason == "abnormal"

    asyncio.run(main())


def test_parse_websocket_requires_a_terminal_event():
    async def main():
        socket = FakeSocket(_events("response.created"))
        with pytest.raises(RuntimeError, match="closed before response.completed"):
            [event async for event in ws.parse_websocket(socket)]

    asyncio.run(main())


def test_parse_websocket_rejects_invalid_json():
    async def main():
        socket = FakeSocket(["{not json"])
        with pytest.raises(CodexProtocolError, match="Invalid Codex WebSocket JSON"):
            [event async for event in ws.parse_websocket(socket)]

    asyncio.run(main())


def test_parse_websocket_idle_timeout():
    async def main():
        class SilentSocket(FakeSocket):
            async def __aiter__(self):
                await asyncio.sleep(30)
                yield "{}"

        socket = SilentSocket([])
        with pytest.raises(TimeoutError, match="WebSocket idle timeout after 50ms"):
            [event async for event in ws.parse_websocket(socket, idle_timeout_ms=50)]

        assert socket.closed == [(1000, "idle_timeout")]

    asyncio.run(main())


def test_parse_websocket_honors_abort():
    async def main():
        class SilentSocket(FakeSocket):
            async def __aiter__(self):
                await asyncio.sleep(30)
                yield "{}"

        controller = AbortController()
        socket = SilentSocket([])

        async def abort_soon():
            await asyncio.sleep(0.05)
            controller.abort()

        asyncio.ensure_future(abort_soon())
        with pytest.raises(Exception, match="Request was aborted"):
            [event async for event in ws.parse_websocket(socket, controller.signal)]

    asyncio.run(main())


# ---------------------------------------------------------------------------
# Cached-context request bodies
# ---------------------------------------------------------------------------


def _body(input_items, **overrides):
    return {"model": "gpt-5", "store": False, "input": input_items, **overrides}


def test_cached_body_requires_matching_request():
    entry = ws.CachedWebSocketConnection(socket=None, busy=False, created_at=0)
    entry.continuation = ws.CachedWebSocketContinuation(
        last_request_body=_body([{"role": "user", "content": "hi"}]),
        last_response_id="resp-1",
        last_response_items=[{"type": "message", "role": "assistant"}],
    )

    # Same body plus one appended item: the delta and response id are sent.
    extended = _body(
        [
            {"role": "user", "content": "hi"},
            {"type": "message", "role": "assistant"},
            {"role": "user", "content": "again"},
        ]
    )
    rewritten = ws.build_cached_websocket_request_body(entry, extended)
    assert rewritten["previous_response_id"] == "resp-1"
    assert rewritten["input"] == [{"role": "user", "content": "again"}]

    # A different model invalidates the cache instead of sending a bogus delta.
    mismatched = _body(list(extended["input"]), model="gpt-5-mini")
    assert ws.build_cached_websocket_request_body(entry, mismatched) == mismatched
    assert entry.continuation is None


def test_cached_body_ignores_rewritten_prefixes_and_shorter_input():
    continuation = ws.CachedWebSocketContinuation(
        last_request_body=_body([{"role": "user", "content": "hi"}]),
        last_response_id="resp-1",
        last_response_items=[{"type": "message", "role": "assistant"}],
    )
    baseline = [{"role": "user", "content": "hi"}, {"type": "message", "role": "assistant"}]

    assert ws.get_cached_websocket_input_delta(_body(baseline), continuation) == []
    assert ws.get_cached_websocket_input_delta(_body(baseline[:1]), continuation) is None
    changed = _body([{"role": "user", "content": "different"}, *baseline[1:]])
    assert ws.get_cached_websocket_input_delta(changed, continuation) is None


# ---------------------------------------------------------------------------
# Live socket: caching, reuse, and cleanup
# ---------------------------------------------------------------------------


class CodexTestServer:
    """Minimal Codex WebSocket backend; records every `response.create` frame."""

    def __init__(self, responder=None):
        self.received: list[dict] = []
        self.connections = 0
        self._server = None
        self._responder = responder or self._default_responder
        self._tasks: list[asyncio.Task] = []

    @property
    def port(self) -> int:
        return self._server.sockets[0].getsockname()[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def _handler(self, socket):
        self.connections += 1
        try:
            async for raw in socket:
                frame = json.loads(raw)
                self.received.append(frame)
                response_id = f"resp-{len(self.received)}"
                for event in self._responder(frame, response_id):
                    await socket.send(json.dumps(event))
                # The socket stays open so follow-up turns can reuse it.
        except Exception:  # pragma: no cover - client-side disconnects
            pass

    @staticmethod
    def _default_responder(frame, response_id):
        return [
            {"type": "response.created", "response": {"id": response_id}},
            {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg-1"}},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "Hello!"},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message", "id": "msg-1", "content": [{"type": "output_text", "text": "Hello!"}]},
            },
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "status": "completed",
                    "usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
                    "output": [],
                },
            },
        ]

    async def __aenter__(self):
        self._server = await serve(self._handler, "127.0.0.1", 0).__aenter__()
        return self

    async def __aexit__(self, *exc):
        for task in self._tasks:
            task.cancel()
        await self._server.__aexit__(*exc)


def _ws_model(base_url: str):
    model = faux_model(provider="openai-codex", api="openai-codex-responses")
    model.base_url = base_url
    model.reasoning = True
    return model


def _drain(stream):
    async def collect():
        return [event async for event in stream]

    return collect


def test_acquire_websocket_reuses_cached_socket():
    async def main():
        async with CodexTestServer() as server:
            url = codex.resolve_codex_websocket_url(server.base_url)
            headers = {"Authorization": "Bearer x"}
            first, entry, reused, release = await ws.acquire_websocket(url, headers, "sess-1", "acct-1")
            assert reused is False
            await release(keep=True)

            second, entry2, reused2, release2 = await ws.acquire_websocket(url, headers, "sess-1", "acct-1")
            assert reused2 is True
            assert second is first
            assert entry2 is entry
            await release2(keep=True)

            await ws.close_openai_codex_websocket_sessions("sess-1")
            assert ws.websocket_session_ids() == []

    asyncio.run(main())


def test_acquire_websocket_without_session_is_not_cached():
    async def main():
        async with CodexTestServer() as server:
            url = codex.resolve_codex_websocket_url(server.base_url)
            socket, entry, reused, release = await ws.acquire_websocket(url, {}, None, "acct-1")
            assert entry is None and reused is False
            await release(keep=True)
            assert ws.websocket_session_ids() == []

    asyncio.run(main())


def test_stream_over_websocket_reuses_context_between_turns():
    async def main():
        ws.reset_openai_codex_websocket_debug_stats()
        async with CodexTestServer() as server:
            model = _ws_model(server.base_url)
            options = OpenAICodexResponsesOptions(
                transport="websocket-cached", api_key=_jwt(), session_id="sess-ws"
            )

            stream = codex.stream(model, _context("hi"), options)
            events = [event async for event in stream]
            message = await stream.result()

            assert "".join(event.delta for event in events if event.type == "text_delta") == "Hello!"
            assert message.stop_reason == "stop"
            assert server.received[0]["type"] == "response.create"
            assert server.received[0]["store"] is False
            assert "previous_response_id" not in server.received[0]
            assert "session-id" not in json.dumps(server.received[0])

            # Second turn: the transcript grows by the assistant reply and a new user turn.
            follow_up = normalize_context(
                Context(messages=[UserMessage(content="hi", timestamp=1), message, UserMessage(content="again", timestamp=2)])
            )
            second = codex.stream(model, follow_up, options)
            await second.result()

            assert server.connections == 1, "the cached socket should be reused"
            delta_frame = server.received[1]
            assert delta_frame["previous_response_id"] == "resp-1"
            # Only the appended turn is sent; the earlier context stays cached server-side.
            assert delta_frame["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "again"}]}]
            assert server.received[0]["input"] == [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            ]

            stats = ws.get_openai_codex_websocket_debug_stats("sess-ws")
            assert stats.requests == 2
            assert stats.connections_created == 1
            assert stats.connections_reused == 1
            assert stats.delta_requests == 1

        await cleanup_session_resources("sess-ws")

    asyncio.run(main())


def test_stream_falls_back_to_sse_and_pins_the_session():
    async def main():
        ws.reset_openai_codex_websocket_debug_stats()
        connections = 0

        async def counting_handler(socket):
            nonlocal connections
            connections += 1
            await socket.close()

        async with serve(counting_handler, "127.0.0.1", 0) as server:
            base_url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            model = _ws_model(base_url)
            options = OpenAICodexResponsesOptions(transport="auto", api_key=_jwt(), session_id="sess-fallback")
            response_body = "".join(
                f"data: {json.dumps(event)}\n\n"
                for event in [
                    {"type": "response.created", "response": {"id": "resp-sse"}},
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {"type": "message", "id": "msg-1"},
                    },
                    {"type": "response.output_text.delta", "output_index": 0, "delta": "Hi"},
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": "msg-1",
                            "content": [{"type": "output_text", "text": "Hi"}],
                        },
                    },
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp-sse",
                            "status": "completed",
                            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                            "output": [],
                        },
                    },
                ]
            )

            with respx.mock:
                route = respx.post(f"{base_url}/codex/responses").mock(
                    return_value=httpx.Response(
                        200, headers={"content-type": "text/event-stream"}, text=response_body
                    )
                )
                first = codex.stream(model, _context("hi"), options)
                message = await first.result()
                assert message.stop_reason == "stop"
                assert connections == 1, "the WebSocket attempt should have been made"

                diagnostics = [d.type for d in (message.diagnostics or [])]
                assert "provider_transport_failure" in diagnostics
                failure = next(d for d in message.diagnostics if d.type == "provider_transport_failure")
                assert failure.details["configuredTransport"] == "auto"
                assert failure.details["fallbackTransport"] == "sse"
                assert failure.details["phase"] == "before_message_stream_start"

                # The session is pinned to SSE: no further WebSocket attempts.
                second = codex.stream(model, _context("hi"), options)
                await second.result()
                assert connections == 1
                assert route.call_count == 2

            stats = ws.get_openai_codex_websocket_debug_stats("sess-fallback")
            assert stats.websocket_failures == 1
            # pi-ai re-records the fallback on every request of a pinned session.
            assert stats.sse_fallbacks == 2
            assert stats.websocket_fallback_active is True

        ws.reset_openai_codex_websocket_debug_stats()

    asyncio.run(main())


def test_connection_limit_error_retries_once_over_websocket():
    async def main():
        ws.reset_openai_codex_websocket_debug_stats()
        attempts = 0

        async def handler(socket):
            nonlocal attempts
            attempts += 1
            async for raw in socket:
                if attempts == 1:
                    await socket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "error": {
                                    "code": "websocket_connection_limit_reached",
                                    "message": "too many connections",
                                },
                            }
                        )
                    )
                    await socket.close()
                    return
                for event in CodexTestServer._default_responder(json.loads(raw), "resp-ok"):
                    await socket.send(json.dumps(event))
                await socket.close()

        async with serve(handler, "127.0.0.1", 0) as server:
            model = _ws_model(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}")
            options = OpenAICodexResponsesOptions(transport="websocket", api_key=_jwt(), session_id="sess-limit")
            stream = codex.stream(model, _context("hi"), options)
            message = await stream.result()

        assert attempts == 2
        assert message.stop_reason == "stop"

    asyncio.run(main())

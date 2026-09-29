"""Tests for the Bedrock ConverseStream adapter, SigV4 signer, and event-stream codec."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import zlib
from datetime import datetime, timezone

import httpx
import pytest
import respx

from karen_ai import (
    AssistantMessage,
    Context,
    SystemMessage,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    normalize_context,
)
from karen_ai.api import bedrock_converse_stream as bedrock
from karen_ai.api.bedrock_converse_stream import BedrockOptions
from karen_ai.providers.amazon_bedrock import AMAZON_BEDROCK_MODELS
from karen_ai.utils.aws_eventstream import EventStreamDecoder, EventStreamError, decode_message
from karen_ai.utils.aws_sigv4 import AwsCredentials, resolve_aws_credentials, sign_request

MODEL = next(m for m in AMAZON_BEDROCK_MODELS if m.id == "anthropic.claude-sonnet-4-5")
ENDPOINT = "https://bedrock-runtime.us-east-1.amazonaws.com"
CONVERSE_URL = f"{ENDPOINT}/model/anthropic.claude-sonnet-4-5/converse-stream"

ENV_CREDS = {"AWS_ACCESS_KEY_ID": "AKIDTEST", "AWS_SECRET_ACCESS_KEY": "secret-test"}


@pytest.fixture(autouse=True)
def _isolate_aws_env(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))


def _options(**overrides) -> BedrockOptions:
    env = dict(ENV_CREDS)
    env.update(overrides.pop("env", {}))
    return BedrockOptions(env=env, **overrides)


# ---------------------------------------------------------------------------
# Event-stream codec helpers
# ---------------------------------------------------------------------------


def _encode_message(headers: dict, payload: bytes) -> bytes:
    header_bytes = b""
    for name, value in headers.items():
        encoded_name = name.encode()
        header_bytes += bytes([len(encoded_name)]) + encoded_name
        if isinstance(value, str):
            encoded = value.encode()
            header_bytes += bytes([7]) + struct.pack(">H", len(encoded)) + encoded
        elif isinstance(value, bool):
            header_bytes += bytes([0 if value else 1])
        else:
            raise AssertionError("unsupported header value in test encoder")
    total = 8 + 4 + len(header_bytes) + len(payload) + 4
    prelude = struct.pack(">II", total, len(header_bytes))
    prelude_crc = struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
    body = prelude + prelude_crc + header_bytes + payload
    return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _event(event_type: str, payload: dict) -> bytes:
    return _encode_message(
        {":message-type": "event", ":event-type": event_type, ":content-type": "application/json"},
        json.dumps(payload).encode(),
    )


def _exception(exception_type: str, message: str) -> bytes:
    return _encode_message(
        {":message-type": "exception", ":exception-type": exception_type, ":content-type": "application/json"},
        json.dumps({"message": message}).encode(),
    )


def _converse_response(*frames: bytes, request_id: str = "req-1") -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "application/vnd.amazon.eventstream", "x-amzn-requestid": request_id},
        content=b"".join(frames),
    )


# ---------------------------------------------------------------------------
# SigV4
# ---------------------------------------------------------------------------


def test_sigv4_classic_aws_documentation_vector():
    # The IAM ListUsers example from the AWS SigV4 documentation.
    headers = sign_request(
        method="GET",
        url="https://iam.amazonaws.com/?Action=ListUsers&Version=2010-05-08",
        headers={"content-type": "application/x-www-form-urlencoded; charset=utf-8"},
        body=b"",
        credentials=AwsCredentials("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"),
        region="us-east-1",
        service="iam",
        request_datetime=datetime(2015, 8, 30, 12, 36, 0, tzinfo=timezone.utc),
        include_content_sha256_header=False,
    )
    authorization = headers["Authorization"]
    assert "Credential=AKIDEXAMPLE/20150830/us-east-1/iam/aws4_request" in authorization
    assert "SignedHeaders=content-type;host;x-amz-date" in authorization
    assert "Signature=5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7" in authorization


def test_sigv4_content_sha256_header_signed_when_enabled():
    headers = sign_request(
        method="POST",
        url="https://bedrock-runtime.us-east-1.amazonaws.com/model/x/converse-stream",
        headers={"content-type": "application/json"},
        body=b'{"a":1}',
        credentials=AwsCredentials("AKID", "secret", "session-token"),
        region="us-east-1",
        service="bedrock",
        request_datetime=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    authorization = headers["Authorization"]
    assert "x-amz-content-sha256" in authorization
    assert headers["x-amz-content-sha256"] == "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"
    assert headers["x-amz-security-token"] == "session-token"
    # Deterministic for identical inputs.
    again = sign_request(
        method="POST",
        url="https://bedrock-runtime.us-east-1.amazonaws.com/model/x/converse-stream",
        headers={"content-type": "application/json"},
        body=b'{"a":1}',
        credentials=AwsCredentials("AKID", "secret", "session-token"),
        region="us-east-1",
        service="bedrock",
        request_datetime=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert again["Authorization"] == authorization


def test_resolve_aws_credentials_env_wins(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-ak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-sk")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "env-st")
    credentials = resolve_aws_credentials()
    assert credentials == AwsCredentials("env-ak", "env-sk", "env-st")


def test_resolve_aws_credentials_shared_file(tmp_path, monkeypatch):
    credentials_file = tmp_path / "credentials"
    credentials_file.write_text("[dev]\naws_access_key_id = file-ak\naws_secret_access_key = file-sk\n")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials_file))
    assert resolve_aws_credentials(profile="dev") == AwsCredentials("file-ak", "file-sk")
    # Config file uses the "profile X" section convention.
    config_file = tmp_path / "config"
    config_file.write_text("[profile prod]\naws_access_key_id = cfg-ak\naws_secret_access_key = cfg-sk\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))
    assert resolve_aws_credentials(profile="prod") == AwsCredentials("cfg-ak", "cfg-sk")


def test_resolve_aws_credentials_missing():
    assert resolve_aws_credentials(profile="nonexistent-profile") is None


# ---------------------------------------------------------------------------
# Event-stream codec
# ---------------------------------------------------------------------------


def test_eventstream_roundtrip():
    frame = _event("messageStart", {"role": "assistant"})
    message, rest = decode_message(frame)
    assert rest == b""
    assert message.message_type == "event"
    assert message.event_type == "messageStart"
    assert message.json_payload() == {"role": "assistant"}


def test_eventstream_incremental_feed_byte_by_byte():
    data = _event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hello"}}) + _event(
        "messageStop", {"stopReason": "end_turn"}
    )
    decoder = EventStreamDecoder()
    messages = []
    for byte in data:
        messages.extend(decoder.feed(bytes([byte])))
    decoder.flush()
    assert [m.event_type for m in messages] == ["contentBlockDelta", "messageStop"]


def test_eventstream_crc_mismatch_rejected():
    frame = bytearray(_event("messageStop", {"stopReason": "end_turn"}))
    frame[-1] ^= 0xFF  # corrupt the message CRC
    with pytest.raises(EventStreamError, match="CRC"):
        decode_message(bytes(frame))


def test_eventstream_flush_rejects_trailing_bytes():
    decoder = EventStreamDecoder()
    decoder.feed(b"\x00\x00")
    with pytest.raises(EventStreamError, match="undelivered"):
        decoder.flush()


# ---------------------------------------------------------------------------
# Adapter: payload conversion
# ---------------------------------------------------------------------------


def _context(*messages, system: str | None = None, tools=None):
    entries = []
    if system is not None:
        entries.append(SystemMessage(content=[TextContent(text=system)], timestamp=0))
    entries.extend(messages)
    return normalize_context(Context(messages=entries, tools=tools or []))


def test_convert_messages_merges_consecutive_tool_results():
    context = _context(
        UserMessage(content="run tools", timestamp=1),
        AssistantMessage(
            content=[ToolCall(id="call-1", name="a", arguments={}), ToolCall(id="call-2", name="b", arguments={})],
            api="bedrock-converse-stream",
            provider="amazon-bedrock",
            model="anthropic.claude-sonnet-4-5",
            timestamp=2,
        ),
        ToolResultMessage(tool_call_id="call-1", tool_name="a", content=[TextContent(text="one")], timestamp=3),
        ToolResultMessage(
            tool_call_id="call-2", tool_name="b", content=[TextContent(text="")], is_error=True, timestamp=4
        ),
    )
    messages = bedrock.convert_messages(context, MODEL, "none")
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    tool_results = messages[2]["content"]
    assert len(tool_results) == 2
    assert tool_results[0]["toolResult"]["toolUseId"] == "call-1"
    assert tool_results[0]["toolResult"]["status"] == "success"
    # Empty tool-result text becomes the placeholder, and errors are flagged.
    assert tool_results[1]["toolResult"]["content"] == [{"text": "<empty>"}]
    assert tool_results[1]["toolResult"]["status"] == "error"


def test_convert_messages_thinking_replay():
    thinking = ThinkingContent(thinking="deep thought", thinking_signature="sig-abc")
    context = _context(
        UserMessage(content="hi", timestamp=1),
        AssistantMessage(
            content=[thinking],
            api="bedrock-converse-stream",
            provider="amazon-bedrock",
            model="anthropic.claude-sonnet-4-5",
            timestamp=2,
        ),
    )
    messages = bedrock.convert_messages(context, MODEL, "none")
    block = messages[1]["content"][0]
    assert block == {"reasoningContent": {"reasoningText": {"text": "deep thought", "signature": "sig-abc"}}}

    # Missing signature falls back to plain text (Bedrock rejects unsigned reasoning).
    thinking2 = ThinkingContent(thinking="no sig", thinking_signature=None)
    context2 = _context(
        UserMessage(content="hi", timestamp=1),
        AssistantMessage(
            content=[thinking2],
            api="bedrock-converse-stream",
            provider="amazon-bedrock",
            model="anthropic.claude-sonnet-4-5",
            timestamp=2,
        ),
    )
    assert bedrock.convert_messages(context2, MODEL, "none")[1]["content"][0] == {"text": "no sig"}

    # Redacted thinking replays the opaque payload as redactedContent.
    redacted = ThinkingContent(
        thinking="[Reasoning redacted]", thinking_signature=base64.b64encode(b"opaque").decode(), redacted=True
    )
    context3 = _context(
        UserMessage(content="hi", timestamp=1),
        AssistantMessage(
            content=[redacted],
            api="bedrock-converse-stream",
            provider="amazon-bedrock",
            model="anthropic.claude-sonnet-4-5",
            timestamp=2,
        ),
    )
    block3 = bedrock.convert_messages(context3, MODEL, "none")[1]["content"][0]
    assert block3 == {"reasoningContent": {"redactedContent": b"opaque"}}


def test_convert_messages_cache_point_and_tool_call_id_normalization():
    long_id = "call" + "/x" * 40  # >64 chars after sanitizing
    context = _context(
        UserMessage(content="hi", timestamp=1),
        # Cross-model replay: tool call id normalization only applies when the
        # assistant message comes from a different provider/api/model.
        AssistantMessage(
            content=[ToolCall(id=long_id, name="t", arguments={"": 1, "ok": 2})],
            api="openai-completions",
            provider="openai",
            model="gpt-5",
            timestamp=2,
        ),
        ToolResultMessage(tool_call_id=long_id, tool_name="t", content=[TextContent(text="done")], timestamp=3),
    )
    # Claude models get a cache point on the last user message.
    messages = bedrock.convert_messages(context, MODEL, "short")
    tool_use = messages[1]["content"][0]["toolUse"]
    assert len(tool_use["toolUseId"]) <= 64
    assert "/" not in tool_use["toolUseId"]
    # Empty document keys are stripped.
    assert tool_use["input"] == {"ok": 2}
    last_user = messages[-1]
    assert last_user["content"][-1] == {"cachePoint": {"type": "default"}}
    # Long retention upgrades the ttl.
    messages_long = bedrock.convert_messages(context, MODEL, "long")
    assert messages_long[-1]["content"][-1] == {"cachePoint": {"type": "default", "ttl": "one_hour"}}
    # Retention "none" drops cache points.
    messages_none = bedrock.convert_messages(context, MODEL, "none")
    assert all("cachePoint" not in block for block in messages_none[-1]["content"])


def test_build_additional_model_request_fields_budget_claude():
    fields = bedrock._build_additional_model_request_fields(MODEL, _options(reasoning="medium"))
    assert fields == {
        "thinking": {"type": "enabled", "budget_tokens": 8192, "display": "summarized"},
        "anthropic_beta": ["interleaved-thinking-2025-05-14"],
    }
    # No reasoning requested → no fields.
    assert bedrock._build_additional_model_request_fields(MODEL, _options()) is None


def test_build_additional_model_request_fields_adaptive_claude():
    adaptive = MODEL.model_copy(update={"id": "anthropic.claude-opus-5", "name": "Claude Opus 5 (Bedrock)"})
    fields = bedrock._build_additional_model_request_fields(adaptive, _options(reasoning="high"))
    assert fields == {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "high"}}


def test_convert_tool_config():
    tool = Tool(name="search", description="Search the web", parameters={"type": "object", "properties": {}})
    config = bedrock.convert_tool_config([tool], "auto", False)
    assert config["toolChoice"] == {"auto": {}}
    assert config["tools"][0]["toolSpec"]["name"] == "search"
    assert config["tools"][0]["toolSpec"]["inputSchema"] == {"json": {"type": "object", "properties": {}}}
    assert bedrock.convert_tool_config([tool], "none", False) is None
    assert bedrock.convert_tool_config(None, "auto", False) is None
    named = bedrock.convert_tool_config([tool], {"type": "tool", "name": "search"}, False)
    assert named["toolChoice"] == {"tool": {"name": "search"}}


# ---------------------------------------------------------------------------
# Adapter: end-to-end streams
# ---------------------------------------------------------------------------


def test_text_stream_end_to_end():
    async def main():
        frames = [
            _event("messageStart", {"role": "assistant"}),
            _event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hello"}}),
            _event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": " world"}}),
            _event("contentBlockStop", {"contentBlockIndex": 0}),
            _event("messageStop", {"stopReason": "end_turn"}),
            _event(
                "metadata",
                {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15, "cacheReadInputTokens": 3}},
            ),
        ]
        with respx.mock:
            route = respx.post(CONVERSE_URL).mock(return_value=_converse_response(*frames))
            stream = bedrock.stream(
                MODEL,
                _context(UserMessage(content="hi", timestamp=1), system="You are helpful."),
                _options(),
            )
            events = [e async for e in stream]
            message = await stream.result()

        assert message.stop_reason == "stop"
        assert "".join(e.delta for e in events if e.type == "text_delta") == "Hello world"
        assert message.usage.input == 10
        assert message.usage.output == 5
        assert message.usage.cache_read == 3
        assert message.usage.total_tokens == 15

        request = route.calls[0].request
        authorization = request.headers["authorization"]
        assert authorization.startswith("AWS4-HMAC-SHA256 Credential=AKIDTEST/")
        assert "/us-east-1/bedrock/aws4_request" in authorization
        assert "content-type;host;x-amz-content-sha256;x-amz-date" in authorization
        body = json.loads(request.content)
        assert body["system"] == [{"text": "You are helpful."}, {"cachePoint": {"type": "default"}}]
        assert body["inferenceConfig"] == {"maxTokens": MODEL.max_tokens}
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"][-1] == {"cachePoint": {"type": "default"}}

    asyncio.run(main())


def test_bearer_token_auth_skips_sigv4():
    async def main():
        with respx.mock:
            route = respx.post(CONVERSE_URL).mock(
                return_value=_converse_response(
                    _event("messageStart", {"role": "assistant"}),
                    _event("messageStop", {"stopReason": "end_turn"}),
                    _event("metadata", {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}),
                )
            )
            stream = bedrock.stream(
                MODEL,
                _context(UserMessage(content="hi", timestamp=1)),
                _options(env={}, bearer_token="bedrock-bearer-1"),
            )
            [e async for e in stream]
            message = await stream.result()

        assert message.stop_reason == "stop"
        request = route.calls[0].request
        assert request.headers["authorization"] == "Bearer bedrock-bearer-1"
        assert "x-amz-date" not in request.headers

    asyncio.run(main())


def test_tool_call_stream():
    async def main():
        frames = [
            _event("messageStart", {"role": "assistant"}),
            _event(
                "contentBlockStart",
                {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "tu-1", "name": "search"}}},
            ),
            _event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"toolUse": {"input": '{"q": "we'}}}),
            _event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"toolUse": {"input": 'ather"}'}}}),
            _event("contentBlockStop", {"contentBlockIndex": 0}),
            _event("messageStop", {"stopReason": "tool_use"}),
            _event("metadata", {"usage": {"inputTokens": 7, "outputTokens": 3, "totalTokens": 10}}),
        ]
        with respx.mock:
            respx.post(CONVERSE_URL).mock(return_value=_converse_response(*frames))
            stream = bedrock.stream(MODEL, _context(UserMessage(content="hi", timestamp=1)), _options())
            events = [e async for e in stream]
            message = await stream.result()

        assert message.stop_reason == "toolUse"
        tool_call = next(e for e in events if e.type == "toolcall_end").tool_call
        assert tool_call.id == "tu-1"
        assert tool_call.name == "search"
        assert tool_call.arguments == {"q": "weather"}
        # Streaming scratch buffers never reach the final message.
        assert message.content[0].partial_json is None

    asyncio.run(main())


def test_thinking_stream_with_signature():
    async def main():
        frames = [
            _event("messageStart", {"role": "assistant"}),
            _event(
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "thinking..."}}},
            ),
            _event(
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig-1"}}},
            ),
            _event("contentBlockStop", {"contentBlockIndex": 0}),
            _event("contentBlockDelta", {"contentBlockIndex": 1, "delta": {"text": "answer"}}),
            _event("contentBlockStop", {"contentBlockIndex": 1}),
            _event("messageStop", {"stopReason": "end_turn"}),
            _event("metadata", {"usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}}),
        ]
        with respx.mock:
            respx.post(CONVERSE_URL).mock(return_value=_converse_response(*frames))
            stream = bedrock.stream(
                MODEL, _context(UserMessage(content="hi", timestamp=1)), _options(reasoning="medium")
            )
            events = [e async for e in stream]
            message = await stream.result()

        thinking = message.content[0]
        assert thinking.type == "thinking"
        assert thinking.thinking == "thinking..."
        assert thinking.thinking_signature == "sig-1"
        assert message.content[1].text == "answer"

    asyncio.run(main())


def test_redacted_thinking_stream():
    async def main():
        frames = [
            _event("messageStart", {"role": "assistant"}),
            _event(
                "contentBlockDelta",
                {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"redactedContent": base64.b64encode(b"blob-1").decode()}},
                },
            ),
            _event("contentBlockStop", {"contentBlockIndex": 0}),
            _event("messageStop", {"stopReason": "end_turn"}),
            _event("metadata", {"usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}}),
        ]
        with respx.mock:
            respx.post(CONVERSE_URL).mock(return_value=_converse_response(*frames))
            stream = bedrock.stream(MODEL, _context(UserMessage(content="hi", timestamp=1)), _options())
            [e async for e in stream]
            message = await stream.result()

        thinking = message.content[0]
        assert thinking.redacted is True
        assert thinking.thinking == "[Reasoning redacted]"
        # The opaque payload is base64-encoded into thinking_signature for replay.
        assert base64.b64decode(thinking.thinking_signature) == b"blob-1"

    asyncio.run(main())


def test_mid_stream_exception_maps_to_prefixed_error():
    async def main():
        frames = [
            _event("messageStart", {"role": "assistant"}),
            _exception("throttlingException", "rate exceeded"),
        ]
        with respx.mock:
            respx.post(CONVERSE_URL).mock(return_value=_converse_response(*frames))
            stream = bedrock.stream(MODEL, _context(UserMessage(content="hi", timestamp=1)), _options())
            events = [e async for e in stream]
            message = await stream.result()

        assert message.stop_reason == "error"
        assert message.error_message.startswith("Throttling error: rate exceeded")
        error_event = events[-1]
        assert error_event.type == "error"

    asyncio.run(main())


def test_http_error_json_type_parsing():
    async def main():
        with respx.mock:
            respx.post(CONVERSE_URL).mock(
                return_value=httpx.Response(
                    400,
                    json={"__type": "com.amazon.bedrock#ValidationException", "message": "bad input"},
                    headers={"x-amzn-requestid": "req-err"},
                )
            )
            stream = bedrock.stream(MODEL, _context(UserMessage(content="hi", timestamp=1)), _options())
            [e async for e in stream]
            message = await stream.result()

        assert message.stop_reason == "error"
        assert message.error_message.startswith("Validation error: 400: bad input")
        # Failure diagnostics carry status, error code, and request id.
        assert message.diagnostics and message.diagnostics[0].type == "bedrock_response_failure"
        assert message.diagnostics[0].details == {"status": 400, "errorCode": "ValidationException", "requestId": "req-err"}

    asyncio.run(main())


def test_stop_reason_mapping():
    assert bedrock._map_stop_reason("end_turn") == ("stop", None)
    assert bedrock._map_stop_reason("stop_sequence") == ("stop", None)
    assert bedrock._map_stop_reason("max_tokens") == ("length", None)
    assert bedrock._map_stop_reason("model_context_window_exceeded") == ("length", None)
    assert bedrock._map_stop_reason("tool_use") == ("toolUse", None)
    assert bedrock._map_stop_reason("content_filtered") == ("error", "Provider stopped with: content_filtered")
    assert bedrock._map_stop_reason(None) == ("error", None)


def test_region_and_endpoint_resolution():
    # ARN-embedded region wins.
    arn_model = MODEL.model_copy(update={"id": "arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/x"})
    options = _options(region="us-west-2")
    assert bedrock._resolve_region(arn_model, options, "us-east-1", True) == "eu-west-1"
    # Explicit option beats endpoint-derived.
    assert bedrock._resolve_region(MODEL, options, "us-east-1", True) == "us-west-2"
    # Endpoint-derived when nothing else configured.
    assert bedrock._resolve_region(MODEL, _options(), "eu-central-1", True) == "eu-central-1"
    # Default fallback.
    assert bedrock._resolve_region(MODEL, _options(), None, False) == "us-east-1"
    # Standard endpoint detection.
    assert bedrock.get_standard_bedrock_endpoint_region(ENDPOINT) == "us-east-1"
    assert bedrock.get_standard_bedrock_endpoint_region("https://vpc-custom.example.com") is None

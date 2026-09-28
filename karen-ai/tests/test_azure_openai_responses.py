"""Tests for the Azure OpenAI Responses adapter."""

import asyncio
import json

import httpx
import pytest
import respx

import karen_ai
from karen_ai import Context, Tool, UserMessage, normalize_context
from karen_ai.api import azure_openai_responses
from karen_ai.api.azure_openai_responses import (
    normalize_azure_base_url,
    parse_deployment_name_map,
    resolve_azure_config,
)
from karen_ai.providers import faux_model
from karen_ai.types import OpenAIResponsesCompat

BASE_URL = "https://my-resource.openai.azure.com/openai/v1"


def _model(**overrides):
    model = faux_model(provider="azure-openai-responses", api="azure-openai-responses")
    model.base_url = BASE_URL
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


def _sse(*payloads: dict) -> httpx.Response:
    body = "".join(f"data: {json.dumps(p)}\n\n" for p in payloads)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def _completed():
    return {
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        },
    }


def test_normalize_azure_base_url():
    assert normalize_azure_base_url("https://res.openai.azure.com") == "https://res.openai.azure.com/openai/v1"
    assert normalize_azure_base_url("https://res.openai.azure.com/openai") == "https://res.openai.azure.com/openai/v1"
    assert (
        normalize_azure_base_url("https://res.openai.azure.com/openai/v1/responses")
        == "https://res.openai.azure.com/openai/v1"
    )
    assert (
        normalize_azure_base_url("https://res.cognitiveservices.azure.com/")
        == "https://res.cognitiveservices.azure.com/openai/v1"
    )
    # Non-Azure hosts keep their path.
    assert normalize_azure_base_url("https://proxy.example.com/custom/v2") == "https://proxy.example.com/custom/v2"
    with pytest.raises(ValueError, match="Invalid Azure OpenAI base URL"):
        normalize_azure_base_url("not a url")


def test_parse_deployment_name_map():
    assert parse_deployment_name_map("gpt-5=dep-a, gpt-5-mini=dep-b") == {"gpt-5": "dep-a", "gpt-5-mini": "dep-b"}
    assert parse_deployment_name_map("") == {}
    assert parse_deployment_name_map("garbage,,=x") == {}


def test_resolve_azure_config_from_resource_name():
    model = _model()
    model.base_url = ""
    config = resolve_azure_config(
        model, azure_openai_responses.AzureOpenAIResponsesOptions(azure_resource_name="my-res")
    )
    assert config["base_url"] == "https://my-res.openai.azure.com/openai/v1"
    assert config["api_version"] == "v1"


def test_text_stream_with_deployment_and_api_key_header():
    async def main():
        model = _model()
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = request.headers
            captured["payload"] = json.loads(request.content)
            return _sse(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": [], "status": "in_progress"},
                },
                {"type": "response.output_text.delta", "output_index": 0, "delta": "hi"},
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                        "status": "completed",
                    },
                },
                _completed(),
            )

        with respx.mock:
            respx.post(url__startswith=f"{BASE_URL}/responses").mock(side_effect=side_effect)
            stream = azure_openai_responses.stream(
                model,
                normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
                azure_openai_responses.AzureOpenAIResponsesOptions(
                    api_key="azure-key", azure_deployment_name="my-deployment"
                ),
            )
            [e async for e in stream]
            message = await stream.result()

        assert message.stop_reason == "stop"
        assert "api-version=v1" in captured["url"]
        assert captured["headers"]["api-key"] == "azure-key"
        assert "authorization" not in captured["headers"]
        # The deployment name replaces the model id in the request body.
        assert captured["payload"]["model"] == "my-deployment"

    asyncio.run(main())


def test_reasoning_payload_and_strict_default():
    async def main():
        model = _model()
        model.reasoning = True
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return _sse(_completed())

        with respx.mock:
            respx.post(url__startswith=f"{BASE_URL}/responses").mock(side_effect=side_effect)
            stream = azure_openai_responses.stream_simple(
                model,
                normalize_context(
                    Context(
                        tools=[Tool(name="search", description="s", parameters={"type": "object"})],
                        messages=[UserMessage(content="hi", timestamp=1)],
                    )
                ),
                karen_ai.SimpleStreamOptions(api_key="azure-key", reasoning="high"),
            )
            [e async for e in stream]
            await stream.result()

        payload = captured["payload"]
        assert payload["reasoning"] == {"effort": "high", "summary": "auto"}
        assert payload["include"] == ["reasoning.encrypted_content"]
        # Azure defaults supportsStrictMode to true.
        assert payload["tools"][0]["strict"] is False

    asyncio.run(main())


def test_stream_simple_requires_auth():
    model = _model()
    with pytest.raises(ValueError, match="No API key"):
        azure_openai_responses.stream_simple(
            model,
            normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)])),
            None,
        )

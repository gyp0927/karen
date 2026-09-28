"""Tests for image generation (OpenRouter) and System One classifiers."""

import asyncio
import base64
import json

import httpx
import respx

from karen_ai import (
    ClassifierBoolQuestion,
    ClassifierChoiceQuestion,
    ClassifierContext,
    ClassifierModel,
    ClassifierScoreQuestion,
    ImageContent,
    ImagesContext,
    ModelCost,
    ProviderRequestOptions,
    TextContent,
)
from karen_ai.api import cloudflare_workers_ai_system_one, openrouter_images, typesafe_system_one
from karen_ai.images_api_registry import generate_images as registry_generate_images
from karen_ai.types import ImageModel

BASE_URL = "https://openrouter.test/api/v1"


def _image_model(**overrides):
    model = ImageModel(
        id="google/gemini-2.5-flash-image",
        name="img",
        api="openrouter-images",
        provider="openrouter",
        base_url=BASE_URL,
        input=["text", "image"],
        output=["image", "text"],
        cost=ModelCost(input=1.0, output=2.0, cache_read=0.5, cache_write=0.0),
    )
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


def test_openrouter_images_generates_text_and_images():
    async def main():
        model = _image_model()
        png_data = base64.b64encode(b"fake-png").decode()
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "gen-1",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "prompt_tokens_details": {"cached_tokens": 40},
                    },
                    "choices": [
                        {
                            "message": {
                                "content": "Here is your image.",
                                "images": [{"image_url": {"url": f"data:image/png;base64,{png_data}"}}],
                            }
                        }
                    ],
                },
            )

        with respx.mock:
            respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=side_effect)
            result = await openrouter_images.generate_images(
                model,
                ImagesContext(input=[TextContent(text="draw a cat")]),
                ProviderRequestOptions(api_key="or-key"),
            )

        assert result.stop_reason == "stop"
        assert result.response_id == "gen-1"
        assert result.output[0].type == "text"
        assert result.output[0].text == "Here is your image."
        assert result.output[1].type == "image"
        assert result.output[1].mime_type == "image/png"
        assert result.output[1].data == png_data
        assert result.usage.input == 60
        assert result.usage.cache_read == 40
        # modalities includes text because the model outputs text.
        assert captured["payload"]["modalities"] == ["image", "text"]
        assert captured["payload"]["stream"] is False

    asyncio.run(main())


def test_openrouter_images_modalities_without_text_output():
    async def main():
        model = _image_model()
        model.output = ["image"]
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "gen-1", "choices": []})

        with respx.mock:
            respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=side_effect)
            await openrouter_images.generate_images(
                model,
                ImagesContext(
                    input=[
                        TextContent(text="edit this"),
                        ImageContent(mime_type="image/png", data=base64.b64encode(b"in").decode()),
                    ]
                ),
                ProviderRequestOptions(api_key="or-key"),
            )

        assert captured["payload"]["modalities"] == ["image"]
        content = captured["payload"]["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "edit this"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    asyncio.run(main())


def test_openrouter_images_http_error_returns_error_result():
    async def main():
        model = _image_model()
        with respx.mock:
            respx.post(f"{BASE_URL}/chat/completions").mock(
                return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
            )
            result = await openrouter_images.generate_images(
                model,
                ImagesContext(input=[TextContent(text="draw")]),
                ProviderRequestOptions(api_key="bad", max_retries=0),
            )

        assert result.stop_reason == "error"
        assert "401" in result.error_message

    asyncio.run(main())


def test_images_api_registry_dispatch():
    async def main():
        model = _image_model()
        with respx.mock:
            respx.post(f"{BASE_URL}/chat/completions").mock(
                return_value=httpx.Response(200, json={"id": "gen-1", "choices": []})
            )
            result = await registry_generate_images(
                model,
                ImagesContext(input=[TextContent(text="draw")]),
                ProviderRequestOptions(api_key="or-key"),
            )
        assert result.stop_reason == "stop"

    asyncio.run(main())


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def _classifier_model(api: str, base_url: str) -> ClassifierModel:
    return ClassifierModel(
        id="typesafe/jev",
        name="jev",
        api=api,
        provider="typesafe",
        base_url=base_url,
        input=["text"],
        cost=ModelCost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
    )


def _context() -> ClassifierContext:
    return ClassifierContext(
        state={"transcript": "user asked about refunds"},
        questions={
            "intent": ClassifierChoiceQuestion(
                instructions="What does the user want?",
                criteria={"refund": "wants money back", "other": "anything else"},
            ),
            "urgency": ClassifierScoreQuestion(
                instructions="How urgent is this?",
                criteria=["immediate action needed", "can wait"],
            ),
            "resolved": ClassifierBoolQuestion(
                instructions="Is this resolved?",
                criteria={"true": "fully resolved", "false": "not resolved"},
            ),
        },
    )


def test_typesafe_system_one_classify():
    async def main():
        model = _classifier_model("typesafe-system-one", "https://typesafe.test/api")
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            captured["headers"] = request.headers
            return httpx.Response(
                200,
                json={
                    "answers": {
                        "intent": {
                            "type": "choice",
                            "choice": "refund",
                            "probabilities": {"refund": 0.9, "other": 0.1},
                            "confidence": 0.95,
                        },
                        "urgency": {"type": "score", "score": 7.5, "confidence": 0.8},
                        "resolved": {"type": "noul", "noul": 0.2},
                    }
                },
            )

        with respx.mock:
            respx.post("https://typesafe.test/api/systemone").mock(side_effect=side_effect)
            result = await typesafe_system_one.classify(
                model, _context(), ProviderRequestOptions(api_key="ts-key")
            )

        assert result.stop_reason == "stop"
        assert result.answers["intent"].choice == "refund"
        assert result.answers["intent"].probabilities == {"refund": 0.9, "other": 0.1}
        assert result.answers["urgency"].score == 7.5
        assert result.answers["resolved"].probability == 0.2
        # bool questions map to wire-level "noul".
        assert captured["payload"]["questions"]["resolved"]["type"] == "noul"
        assert captured["payload"]["model"] == "typesafe/jev"
        assert captured["payload"]["state"] == {"transcript": "user asked about refunds"}
        assert captured["headers"]["authorization"] == "Bearer ts-key"

    asyncio.run(main())


def test_cloudflare_system_one_envelope():
    async def main():
        model = _classifier_model(
            "cloudflare-workers-ai-system-one", "https://api.cloudflare.test/client/v4/accounts/abc/ai"
        )
        captured = {}

        def side_effect(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "result": {
                        "state": "Completed",
                        "result": {
                            "answers": {
                                "intent": {
                                    "type": "choice",
                                    "choice": "other",
                                    "probabilities": {"refund": 0.2, "other": 0.8},
                                    "confidence": 0.7,
                                },
                                "urgency": {"type": "score", "score": 3.0, "confidence": 0.6},
                                "resolved": {"type": "noul", "noul": 0.9},
                            }
                        },
                    },
                },
            )

        with respx.mock:
            respx.post("https://api.cloudflare.test/client/v4/accounts/abc/ai/run").mock(
                side_effect=side_effect
            )
            result = await cloudflare_workers_ai_system_one.classify(
                model, _context(), ProviderRequestOptions(api_key="cf-key")
            )

        assert result.stop_reason == "stop"
        assert result.answers["intent"].choice == "other"
        assert result.answers["resolved"].probability == 0.9
        # Cloudflare wraps the request: {model, input: {state, questions}}.
        assert captured["payload"]["model"] == "typesafe/jev"
        assert set(captured["payload"]["input"].keys()) == {"state", "questions"}

    asyncio.run(main())


def test_cloudflare_system_one_failure_envelope_returns_error():
    async def main():
        model = _classifier_model(
            "cloudflare-workers-ai-system-one", "https://api.cloudflare.test/client/v4/accounts/abc/ai"
        )
        with respx.mock:
            respx.post("https://api.cloudflare.test/client/v4/accounts/abc/ai/run").mock(
                return_value=httpx.Response(
                    200,
                    json={"success": False, "errors": [{"message": "model unavailable"}]},
                )
            )
            result = await cloudflare_workers_ai_system_one.classify(
                model, _context(), ProviderRequestOptions(api_key="cf-key", max_retries=0)
            )

        assert result.stop_reason == "error"
        assert "model unavailable" in result.error_message

    asyncio.run(main())


def test_classifier_missing_answer_returns_error():
    async def main():
        model = _classifier_model("typesafe-system-one", "https://typesafe.test/api")
        with respx.mock:
            respx.post("https://typesafe.test/api/systemone").mock(
                return_value=httpx.Response(200, json={"answers": {}})
            )
            result = await typesafe_system_one.classify(
                model, _context(), ProviderRequestOptions(api_key="ts-key", max_retries=0)
            )

        assert result.stop_reason == "error"
        assert "did not return an answer" in result.error_message

    asyncio.run(main())

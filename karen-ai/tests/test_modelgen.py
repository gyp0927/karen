"""Model-catalog generator tests (scripts/modelgen).

The generator normally talks to models.dev, OpenRouter, the Vercel AI Gateway,
Nvidia NIM and Radius; these tests drive the same code paths with synthetic
catalogs so the pipeline stays verifiable offline.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_models import (  # noqa: E402
    GeneratorOptions,
    build_catalogs,
    group_by_api,
    read_generator_options,
    split_catalogs,
)
from modelgen.compat import (  # noqa: E402
    apply_anthropic_allowed_fallback_model_metadata,
    apply_chat_metadata,
    detect_openai_completions_compat,
    get_anthropic_messages_compat,
    is_anthropic_fallback_metadata_model,
    openai_completions_compat_delta,
    supports_anthropic_mid_convo_effort,
)
from modelgen.model_data import (  # noqa: E402
    MODEL_DATA_MANIFEST_FILE,
    create_model_data_manifest,
    model_data_structure_hash,
    serialize_json,
    validate_model_data_directory,
)
from modelgen.openrouter import build_openrouter_catalog, get_openrouter_thinking_level_map  # noqa: E402
from modelgen.providers_dev import load_models_dev_data, process_baseten_models, process_fireworks_models  # noqa: E402
from modelgen.reasoning_options import get_effort_thinking_level_map  # noqa: E402
from modelgen.tables import COPILOT_STATIC_HEADERS  # noqa: E402


# ---------------------------------------------------------------------------
# Reasoning options
# ---------------------------------------------------------------------------

FULL_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")


def test_effort_thinking_level_map_from_effort_values():
    assert get_effort_thinking_level_map([{"type": "effort", "values": ["low", "high", "none"]}]) == {
        "off": "none",
        "minimal": None,
        "low": "low",
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": None,
    }


def test_effort_thinking_level_map_rejects_unmappable_values():
    # "default" and JSON null have no Pi equivalent, so a model advertising only
    # those gets no map at all.
    assert get_effort_thinking_level_map([{"type": "effort", "values": ["default", None]}]) is None
    assert get_effort_thinking_level_map([{"type": "toggle"}]) is None
    assert get_effort_thinking_level_map([]) is None
    assert get_effort_thinking_level_map(None) is None
    # A lone `none` still means "off is supported, nothing else is".
    assert get_effort_thinking_level_map([{"type": "effort", "values": ["none"]}]) == {
        "off": "none",
        **dict.fromkeys(FULL_LEVELS),
    }


def test_openrouter_thinking_level_map():
    assert get_openrouter_thinking_level_map(None) is None
    # A mandatory reasoner cannot be turned off.
    assert get_openrouter_thinking_level_map({"mandatory": True}) == {"off": None}
    assert get_openrouter_thinking_level_map({"mandatory": False}) is None
    assert get_openrouter_thinking_level_map({"supported_efforts": ["low", "high"], "mandatory": True}) == {
        "off": None,
        "minimal": None,
        "low": "low",
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": None,
    }


# ---------------------------------------------------------------------------
# Compat detection
# ---------------------------------------------------------------------------


def _model(**overrides):
    base = {
        "id": "some-model",
        "name": "Some Model",
        "api": "openai-completions",
        "provider": "openai",
        "baseUrl": "https://api.openai.com/v1",
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "reasoning": True,
        "contextWindow": 1000,
        "maxTokens": 100,
    }
    base.update(overrides)
    return base


def test_detect_openai_completions_compat_per_provider():
    deepseek = openai_completions_compat_delta(
        detect_openai_completions_compat(_model(provider="deepseek", baseUrl="https://api.deepseek.com"))
    )
    assert deepseek["maxTokensField"] == "max_tokens"
    assert deepseek["thinkingFormat"] == "deepseek"
    assert deepseek["requiresReasoningContentOnAssistantMessages"] is True
    assert deepseek["supportsStore"] is False

    openrouter = openai_completions_compat_delta(
        detect_openai_completions_compat(_model(provider="openrouter", baseUrl="https://openrouter.ai/api/v1"))
    )
    assert openrouter["thinkingFormat"] == "openrouter"
    assert openrouter["sendSessionAffinityHeaders"] is True
    # OpenRouter is non-standard, so developer role is off for third-party ids.
    assert openrouter["supportsDeveloperRole"] is False

    openrouter_anthropic = openai_completions_compat_delta(
        detect_openai_completions_compat(
            _model(provider="openrouter", id="anthropic/claude-sonnet-5", baseUrl="https://openrouter.ai/api/v1")
        )
    )
    # anthropic/* and openai/* ids keep the developer role, which is the
    # runtime default, so the delta leaves it unset.
    assert "supportsDeveloperRole" not in openrouter_anthropic
    assert openrouter_anthropic["cacheControlFormat"] == "anthropic"

    # Values matching the runtime defaults are omitted from the delta, so a
    # standard OpenAI-compatible provider only carries its strict-mode opt-in.
    plain = openai_completions_compat_delta(
        detect_openai_completions_compat(_model(provider="acme", baseUrl="https://api.acme.test/v1"))
    )
    assert plain == {"supportsStrictMode": True}


def test_anthropic_messages_compat():
    assert get_anthropic_messages_compat("anthropic", "claude-opus-5") == {
        "supportsMidConvoEffort": True,
        "supportsMidConvoSystemMessages": True,
        "supportsMidConvoToolChanges": True,
    }
    # OpenCode forwards plain system messages but not tool changes.
    assert get_anthropic_messages_compat("opencode", "claude-opus-5") == {"supportsMidConvoSystemMessages": True}
    # Eager tool input streaming is unsupported for these Copilot models.
    assert get_anthropic_messages_compat("github-copilot", "claude-sonnet-4") == {
        "supportsEagerToolInputStreaming": False
    }
    assert get_anthropic_messages_compat("xiaomi", "mimo-v2.6-pro") == {"allowEmptySignature": True}
    assert get_anthropic_messages_compat("acme", "whatever") is None
    # Only Opus 5 and the Fable/Mythos 5.1 family take mid-conversation effort.
    assert supports_anthropic_mid_convo_effort("claude-opus-5")
    assert supports_anthropic_mid_convo_effort("anthropic/claude-fable-5.1")
    assert not supports_anthropic_mid_convo_effort("claude-opus-4-8")


def test_chat_metadata_appliers():
    anthropic = _model(api="anthropic-messages", provider="anthropic", id="claude-opus-5")
    apply_chat_metadata(anthropic, {})
    assert anthropic["promptCache"] == {"short": 300, "long": 3600}
    assert anthropic["compat"]["supportsMidConvoEffort"] is True
    assert anthropic["thinkingLevelMap"]["off"] is None
    assert anthropic["compat"]["forceAdaptiveThinking"] is True
    assert anthropic["compat"]["supportsTemperature"] is False
    assert anthropic["compat"]["supportsStrictTools"] is True

    openai = _model(api="openai-responses", provider="openai", id="gpt-5.6-sol", input=["text", "image"])
    openai["cost"]["cacheWrite"] = 5
    apply_chat_metadata(openai, {})
    assert openai["compat"]["supportsStrictMode"] is True
    assert openai["compat"]["supportsToolSearch"] is True
    assert openai["compat"]["supportsAdditionalTools"] is True
    assert openai["compat"]["supportsExplicitPromptCacheMode"] is True
    assert openai["inputLimits"]["images"]["maxPerRequest"] == 1500
    assert openai["inputLimits"]["images"]["resize"]["maxWidth"] == 2000

    text_only = _model(api="openai-responses", provider="openai", id="gpt-5.4", input=["text"])
    apply_chat_metadata(text_only, {})
    # Image limits are only added for models that accept images.
    assert text_only.get("inputLimits") is None
    # gpt-5.x on the Responses API takes an explicit "none" effort.
    assert text_only["thinkingLevelMap"]["off"] == "none"


def test_models_dev_reasoning_options_flow_into_thinking_levels():
    model = _model(provider="acme", baseUrl="https://api.acme.test/v1", id="acme-reasoner")
    model["compat"] = {"thinkingFormat": "openai", "supportsReasoningEffort": True}
    apply_chat_metadata(model, {"acme:acme-reasoner": [{"type": "effort", "values": ["low", "high"]}]})
    assert model["thinkingLevelMap"] == {
        "off": None,
        "minimal": None,
        "low": "low",
        "medium": None,
        "high": "high",
        "xhigh": None,
        "max": None,
    }

    # Without direct effort support the recorded options are ignored.
    plain = _model(provider="acme", baseUrl="https://api.acme.test/v1", id="acme-chat")
    plain["compat"] = {"thinkingFormat": "qwen", "supportsReasoningEffort": True}
    apply_chat_metadata(plain, {"acme:acme-chat": [{"type": "effort", "values": ["low"]}]})
    assert "thinkingLevelMap" not in plain


def test_anthropic_allowed_fallback_models():
    fallback = _model(api="anthropic-messages", provider="anthropic", id="claude-opus-5")
    primary = _model(api="anthropic-messages", provider="anthropic", id="claude-fable-5")
    unrelated = _model(api="anthropic-messages", provider="anthropic", id="claude-sonnet-4-5")
    candidates = [model for model in (fallback, primary, unrelated) if is_anthropic_fallback_metadata_model(model)]
    assert [model["id"] for model in candidates] == ["claude-opus-5", "claude-fable-5"]

    apply_chat_metadata(fallback, {})
    apply_chat_metadata(primary, {})
    apply_anthropic_allowed_fallback_model_metadata(candidates)
    # Fable 5 takes mid-conversation effort, so its fallbacks are narrowed to
    # the ones that also do (Opus 4.8 drops out).
    assert [entry["model"] for entry in primary["compat"]["allowedFallbackModels"]] == ["claude-opus-5"]
    assert primary["compat"]["allowedFallbackModels"][0]["cost"] == fallback["cost"]
    assert primary["compat"]["allowedFallbackModels"][0]["provider"] == "anthropic"


# ---------------------------------------------------------------------------
# OpenRouter catalog
# ---------------------------------------------------------------------------


def test_build_openrouter_catalog():
    catalog = build_openrouter_catalog(
        listed=[
            {
                "id": "anthropic/claude-sonnet-5",
                "name": "Claude Sonnet 5",
                "supported_parameters": ["tools", "reasoning"],
                "architecture": {"modality": "text+image->text"},
                "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                "top_provider": {"context_length": 200000, "max_completion_tokens": 64000},
            },
            # No tool support -> dropped from the chat catalog.
            {"id": "no-tools", "supported_parameters": ["temperature"]},
        ],
        image_listed=[
            {
                "id": "black-forest-labs/flux",
                "name": "FLUX",
                "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
                "pricing": {"prompt": "0"},
            },
            {"id": "text-only", "architecture": {"output_modalities": ["text"]}},
        ],
        decision_listed=[
            {
                "id": "typesafe/jev",
                "name": "Jev",
                "architecture": {"input_modalities": ["text"], "output_modalities": ["decisions"]},
            }
        ],
    )

    assert [model["id"] for model in catalog["chat"]] == ["anthropic/claude-sonnet-5"]
    chat = catalog["chat"][0]
    assert chat["type"] == "chat"
    assert chat["api"] == "anthropic-messages"
    assert chat["baseUrl"] == "https://openrouter.ai/api"
    assert chat["cost"] == {"input": 3.0, "output": 15.0, "cacheRead": 0.0, "cacheWrite": 0.0}
    assert chat["contextWindow"] == 200000
    assert chat["maxTokens"] == 64000
    assert chat["input"] == ["text", "image"]
    assert chat["reasoning"] is True

    assert [model["id"] for model in catalog["images"]] == ["black-forest-labs/flux"]
    assert catalog["images"][0]["type"] == "image"
    assert catalog["images"][0]["output"] == ["image"]

    assert [model["id"] for model in catalog["classifiers"]] == ["typesafe/jev"]
    assert catalog["classifiers"][0]["api"] == "typesafe-system-one"


# ---------------------------------------------------------------------------
# models.dev mapping
# ---------------------------------------------------------------------------


def _models_dev_entry(**overrides):
    entry = {
        "name": "Some Model",
        "tool_call": True,
        "reasoning": True,
        "modalities": {"input": ["text"], "output": ["text"]},
        "limit": {"context": 128000, "output": 8192},
        "cost": {"input": 1.0, "output": 2.0, "cache_read": 0.1, "cache_write": 0.2},
    }
    entry.update(overrides)
    return entry


def test_load_models_dev_data_maps_providers():
    catalog = {
        "amazon-bedrock": {
            "models": {
                "anthropic.claude-sonnet-5": _models_dev_entry(),
                # Skipped: no streaming tool support.
                "ai21.jamba-1-5-large": _models_dev_entry(),
                # Skipped: no system message support.
                "mistral.mistral-7b-instruct-v0:2": _models_dev_entry(),
                # Skipped: inference-profile-only id.
                "anthropic.claude-opus-5": _models_dev_entry(),
                # Skipped: no tools.
                "amazon.nova-lite": _models_dev_entry(tool_call=False),
            }
        },
        "anthropic": {
            "models": {
                "claude-sonnet-5": _models_dev_entry(
                    reasoning_options=[{"type": "effort", "values": ["low", "high"]}]
                )
            }
        },
        "openai": {
            "models": {
                "gpt-5.5": _models_dev_entry(),
                # models.dev lists this alias, but OpenAI APIs reject it.
                "gpt-5.6": _models_dev_entry(),
            }
        },
        "github-copilot": {
            "models": {
                "claude-opus-5": _models_dev_entry(),
                "gpt-5.5": _models_dev_entry(),
                "kimi-k3": _models_dev_entry(),
                "deprecated-one": _models_dev_entry(status="deprecated"),
            }
        },
        "opencode": {
            "models": {
                "via-anthropic": _models_dev_entry(provider={"npm": "@ai-sdk/anthropic"}),
                "via-openai": _models_dev_entry(provider={"npm": "@ai-sdk/openai"}),
                "via-google": _models_dev_entry(provider={"npm": "@ai-sdk/google"}),
                "via-alibaba": _models_dev_entry(provider={"npm": "@ai-sdk/alibaba"}),
                "grok-build-0.1": _models_dev_entry(provider={"npm": "@ai-sdk/openai"}),
            }
        },
        "nvidia": {
            "models": {
                "meta/llama-3.3-70b": _models_dev_entry(),
                # Not present in the live NIM catalog.
                "meta/llama-9": _models_dev_entry(),
                # Present live, but unsupported.
                "deepseek-ai/deepseek-v4-pro": _models_dev_entry(),
            }
        },
    }
    recorder = {}
    models = load_models_dev_data(
        catalog,
        recorder,
        {
            "meta/llama-3.3-70b": "meta/llama-3.3-70b",
            "deepseek-ai/deepseek-v4-pro": "deepseek-ai/deepseek-v4-pro",
        },
    )
    by_key = {(model["provider"], model["id"]): model for model in models}

    assert [model["id"] for model in models if model["provider"] == "amazon-bedrock"] == [
        "anthropic.claude-sonnet-5"
    ]
    assert by_key[("amazon-bedrock", "anthropic.claude-sonnet-5")]["baseUrl"].endswith("us-east-1.amazonaws.com")
    assert by_key[("amazon-bedrock", "anthropic.claude-sonnet-5")]["cost"] == {
        "input": 1.0,
        "output": 2.0,
        "cacheRead": 0.1,
        "cacheWrite": 0.2,
    }

    assert by_key[("openai", "gpt-5.5")]["api"] == "openai-responses"
    assert ("openai", "gpt-5.6") not in by_key

    assert by_key[("github-copilot", "claude-opus-5")]["api"] == "anthropic-messages"
    assert by_key[("github-copilot", "gpt-5.5")]["api"] == "openai-responses"
    assert by_key[("github-copilot", "kimi-k3")]["compat"]["supportsReasoningEffort"] is False
    assert by_key[("github-copilot", "kimi-k3")]["headers"] == COPILOT_STATIC_HEADERS
    assert ("github-copilot", "deprecated-one") not in by_key

    assert by_key[("opencode", "via-anthropic")]["baseUrl"] == "https://opencode.ai/zen"
    assert by_key[("opencode", "via-openai")]["compat"] == {"sessionAffinityFormat": "openai-nosession"}
    assert by_key[("opencode", "via-google")]["api"] == "google-generative-ai"
    assert by_key[("opencode", "via-alibaba")]["compat"] == {
        "cacheControlFormat": "anthropic",
        "maxTokensField": "max_tokens",
    }
    assert by_key[("opencode", "grok-build-0.1")]["compat"]["supportsReasoningEffort"] is False

    assert by_key[("nvidia", "meta/llama-3.3-70b")]["headers"] == {"NVCF-POLL-SECONDS": "3600"}
    assert by_key[("nvidia", "meta/llama-3.3-70b")]["compat"]["maxTokensField"] == "max_tokens"
    assert ("nvidia", "meta/llama-9") not in by_key
    assert ("nvidia", "deepseek-ai/deepseek-v4-pro") not in by_key

    # Raw models.dev reasoning options are recorded for the metadata pass.
    assert recorder["anthropic:claude-sonnet-5"] == [{"type": "effort", "values": ["low", "high"]}]
    assert "openai:gpt-5.5" not in recorder


def test_load_models_dev_data_qwen_token_plan_allowlist():
    catalog = {
        "alibaba-token-plan": {
            "models": {
                "qwen3.7-max": _models_dev_entry(),
                "not-in-the-allowlist": _models_dev_entry(),
                # Retired preview id, excluded even though the catalog still lists it.
                "qwen3.8-max-preview": _models_dev_entry(),
            }
        }
    }
    models = load_models_dev_data(catalog, {}, {})
    # Without an allowlist the international plan takes the whole catalog; the
    # Individual view is narrowed to its verified ids.
    assert [(model["provider"], model["id"]) for model in models] == [
        ("qwen-token-plan", "qwen3.7-max"),
        ("qwen-token-plan", "not-in-the-allowlist"),
        ("qwen-token-plan-individual", "qwen3.7-max"),
    ]

    # --strict requires the Individual allowlist to be served in full.
    with pytest.raises(RuntimeError, match="model IDs do not match"):
        load_models_dev_data(
            {"alibaba-token-plan": {"models": {"qwen3.8-flash": _models_dev_entry()}}},
            {},
            {},
            strict=True,
        )


def test_process_fireworks_and_baseten_models():
    fireworks = process_fireworks_models(
        {
            "models": {
                "accounts/fireworks/models/glm-5p2": _models_dev_entry(),
                "accounts/fireworks/models/kimi-k3": _models_dev_entry(),
                "accounts/fireworks/models/qwen3p8-max": _models_dev_entry(
                    reasoning_options=[{"type": "effort", "values": ["low", "high"]}]
                ),
            }
        },
        {},
    )
    by_id = {model["id"]: model for model in fireworks}
    assert by_id["accounts/fireworks/models/glm-5p2"]["api"] == "openai-completions"
    assert by_id["accounts/fireworks/models/kimi-k3"]["compat"]["thinkingFormat"] == "openai"
    assert by_id["accounts/fireworks/models/qwen3p8-max"]["api"] == "anthropic-messages"
    assert by_id["accounts/fireworks/models/qwen3p8-max"]["compat"]["forceAdaptiveThinking"] is True
    assert by_id["accounts/fireworks/models/qwen3p8-max"]["compat"]["allowEmptySignature"] is True

    baseten = process_baseten_models(
        {
            "models": {
                "zai-org/GLM-5.2": _models_dev_entry(modalities={"input": ["text", "image"], "output": ["text"]}),
                "openai/gpt-oss-120b": _models_dev_entry(reasoning_options=[{"type": "toggle"}]),
            }
        }
    )
    by_id = {model["id"]: model for model in baseten}
    # GLM-5.2 is text-only on Baseten despite models.dev reporting image input.
    assert by_id["zai-org/GLM-5.2"]["input"] == ["text"]
    assert by_id["zai-org/GLM-5.2"]["compat"]["thinkingFormat"] == "baseten"
    assert by_id["openai/gpt-oss-120b"]["compat"]["thinkingFormat"] == "baseten"
    assert by_id["openai/gpt-oss-120b"]["thinkingLevelMap"]["off"] == "off"
    assert process_baseten_models(None) == []


# ---------------------------------------------------------------------------
# Generation options
# ---------------------------------------------------------------------------


def test_read_generator_options():
    options = read_generator_options(["--strict", "--data-only"])
    assert options.strict and options.data_only and not options.json_only

    options = read_generator_options(["--json-only", "--json-output", "out", "--pretty"])
    assert options.json_only and options.pretty and options.json_output_dir.name == "out"

    with pytest.raises(SystemExit, match="--json-only requires --json-output"):
        read_generator_options(["--json-only"])
    with pytest.raises(SystemExit, match="--data-only cannot be combined"):
        read_generator_options(["--data-only", "--json-only", "--json-output", "out"])
    with pytest.raises(SystemExit, match="Unknown argument: --nope"):
        read_generator_options(["--nope"])


# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------


def test_build_split_and_group_catalogs():
    models_dev = {"openai": {"models": {"gpt-5.5": _models_dev_entry()}}}
    providers = build_catalogs(
        GeneratorOptions(),
        models_dev,
        classifier_models=[
            {
                "type": "classifier",
                "id": "jev-latest",
                "name": "Jev",
                "api": "typesafe-system-one",
                "provider": "typesafe",
                "baseUrl": "https://api.typesafe.ai/v1/",
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 64000,
            }
        ],
        open_router_catalog={"chat": [], "images": [], "classifiers": []},
        ai_gateway_models=[],
        radius_models=[],
        nvidia_nim_model_ids={},
    )

    # The static appends land in the pipeline's output.
    assert "claude-opus-5-5" in providers["anthropic"]["chat"]
    assert "gpt-5-chat-latest" in providers["openai"]["chat"]
    assert "deepseek-flash" in providers["deepseek"]["chat"]
    assert "Ling-2.6-flash" in providers["ant-ling"]["chat"]
    assert "gpt-5.6-sol" in providers["openai-codex"]["chat"]
    assert "mistral-medium-3.5" in providers["mistral"]["chat"]
    assert providers["deepseek"]["chat"]["deepseek-flash"]["type"] == "chat"
    assert providers["typesafe"]["classifier"]["jev-latest"]["type"] == "classifier"
    assert providers["cloudflare-workers-ai"]["classifier"]["typesafe/jev"]["provider"] == "cloudflare-workers-ai"
    # Azure mirrors every direct OpenAI Responses model.
    assert providers["azure-openai-responses"]["chat"]["gpt-5.5"]["api"] == "azure-openai-responses"
    # Every emitted model carries the metadata appliers' output.
    assert providers["deepseek"]["chat"]["deepseek-flash"]["inputLimits"]["images"]["resize"]["maxWidth"] == 2000
    assert providers["anthropic"]["chat"]["claude-opus-5-5"]["promptCache"] == {"short": 300, "long": 3600}

    catalogs = split_catalogs(providers)
    assert catalogs["providerIds"] == sorted(catalogs["providerIds"])
    assert "claude-opus-5-5" in catalogs["chat"]["anthropic"]
    assert catalogs["all"]["anthropic"] == list(catalogs["chat"]["anthropic"].values())
    assert catalogs["image"] == {pid: {} for pid in catalogs["providerIds"]}

    generated, structure = group_by_api(catalogs, ["anthropic", "typesafe"])
    assert structure["anthropic"]["chat:claude-opus-5-5"] == "anthropic-messages"
    assert structure["typesafe"]["classifier:jev-latest"] == "typesafe-system-one"
    assert set(generated["anthropic"]) == {"anthropic-messages"}
    assert set(generated["anthropic"]["anthropic-messages"]) == set(structure["anthropic"])


# ---------------------------------------------------------------------------
# Manifest and validation
# ---------------------------------------------------------------------------


def test_serialize_json_matches_javascript_stringify():
    value = {"b": 1, "a": [1, 2], "c": None}
    assert serialize_json(value) == '{"b":1,"a":[1,2],"c":null}\n'
    assert serialize_json({"a": 1}, pretty=True) == '{\n  "a": 1\n}\n'


def test_model_data_manifest_hashes():
    structure = {"acme": {"chat:one": "openai-completions"}}
    content = '{"openai-completions":{}}\n'
    manifest = create_model_data_manifest(structure, {"acme.json": content}, "2026-01-01T00:00:00.000Z")
    assert manifest["schemaVersion"] == 6
    assert manifest["generatedAt"] == "2026-01-01T00:00:00.000Z"
    assert manifest["files"] == {"acme.json": hashlib.sha256(content.encode("utf-8")).hexdigest()}
    # The hash covers the structure only, so it is stable across formatting.
    assert model_data_structure_hash(structure) == model_data_structure_hash(
        {"acme": {"chat:one": "openai-completions"}}
    )
    assert model_data_structure_hash(structure) != model_data_structure_hash(
        {"acme": {"chat:one": "anthropic-messages"}}
    )


def _chat_model(**overrides):
    model = {
        "type": "chat",
        "id": "one",
        "name": "One",
        "api": "openai-completions",
        "provider": "acme",
        "baseUrl": "https://api.acme.test/v1",
        "input": ["text"],
        "reasoning": True,
        "cost": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 1000,
        "maxTokens": 100,
    }
    model.update(overrides)
    return model


def _write_catalog(directory, provider_id, models):
    """Write one catalog file plus a manifest covering it; returns the structure."""
    directory.mkdir(parents=True, exist_ok=True)
    groups = {}
    for key, model in models.items():
        groups.setdefault(model["api"], {})[key] = model
    content = serialize_json(groups)
    (directory / f"{provider_id}.json").write_text(content, encoding="utf-8", newline="\n")
    structure = {
        provider_id: {key: model["api"] for key, model in models.items()},
    }
    (directory / MODEL_DATA_MANIFEST_FILE).write_text(
        serialize_json(
            create_model_data_manifest(structure, {f"{provider_id}.json": content}, "2026-01-01T00:00:00.000Z")
        ),
        encoding="utf-8",
        newline="\n",
    )
    return structure


def test_validate_model_data_directory_round_trip(tmp_path):
    structure = _write_catalog(tmp_path, "acme", {"chat:one": _chat_model()})
    validate_model_data_directory(structure, tmp_path)

    # Truncating the catalog breaks both the file hash and the model ids.
    (tmp_path / "acme.json").write_text(serialize_json({"openai-completions": {}}), encoding="utf-8", newline="\n")
    with pytest.raises(RuntimeError) as error:
        validate_model_data_directory(structure, tmp_path)
    assert "does not match its manifest hash" in str(error.value)
    assert "model IDs do not match" in str(error.value)


def test_validate_model_data_directory_rejects_bad_entries(tmp_path):
    structure = _write_catalog(tmp_path, "acme", {"chat:one": _chat_model()})

    def rewrite(model):
        content = serialize_json({"openai-completions": {"chat:one": model}})
        (tmp_path / "acme.json").write_text(content, encoding="utf-8", newline="\n")

    for patch, expected in (
        ({"maxTokens": 0}, "has invalid maxTokens"),
        ({"reasoning": "yes"}, "has no reasoning boolean"),
        ({"provider": "other"}, "has provider 'other'"),
        ({"input": []}, "has invalid input modalities"),
        ({"cost": {"input": 1}}, "has invalid cost.output"),
        ({"output": ["image"]}, "has unsupported output modalities"),
    ):
        rewrite(_chat_model(**patch))
        with pytest.raises(RuntimeError, match=expected):
            validate_model_data_directory(structure, tmp_path)

    # A key that disagrees with the entry's own type/id is rejected.
    (tmp_path / "acme.json").write_text(
        serialize_json({"openai-completions": {"chat:two": _chat_model()}}), encoding="utf-8", newline="\n"
    )
    with pytest.raises(RuntimeError, match="mismatched type/id identity"):
        validate_model_data_directory(structure, tmp_path)


def test_validate_model_data_directory_requires_the_declared_schema(tmp_path):
    classifier = {
        "type": "classifier",
        "id": "one",
        "name": "One",
        "api": "typesafe-system-one",
        "provider": "acme",
        "baseUrl": "https://api.acme.test/v1",
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 1000,
    }
    structure = _write_catalog(tmp_path, "acme", {"classifier:one": classifier})
    manifest = json.loads((tmp_path / MODEL_DATA_MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["schemaVersion"] = 3
    (tmp_path / MODEL_DATA_MANIFEST_FILE).write_text(serialize_json(manifest), encoding="utf-8", newline="\n")
    with pytest.raises(RuntimeError, match="model data schema is 3, expected 6"):
        validate_model_data_directory(structure, tmp_path)


# ---------------------------------------------------------------------------
# The checked-in catalogs
# ---------------------------------------------------------------------------


def test_generated_catalogs_are_valid():
    """The vendored catalogs pass `scripts/check_model_data.py`."""
    from modelgen import PACKAGE_ROOT
    from modelgen.model_data import validate_generated_model_data

    validate_generated_model_data(PACKAGE_ROOT)


def test_manifest_matches_the_checked_in_catalogs():
    from modelgen import PACKAGE_ROOT
    from modelgen.model_data import data_dir, read_model_data_structure

    structure = read_model_data_structure(PACKAGE_ROOT)
    manifest = json.loads((data_dir(PACKAGE_ROOT) / MODEL_DATA_MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["structureHash"] == model_data_structure_hash(structure)
    assert set(manifest["files"]) == {f"{provider_id}.json" for provider_id in structure}


def test_generated_catalogs_load_through_model_catalog():
    """Every generated entry flattens into a typed model under its own id."""
    from karen_ai.model_catalog import flatten_all_model_catalog, list_catalog_provider_ids, load_catalog_groups

    for provider_id in list_catalog_provider_ids():
        groups = load_catalog_groups(provider_id)
        keys = [key for group in groups.values() for key in group]
        # The current generator repeats the entry type in the model key.
        assert all(key.split(":", 1)[0] in ("chat", "image", "classifier") for key in keys), provider_id
        models = flatten_all_model_catalog(provider_id)
        assert models, provider_id
        raw_ids = {model["id"] for group in groups.values() for model in group.values()}
        assert set(models) == raw_ids
        assert all(model.id for model in models.values())

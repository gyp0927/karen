#!/usr/bin/env python3
"""Generate the vendored model catalogs.

Port of pi-ai's `scripts/generate-models.ts`. Fetches models.dev, OpenRouter,
the Vercel AI Gateway, Nvidia NIM and Radius, folds in pi-ai's per-provider
corrections, and writes one JSON per provider plus a `.manifest.json` under
`src/karen_ai/providers/data/`.

Usage:
    python scripts/generate_models.py [--strict] [--data-only]
                                      [--json-only --json-output DIR] [--pretty]

- `--strict`          turn upstream failures and allowlist mismatches into errors
- `--data-only`       only regenerate providers that already have catalog data
- `--json-only`       skip the data directory and only write the JSON catalog
- `--json-output DIR` write `models.json`, `models.all.json`, `providers.json`
                      and `providers/*.json` under DIR
- `--pretty`          indent the generated JSON
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modelgen import PACKAGE_ROOT  # noqa: E402
from modelgen.compat import (  # noqa: E402
    apply_anthropic_allowed_fallback_model_metadata,
    apply_chat_metadata,
    apply_image_input_metadata,
    is_anthropic_fallback_metadata_model,
)
from modelgen.model_data import (  # noqa: E402
    MODEL_DATA_MANIFEST_FILE,
    create_model_data_manifest,
    data_dir,
    read_model_data_provider_ids,
    serialize_json,
    validate_generated_model_data,
    validate_model_data_directory,
)
from modelgen.providers_dev import (  # noqa: E402
    load_models_dev,
    load_models_dev_classifier_models,
    load_models_dev_data,
)
from modelgen.sources import (  # noqa: E402
    fetch_ai_gateway_models,
    fetch_nvidia_nim_model_ids,
    fetch_openrouter_models,
    fetch_radius_models,
)
from modelgen.tables import (  # noqa: E402
    ANT_LING_STATIC_MODELS,
    AZURE_CONTEXT_WINDOW_OVERRIDES,
    CLOUDFLARE_WORKERS_AI_CLASSIFIER_MODELS,
    CODEX_MODELS,
    DEEPSEEK_COMPAT,
    DEEPSEEK_STATIC_MODELS,
    GITHUB_COPILOT_EXTENDED_CONTEXT_MODELS,
    KIMI_K3_MAX_TOKENS,
    MINIMAX_DIRECT_SUPPORTED_IDS,
    MISSING_ANTHROPIC_MODEL,
    MISSING_COPILOT_MODELS,
    MISSING_OPENAI_MODELS,
    MISTRAL_STATIC_MODEL,
    OPENAI_LONG_CONTEXT_INPUT_THRESHOLD,
    OPENAI_LONG_CONTEXT_PRICING_MODEL_IDS,
    OPENAI_SHORT_CONTEXT_CAPPED_MODEL_IDS,
    OPENAI_STANDARD_COSTS,
    OPENROUTER_AUTO_MODEL,
    OPENROUTER_FUSION_MODEL,
    OPENROUTER_KIMI_K3_MODEL_IDS,
    QWEN_TOKEN_PLAN_PROVIDER_IDS,
    XAI_BUILTIN_EXCLUDED_MODEL_IDS,
    with_open_ai_long_context_pricing,
)

Model = Dict[str, Any]


@dataclass
class GeneratorOptions:
    strict: bool = False
    data_only: bool = False
    json_only: bool = False
    json_output_dir: Optional[Path] = None
    pretty: bool = False


def read_generator_options(argv: Sequence[str]) -> GeneratorOptions:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--json-output")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("-h", "--help", action="help", help="show this help message and exit")
    known, unknown = parser.parse_known_args(argv)
    if unknown:
        raise SystemExit(f"Unknown argument: {unknown[0]}")

    options = GeneratorOptions(
        strict=known.strict,
        data_only=known.data_only,
        json_only=known.json_only,
        json_output_dir=Path(known.json_output).resolve() if known.json_output else None,
        pretty=known.pretty,
    )
    if options.json_only and options.json_output_dir is None:
        raise SystemExit("--json-only requires --json-output")
    if options.data_only and (options.json_only or options.json_output_dir):
        raise SystemExit("--data-only cannot be combined with JSON catalog output")
    return options


# ---------------------------------------------------------------------------
# Temporary overrides for upstream metadata that has not caught up yet
# ---------------------------------------------------------------------------


def apply_temporary_overrides(models: Sequence[Model]) -> None:
    """Correct catalog entries until upstream model metadata is fixed."""
    for candidate in models:
        provider = candidate.get("provider")
        model_id = candidate.get("id", "")

        if provider == "github-copilot" and model_id in GITHUB_COPILOT_EXTENDED_CONTEXT_MODELS:
            candidate["contextWindow"] = 1000000

        # models.dev may list Opus 5.5 before its effort metadata is complete.
        if (provider == "anthropic" and model_id == "claude-opus-5-5") or (
            provider == "github-copilot" and model_id == "claude-opus-5.5"
        ):
            candidate["thinkingLevelMap"] = {
                **(candidate.get("thinkingLevelMap") or {}),
                "off": None,
                "minimal": None,
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "xhigh",
                "max": "max",
            }

        if provider in ("anthropic", "opencode", "opencode-go") and model_id in (
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-opus-4.6",
            "claude-sonnet-4.6",
        ):
            candidate["contextWindow"] = 1000000

        # OpenCode variants list Claude Sonnet 4/4.5 with 1M context, actual limit is 200K
        if provider in ("opencode", "opencode-go") and model_id in ("claude-sonnet-4-5", "claude-sonnet-4"):
            candidate["contextWindow"] = 200000
        if provider in ("opencode", "opencode-go") and model_id == "gpt-5.4":
            candidate["contextWindow"] = 272000
            candidate["maxTokens"] = 128000
        # Keep direct OpenAI requests in the short-context pricing tier by default. Users can opt
        # into the larger context through model overrides, so retain long-context cost metadata on
        # the capped models.
        if provider == "openai" and model_id in OPENAI_SHORT_CONTEXT_CAPPED_MODEL_IDS:
            candidate["contextWindow"] = OPENAI_LONG_CONTEXT_INPUT_THRESHOLD
            candidate["maxTokens"] = 128000
        if provider == "openai" and model_id in OPENAI_LONG_CONTEXT_PRICING_MODEL_IDS:
            standard_cost = OPENAI_STANDARD_COSTS.get(model_id)
            candidate["cost"] = with_open_ai_long_context_pricing(standard_cost or candidate["cost"])
        # Cloudflare AI Gateway passes OpenAI usage through at OpenAI list prices.
        if provider == "cloudflare-ai-gateway":
            standard_cost = OPENAI_STANDARD_COSTS.get(model_id)
            if standard_cost:
                candidate["cost"] = with_open_ai_long_context_pricing(standard_cost)
        # models.dev reports gpt-5-pro output as 272000 (a duplicate of the input sub-limit);
        # the actual max output is 128000. Also propagates to the derived Azure clone.
        if provider == "openai" and model_id == "gpt-5-pro":
            candidate["maxTokens"] = 128000
        # Keep Kimi K3's canonical output limit when gateway metadata is missing or incorrect.
        if (provider == "openrouter" and model_id in OPENROUTER_KIMI_K3_MODEL_IDS) or (
            provider == "vercel-ai-gateway" and model_id == "moonshotai/kimi-k3"
        ):
            candidate["maxTokens"] = KIMI_K3_MAX_TOKENS
        # Keep selected OpenRouter model metadata stable until upstream settles.
        if provider == "openrouter" and model_id == "moonshotai/kimi-k2.5":
            candidate["cost"]["input"] = 0.41
            candidate["cost"]["output"] = 2.06
            candidate["cost"]["cacheRead"] = 0.07
            candidate["maxTokens"] = 4096
        if provider == "openrouter" and model_id.startswith("moonshotai/kimi-k2.6"):
            candidate["compat"] = {
                **(candidate.get("compat") or {}),
                "supportsDeveloperRole": False,
                "requiresReasoningContentOnAssistantMessages": True,
            }
        if provider == "openrouter" and model_id == "z-ai/glm-5":
            candidate["cost"]["input"] = 0.6
            candidate["cost"]["output"] = 1.9
            candidate["cost"]["cacheRead"] = 0.119


def apply_deepseek_v4_compat(models: Sequence[Model]) -> None:
    """DeepSeek V4 reasoning-content handling per provider."""
    for candidate in models:
        if (
            candidate.get("api") == "openai-completions"
            and "deepseek-v4" in candidate.get("id", "")
            and candidate.get("provider") not in QWEN_TOKEN_PLAN_PROVIDER_IDS
        ):
            preserves_native_reasoning_effort = candidate.get("provider") in ("openrouter", "opencode")
            candidate["compat"] = {
                **(candidate.get("compat") or {}),
                **(
                    {
                        "requiresReasoningContentOnAssistantMessages": DEEPSEEK_COMPAT[
                            "requiresReasoningContentOnAssistantMessages"
                        ]
                    }
                    if preserves_native_reasoning_effort
                    else DEEPSEEK_COMPAT
                ),
            }


def drop_unsupported_minimax_models(models: List[Model]) -> None:
    """MiniMax's Anthropic-compatible endpoint only serves a few direct models."""
    for index in range(len(models) - 1, -1, -1):
        candidate = models[index]
        if (
            candidate.get("provider") in ("minimax", "minimax-cn")
            and candidate.get("id") not in MINIMAX_DIRECT_SUPPORTED_IDS
        ):
            del models[index]


def derive_azure_openai_models(models: Sequence[Model]) -> List[Model]:
    """Azure clones of every direct OpenAI Responses model."""
    return [
        {
            **model,
            "api": "azure-openai-responses",
            "provider": "azure-openai-responses",
            "baseUrl": "",
            "cost": {
                "input": model["cost"]["input"],
                "output": model["cost"]["output"],
                "cacheRead": model["cost"]["cacheRead"],
                "cacheWrite": model["cost"]["cacheWrite"],
            },
            "contextWindow": AZURE_CONTEXT_WINDOW_OVERRIDES.get(model["id"], model.get("contextWindow")),
        }
        for model in models
        if model.get("provider") == "openai" and model.get("api") == "openai-responses"
    ]


def _push_if_missing(models: List[Model], entry: Mapping[str, Any]) -> None:
    if not any(model.get("provider") == entry["provider"] and model.get("id") == entry["id"] for model in models):
        models.append(dict(entry))


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


def build_catalogs(
    options: GeneratorOptions,
    models_dev: Mapping[str, Any],
    classifier_models: Sequence[Model],
    open_router_catalog: Mapping[str, List[Model]],
    ai_gateway_models: Sequence[Model],
    radius_models: Sequence[Model],
    nvidia_nim_model_ids: Mapping[str, str],
) -> Dict[str, Dict[str, Dict[str, Model]]]:
    """Run the whole pipeline and return `{provider: {chat: {}, image: {}, classifier: {}}}`."""
    reasoning_options_by_model: Dict[str, Any] = {}
    models_dev_models = load_models_dev_data(models_dev, reasoning_options_by_model, nvidia_nim_model_ids, options.strict)

    # Combine chat models (models.dev has priority where sources overlap).
    all_models: List[Model] = [
        model
        for model in [
            *models_dev_models,
            *open_router_catalog["chat"],
            *ai_gateway_models,
            *radius_models,
        ]
        if not (
            (model.get("provider") == "xai" and model.get("id") in XAI_BUILTIN_EXCLUDED_MODEL_IDS)
            or (
                model.get("provider") in ("opencode", "opencode-go")
                and model.get("id") == "gpt-5.3-codex-spark"
            )
        )
    ]

    _push_if_missing(all_models, MISSING_ANTHROPIC_MODEL)
    for model in MISSING_COPILOT_MODELS:
        _push_if_missing(all_models, model)

    apply_temporary_overrides(all_models)

    for model in MISSING_OPENAI_MODELS:
        _push_if_missing(all_models, model)

    all_models.extend(dict(model) for model in DEEPSEEK_STATIC_MODELS)
    all_models.extend(dict(model) for model in ANT_LING_STATIC_MODELS)
    apply_deepseek_v4_compat(all_models)
    drop_unsupported_minimax_models(all_models)
    all_models.extend(dict(model) for model in CODEX_MODELS)

    _push_if_missing(all_models, MISTRAL_STATIC_MODEL)
    _push_if_missing(all_models, OPENROUTER_AUTO_MODEL)
    _push_if_missing(all_models, OPENROUTER_FUSION_MODEL)

    all_models.extend(derive_azure_openai_models(all_models))

    for model in all_models:
        apply_chat_metadata(model, reasoning_options_by_model)
    apply_anthropic_allowed_fallback_model_metadata(
        [model for model in all_models if is_anthropic_fallback_metadata_model(model)]
    )

    # Keep chat, image and classifier catalogs separate so one upstream id can
    # expose both operations with different API implementations.
    providers: Dict[str, Dict[str, Dict[str, Model]]] = {}
    for model in all_models:
        groups = providers.setdefault(model["provider"], {"chat": {}, "image": {}, "classifier": {}})
        # Only add if not already present (models.dev takes priority over OpenRouter).
        groups["chat"].setdefault(model["id"], {**model, "type": "chat"})
    for model in open_router_catalog["images"]:
        apply_image_input_metadata(model)
        groups = providers.setdefault(model["provider"], {"chat": {}, "image": {}, "classifier": {}})
        groups["image"].setdefault(model["id"], model)
    for model in [*classifier_models, *open_router_catalog["classifiers"], *CLOUDFLARE_WORKERS_AI_CLASSIFIER_MODELS]:
        groups = providers.setdefault(model["provider"], {"chat": {}, "image": {}, "classifier": {}})
        groups["classifier"].setdefault(model["id"], model)

    return providers


def split_catalogs(providers: Mapping[str, Mapping[str, Mapping[str, Model]]]) -> Dict[str, Any]:
    """Sort providers and their entries, and derive the per-type provider views."""
    sorted_provider_ids = sorted(providers)
    chat: Dict[str, Dict[str, Model]] = {}
    image: Dict[str, Dict[str, Model]] = {}
    classifier: Dict[str, Dict[str, Model]] = {}
    every: Dict[str, List[Model]] = {}
    for provider_id in sorted_provider_ids:
        groups = providers[provider_id]
        chat[provider_id] = {key: groups["chat"][key] for key in sorted(groups["chat"])}
        image[provider_id] = {key: groups["image"][key] for key in sorted(groups["image"])}
        classifier[provider_id] = {key: groups["classifier"][key] for key in sorted(groups["classifier"])}
        every[provider_id] = [
            *chat[provider_id].values(),
            *image[provider_id].values(),
            *classifier[provider_id].values(),
        ]
    return {
        "providerIds": sorted_provider_ids,
        "chat": chat,
        "image": image,
        "classifier": classifier,
        "all": every,
    }


def group_by_api(
    catalogs: Mapping[str, Any], provider_ids: Sequence[str]
) -> "tuple[Dict[str, Dict[str, Dict[str, Model]]], Dict[str, Dict[str, str]]]":
    """Only the ignored internal data is grouped by API, for type derivation."""
    generated: Dict[str, Dict[str, Dict[str, Model]]] = {}
    structure: Dict[str, Dict[str, str]] = {}
    for provider_id in provider_ids:
        models = catalogs["all"][provider_id]
        generated[provider_id] = {}
        structure[provider_id] = {}
        for api in sorted({model["api"] for model in models}):
            generated[provider_id][api] = {}
            for model in models:
                if model["api"] != api:
                    continue
                identity = f"{model['type']}:{model['id']}"
                if identity in generated[provider_id][api]:
                    raise RuntimeError(f"{provider_id}/{identity} has duplicate {api} catalog entries")
                generated[provider_id][api][identity] = model
                structure[provider_id][identity] = api
    return generated, structure


def write_data_directory(
    options: GeneratorOptions,
    generated: Mapping[str, Mapping[str, Mapping[str, Model]]],
    structure: Mapping[str, Mapping[str, str]],
    provider_ids: Sequence[str],
    generated_at: str,
) -> None:
    """Stage, validate and swap in the regenerated catalogs."""
    target = data_dir(PACKAGE_ROOT)
    staging_root = Path(tempfile.mkdtemp(prefix=".model-generation-", dir=target.parent))
    staged = staging_root / "data"
    previous = staging_root / "previous-data"
    try:
        staged.mkdir(parents=True, exist_ok=True)
        contents: Dict[str, str] = {}
        for provider_id in provider_ids:
            filename = f"{provider_id}.json"
            content = serialize_json(generated[provider_id], options.pretty)
            contents[filename] = content
            (staged / filename).write_text(content, encoding="utf-8", newline="\n")
        (staged / MODEL_DATA_MANIFEST_FILE).write_text(
            serialize_json(create_model_data_manifest(structure, contents, generated_at), options.pretty),
            encoding="utf-8",
            newline="\n",
        )
        validate_model_data_directory(structure, staged)

        had_previous = target.is_dir()
        if had_previous:
            target.rename(previous)
        try:
            staged.rename(target)
            validate_generated_model_data(PACKAGE_ROOT)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            if had_previous and previous.is_dir():
                previous.rename(target)
            raise
        print(
            "Hydrated JSON model values under src/karen_ai/providers/data/"
            if options.data_only
            else "Generated JSON model values under src/karen_ai/providers/data/"
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def write_json_catalog(options: GeneratorOptions, catalogs: Mapping[str, Any]) -> None:
    """`--json-output`: the legacy keyed chat catalog plus the all-types views."""
    output_dir = options.json_output_dir
    assert output_dir is not None
    provider_output_dir = output_dir / "providers"
    shutil.rmtree(output_dir, ignore_errors=True)
    provider_output_dir.mkdir(parents=True, exist_ok=True)

    def write(path: Path, value: Any) -> None:
        path.write_text(serialize_json(value, options.pretty), encoding="utf-8", newline="\n")

    write(output_dir / "models.json", catalogs["chat"])
    write(output_dir / "models.all.json", catalogs["all"])
    write(output_dir / "providers.json", catalogs["providerIds"])
    for provider_id in catalogs["providerIds"]:
        write(provider_output_dir / f"{provider_id}.json", catalogs["chat"][provider_id])
        write(provider_output_dir / f"{provider_id}.all.json", catalogs["all"][provider_id])
    print(f"Generated JSON model catalog under {output_dir}")


def print_statistics(catalogs: Mapping[str, Any]) -> None:
    all_models = [model for models in catalogs["all"].values() for model in models]
    print("\nModel Statistics:")
    print(f"  Total tool-capable models: {len(all_models)}")
    print(f"  Reasoning-capable models: {sum(1 for model in all_models if model.get('reasoning'))}")
    for provider_id in catalogs["providerIds"]:
        print(
            f"  {provider_id}: {len(catalogs['chat'][provider_id])} chat models, "
            f"{len(catalogs['image'][provider_id])} image models, "
            f"{len(catalogs['classifier'][provider_id])} classifier models"
        )


def generate_models(options: GeneratorOptions) -> None:
    models_dev = load_models_dev(options.strict)
    classifier_models = load_models_dev_classifier_models(options.strict)
    open_router_catalog = fetch_openrouter_models(options.strict)
    ai_gateway_models = fetch_ai_gateway_models(options.strict)
    radius_models = fetch_radius_models(options.strict)
    nvidia_nim_model_ids = fetch_nvidia_nim_model_ids(options.strict) if models_dev.get("nvidia") else {}

    providers = build_catalogs(
        options,
        models_dev,
        classifier_models,
        open_router_catalog,
        ai_gateway_models,
        radius_models,
        nvidia_nim_model_ids,
    )
    catalogs = split_catalogs(providers)

    generated_provider_ids = (
        read_model_data_provider_ids(PACKAGE_ROOT) if options.data_only else catalogs["providerIds"]
    )
    missing_provider_ids = [pid for pid in generated_provider_ids if pid not in catalogs["all"]]
    if missing_provider_ids:
        raise RuntimeError(f"Cannot hydrate missing providers: {', '.join(missing_provider_ids)}")

    generated, structure = group_by_api(catalogs, generated_provider_ids)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    if not options.json_only:
        write_data_directory(options, generated, structure, generated_provider_ids, generated_at)
    if options.json_output_dir:
        write_json_catalog(options, catalogs)

    print_statistics(catalogs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = read_generator_options(sys.argv[1:] if argv is None else argv)
    generate_models(options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

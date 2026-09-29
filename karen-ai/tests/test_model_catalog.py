"""Model catalog data layer tests (vendored pi-ai providers/data/*.json)."""

from karen_ai.model_catalog import (
    catalog_generated_at,
    flatten_all_model_catalog,
    flatten_chat_model_catalog,
    flatten_classifier_model_catalog,
    flatten_image_model_catalog,
    list_catalog_provider_ids,
    load_catalog_groups,
)


def test_all_catalogs_parse():
    ids = list_catalog_provider_ids()
    assert len(ids) == 42
    total = 0
    for provider_id in ids:
        models = flatten_all_model_catalog(provider_id)
        assert models, provider_id
        for model in models.values():
            assert model.provider == provider_id
            assert model.id
            assert model.api
        total += len(models)
    assert total > 1400


def test_catalog_groups_cached():
    assert load_catalog_groups("groq") is load_catalog_groups("groq")


def test_chat_entries_carry_costs_and_limits():
    model = flatten_chat_model_catalog("groq")["llama-3.3-70b-versatile"]
    assert model.base_url == "https://api.groq.com/openai/v1"
    assert model.cost.input > 0
    assert model.context_window == 131072
    assert model.compat and model.compat.supports_strict_mode is True


def test_anthropic_compat_richness():
    # Generated catalogs carry hand-tuned compat from the generator overrides.
    model = flatten_chat_model_catalog("anthropic")["claude-opus-5"]
    compat = model.compat
    assert type(compat).__name__ == "AnthropicMessagesCompat"
    assert compat.force_adaptive_thinking is True
    assert compat.supports_temperature is False


def test_openrouter_image_and_classifier_groups():
    images = flatten_image_model_catalog("openrouter")
    assert len(images) == 55
    sample = images["black-forest-labs/flux.2-flex"]
    assert sample.type == "image"
    assert sample.api == "openrouter-images"

    classifiers = flatten_classifier_model_catalog("openrouter")
    assert classifiers["typesafe/jev"].api == "typesafe-system-one"


def test_typesafe_and_cloudflare_classifier_catalogs():
    typesafe = flatten_classifier_model_catalog("typesafe")
    assert list(typesafe) == ["jev-latest"]
    assert typesafe["jev-latest"].base_url == "https://api.typesafe.ai/v1/"

    cloudflare = flatten_classifier_model_catalog("cloudflare-workers-ai")
    assert "typesafe/jev" in cloudflare
    assert "{CLOUDFLARE_ACCOUNT_ID}" in cloudflare["typesafe/jev"].base_url


def test_flatten_all_merges_types():
    merged = flatten_all_model_catalog("openrouter")
    kinds = {getattr(m, "type", "chat") or "chat" for m in merged.values()}
    assert kinds == {"chat", "image", "classifier"}


def test_catalog_generated_at_from_manifest():
    assert catalog_generated_at() == 1790105504346

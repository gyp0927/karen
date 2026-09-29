"""Static model catalogs, mirroring pi-ai's model-catalog.ts + providers/data/*.json.

pi-ai generates one JSON per provider from models.dev (plus a few hand-maintained
sources) at build time; the generated files ship with the package. karen-ai vendors
the same JSONs under `karen_ai/providers/data/` and flattens them into typed
models on first use.

Catalog JSON shape: `{ "<api>": { "<model-key>": { ...model fields } } }`. The
current generator keys entries as `"<type>:<id>"` and repeats `type` inside each
model; catalogs vendored from older pi-ai releases key chat models by bare id and
omit `type` for them. Both shapes flatten to the same model ids, which are read
from the entry's own `id` field.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from .types import AnyModel, ClassifierModel, ImageModel, Model, ProviderId

ModelGroups = Dict[str, Dict[str, Dict[str, Any]]]

_DATA_DIR = Path(__file__).parent / "providers" / "data"


@lru_cache(maxsize=None)
def load_catalog_groups(provider_id: str) -> ModelGroups:
    """Raw `{api: {model_id: model}}` groups for a provider, cached."""
    path = _DATA_DIR / f"{provider_id}.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def catalog_available(provider_id: str) -> bool:
    return (_DATA_DIR / f"{provider_id}.json").is_file()


def _model_id(key: str, model: Dict[str, Any]) -> str:
    """The catalog entry's model id, whatever shape the key uses.

    Generated catalogs key entries as `"<type>:<id>"`; older ones keyed chat
    models by bare id. Either way the entry's own `id` field is authoritative,
    with the key as a fallback for hand-edited data.
    """
    raw_id = model.get("id")
    if isinstance(raw_id, str) and raw_id:
        return raw_id
    entry_type = model.get("type")
    if isinstance(entry_type, str) and key.startswith(f"{entry_type}:"):
        return key[len(entry_type) + 1 :]
    return key


def _flatten(groups: ModelGroups, model_type: str) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for models in groups.values():
        for key, model in models.items():
            entry_type = model.get("type", "chat")
            if entry_type == model_type:
                merged[_model_id(key, model)] = model
    return merged


def flatten_chat_model_catalog(provider_id: ProviderId, groups: Optional[ModelGroups] = None) -> Dict[str, Model]:
    """Every chat model in the provider's catalog, keyed by model id."""
    groups = groups if groups is not None else load_catalog_groups(provider_id)
    return {model_id: Model.model_validate(model) for model_id, model in _flatten(groups, "chat").items()}


def flatten_image_model_catalog(provider_id: ProviderId, groups: Optional[ModelGroups] = None) -> Dict[str, ImageModel]:
    groups = groups if groups is not None else load_catalog_groups(provider_id)
    return {model_id: ImageModel.model_validate(model) for model_id, model in _flatten(groups, "image").items()}


def flatten_classifier_model_catalog(
    provider_id: ProviderId, groups: Optional[ModelGroups] = None
) -> Dict[str, ClassifierModel]:
    groups = groups if groups is not None else load_catalog_groups(provider_id)
    return {model_id: ClassifierModel.model_validate(model) for model_id, model in _flatten(groups, "classifier").items()}


def flatten_all_model_catalog(provider_id: ProviderId, groups: Optional[ModelGroups] = None) -> Dict[str, AnyModel]:
    """Chat + image + classifier models, keyed by model id."""
    groups = groups if groups is not None else load_catalog_groups(provider_id)
    merged: Dict[str, AnyModel] = {}
    merged.update(flatten_chat_model_catalog(provider_id, groups))
    merged.update(flatten_image_model_catalog(provider_id, groups))
    merged.update(flatten_classifier_model_catalog(provider_id, groups))
    return merged


@lru_cache(maxsize=1)
def catalog_generated_at() -> Optional[int]:
    """Generation timestamp of the vendored catalog data, in epoch milliseconds."""
    path = _DATA_DIR / ".manifest.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    raw = manifest.get("generatedAt")
    if not raw:
        return None
    try:
        return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def list_catalog_provider_ids() -> list[str]:
    """Provider ids with a vendored catalog file."""
    return sorted(path.stem for path in _DATA_DIR.glob("*.json") if not path.stem.startswith("."))

"""Generated model data: manifest, structure hash and validation.

Port of pi-ai's `scripts/model-data.ts`, including `check-model-data.ts`'s
`validateGeneratedModelData` entry point.

Two layout differences from pi-ai:

- pi-ai derives the provider list for `--data-only` hydration from its generated
  TypeScript aggregator (`src/models.generated.ts` imports); karen-ai reads the
  catalog files that are on disk instead, and cross-checks them against the
  built-in provider registry when `karen_ai` is importable.
- pi-ai also writes per-provider `.models.ts` shards next to the JSON. karen-ai
  has no shards, so there is nothing to regenerate or delete there.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

MODEL_DATA_SCHEMA_VERSION = 6
MODEL_DATA_MANIFEST_FILE = ".manifest.json"

#: `{provider_id: {"<type>:<model_id>": "<api>"}}`
ModelDataStructure = Dict[str, Dict[str, str]]
#: `{api: {"<type>:<model_id>": model}}`
ProviderGroups = Dict[str, Dict[str, Dict[str, Any]]]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sorted_record(entries: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    return {key: value for key, value in sorted(entries, key=lambda item: item[0])}


def same_strings(a: Sequence[str], b: Sequence[str]) -> bool:
    return len(a) == len(b) and all(left == right for left, right in zip(a, b))


def describe_set_difference(expected: Sequence[str], actual: Sequence[str]) -> str:
    expected_set, actual_set = set(expected), set(actual)
    missing = [value for value in expected if value not in actual_set]
    extra = [value for value in actual if value not in expected_set]
    parts = []
    if missing:
        parts.append(f"missing: {', '.join(missing)}")
    if extra:
        parts.append(f"extra: {', '.join(extra)}")
    return "; ".join(parts)


def assert_exact_model_ids(label: str, expected: Iterable[str], actual: Iterable[str]) -> None:
    """Raise unless the generated model ids match a verified allowlist exactly."""
    expected_ids = sorted(set(expected))
    actual_ids = sorted(set(actual))
    if same_strings(expected_ids, actual_ids):
        return
    raise RuntimeError(f"{label} model IDs do not match ({describe_set_difference(expected_ids, actual_ids)})")


def serialize_json(value: Any, pretty: bool = False) -> str:
    """Match `JSON.stringify(value, null, pretty ? 2 : undefined) + "\\n"`."""
    text = json.dumps(value, ensure_ascii=False, indent=2 if pretty else None, separators=None if pretty else (",", ":"))
    return f"{text}\n"


def is_record(value: Any) -> bool:
    return isinstance(value, dict)


def is_modality_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(entry in ("text", "image") for entry in value)


# ---------------------------------------------------------------------------
# Reading the current structure
# ---------------------------------------------------------------------------


def data_dir(package_root: Path) -> Path:
    return Path(package_root) / "src" / "karen_ai" / "providers" / "data"


def read_model_data_provider_ids(package_root: Path) -> List[str]:
    """Provider ids that have generated catalog data on disk."""
    directory = data_dir(package_root)
    if not directory.is_dir():
        raise RuntimeError(f"No generated model data directory at {directory}")
    provider_ids = sorted(path.stem for path in directory.glob("*.json") if not path.stem.startswith("."))
    if not provider_ids:
        raise RuntimeError(f"No generated provider catalogs found in {directory}")
    return provider_ids


def builtin_provider_ids() -> Optional[List[str]]:
    """Provider ids from the built-in registry, or None if `karen_ai` is unavailable."""
    try:
        from karen_ai.providers import builtin_providers
    except Exception:  # pragma: no cover - only when run outside the project venv
        return None
    return sorted(provider.id for provider in builtin_providers())


def read_provider_structure(path: Path, provider_id: str) -> Dict[str, str]:
    """`{"<type>:<model_id>": api}` for one provider catalog file."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{provider_id}.json is not valid JSON: {error}") from error
    if not is_record(raw):
        raise RuntimeError(f"{provider_id}.json must contain a JSON object")

    models: Dict[str, str] = {}
    for api, group in raw.items():
        if not is_record(group):
            raise RuntimeError(f"{path} API group {api!r} must be an object")
        for model_key in group:
            if model_key in models:
                raise RuntimeError(f"{path} contains {model_key} in more than one API group")
            models[model_key] = api
    if not models:
        raise RuntimeError(f"{path} contains no generated model data")
    return sorted_record(models.items())


def read_model_data_structure(package_root: Path) -> ModelDataStructure:
    """The generated structure of every catalog on disk."""
    directory = data_dir(package_root)
    provider_ids = read_model_data_provider_ids(package_root)

    registered = builtin_provider_ids()
    if registered is not None and not same_strings(sorted(registered), sorted(provider_ids)):
        raise RuntimeError(
            "Built-in providers and generated catalogs do not match "
            f"({describe_set_difference(registered, provider_ids)})"
        )

    return sorted_record(
        (provider_id, read_provider_structure(directory / f"{provider_id}.json", provider_id))
        for provider_id in provider_ids
    )


def model_data_structure_hash(structure: Mapping[str, Mapping[str, str]]) -> str:
    normalized = sorted_record(
        (provider_id, sorted_record(models.items())) for provider_id, models in structure.items()
    )
    return sha256(json.dumps(normalized, ensure_ascii=False))


def create_model_data_manifest(
    structure: Mapping[str, Mapping[str, str]], file_contents: Mapping[str, str], generated_at: str
) -> Dict[str, Any]:
    return {
        "schemaVersion": MODEL_DATA_SCHEMA_VERSION,
        "generatedAt": generated_at,
        "structureHash": model_data_structure_hash(structure),
        "files": sorted_record((filename, sha256(content)) for filename, content in file_contents.items()),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_model_value(
    value: Any, provider_id: str, model_id: str, expected_api: str, errors: List[str]
) -> None:
    label = f"{provider_id}/{model_id}"
    if not is_record(value):
        errors.append(f"{label} must be an object")
        return
    if value.get("id") != model_id:
        errors.append(f"{label} has id {value.get('id')!r}, expected {model_id!r}")
    if value.get("provider") != provider_id:
        errors.append(f"{label} has provider {value.get('provider')!r}, expected {provider_id!r}")
    if value.get("api") != expected_api:
        errors.append(f"{label} has api {value.get('api')!r}, expected {expected_api!r}")
    if not isinstance(value.get("name"), str) or not value["name"]:
        errors.append(f"{label} has no model name")
    if not isinstance(value.get("baseUrl"), str):
        errors.append(f"{label} has no baseUrl string")
    if not is_modality_list(value.get("input")):
        errors.append(f"{label} has invalid input modalities")

    model_type = value.get("type")
    if model_type == "image":
        if not is_modality_list(value.get("output")) or "image" not in value["output"]:
            errors.append(f"{label} has invalid output modalities")
    elif "output" in value:
        errors.append(f"{label} has unsupported output modalities")

    if model_type == "chat":
        if not isinstance(value.get("reasoning"), bool):
            errors.append(f"{label} has no reasoning boolean")
        if not _is_positive_number(value.get("contextWindow")):
            errors.append(f"{label} has invalid contextWindow")
        if not _is_positive_number(value.get("maxTokens")):
            errors.append(f"{label} has invalid maxTokens")
    elif model_type == "classifier":
        if not _is_positive_number(value.get("contextWindow")):
            errors.append(f"{label} has invalid contextWindow")
    elif model_type != "image":
        errors.append(f'{label} has type {model_type!r}, expected "chat", "image", or "classifier"')

    cost = value.get("cost")
    if not is_record(cost):
        errors.append(f"{label} has invalid cost metadata")
    else:
        for field in ("input", "output", "cacheRead", "cacheWrite"):
            if not _is_finite_number(cost.get(field)):
                errors.append(f"{label} has invalid cost.{field}")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and value not in (
        float("inf"),
        float("-inf"),
    )


def _is_positive_number(value: Any) -> bool:
    return _is_finite_number(value) and value > 0


def throw_validation_errors(errors: List[str]) -> None:
    visible = errors[:30]
    suffix = f"\n  ... and {len(errors) - len(visible)} more" if len(errors) > len(visible) else ""
    body = "\n".join(f"  - {error}" for error in visible)
    raise RuntimeError(f"Invalid generated model data:\n{body}{suffix}")


def _read_json_object(path: Path, description: str, errors: List[str]) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{description} is not valid JSON: {error}")
        return None
    if not is_record(parsed):
        errors.append(f"{description} must contain a JSON object")
        return None
    return parsed


def validate_model_data_directory(structure: Mapping[str, Mapping[str, str]], directory: Path) -> None:
    directory = Path(directory)
    if not directory.is_dir():
        raise RuntimeError(f"Generated model data directory does not exist: {directory}")

    errors: List[str] = []
    expected_files = sorted(f"{provider_id}.json" for provider_id in structure)
    actual_files = sorted(
        entry.name
        for entry in directory.glob("*.json")
        if entry.name != MODEL_DATA_MANIFEST_FILE
    )
    if not same_strings(expected_files, actual_files):
        errors.append(
            "provider data files do not match the generated catalog "
            f"({describe_set_difference(expected_files, actual_files)})"
        )

    manifest = _read_json_object(directory / MODEL_DATA_MANIFEST_FILE, "model data manifest", errors)
    if (manifest or {}).get("schemaVersion") != MODEL_DATA_SCHEMA_VERSION:
        errors.append(
            f"model data schema is {(manifest or {}).get('schemaVersion')!r}, "
            f"expected {MODEL_DATA_SCHEMA_VERSION}"
        )
    generated_at = (manifest or {}).get("generatedAt")
    if not isinstance(generated_at, str) or not _is_valid_timestamp(generated_at):
        errors.append("model data manifest has an invalid generation timestamp")
    if (manifest or {}).get("structureHash") != model_data_structure_hash(structure):
        errors.append("model data generation stamp does not match the generated catalog")

    manifest_files = (manifest or {}).get("files")
    if not is_record(manifest_files):
        errors.append("model data manifest has no file hashes")
        manifest_files = None
    else:
        manifest_file_names = sorted(manifest_files)
        if not same_strings(expected_files, manifest_file_names):
            errors.append(
                "manifest file hashes do not match provider data files "
                f"({describe_set_difference(expected_files, manifest_file_names)})"
            )

    for provider_id, expected_models in structure.items():
        filename = f"{provider_id}.json"
        path = directory / filename
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        if manifest_files is not None and manifest_files.get(filename) != sha256(content):
            errors.append(f"{filename} does not match its manifest hash")
        groups = _read_json_object(path, filename, errors)
        if groups is None:
            continue

        actual_models: Dict[str, str] = {}
        for api, group in groups.items():
            if not is_record(group):
                errors.append(f"{filename} API group {api!r} must be an object")
                continue
            for model_key, model in group.items():
                if model_key in actual_models:
                    errors.append(f"{provider_id}/{model_key} appears in more than one API group")
                    continue
                actual_models[model_key] = api
                separator = model_key.find(":")
                model_id = model_key[separator + 1 :] if separator >= 0 else model_key
                validate_model_value(model, provider_id, model_id, api, errors)
                if is_record(model) and model_key != f"{model.get('type')}:{model.get('id')}":
                    errors.append(f"{provider_id}/{model_key} has mismatched type/id identity")

        expected_model_ids = sorted(expected_models)
        actual_model_ids = sorted(actual_models)
        if not same_strings(expected_model_ids, actual_model_ids):
            errors.append(
                f"{filename} model IDs do not match the generated catalog "
                f"({describe_set_difference(expected_model_ids, actual_model_ids)})"
            )
        for model_id, expected_api in expected_models.items():
            actual_api = actual_models.get(model_id)
            if actual_api is not None and actual_api != expected_api:
                errors.append(
                    f"{provider_id}/{model_id} is grouped under API {actual_api!r}, expected {expected_api!r}"
                )

    if errors:
        throw_validation_errors(errors)


def _is_valid_timestamp(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def validate_generated_model_data(package_root: Path) -> None:
    """Validate the generated catalogs under `src/karen_ai/providers/data/`."""
    structure = read_model_data_structure(package_root)
    validate_model_data_directory(structure, data_dir(package_root))


__all__ = [
    "MODEL_DATA_MANIFEST_FILE",
    "MODEL_DATA_SCHEMA_VERSION",
    "ModelDataStructure",
    "ProviderGroups",
    "assert_exact_model_ids",
    "create_model_data_manifest",
    "data_dir",
    "describe_set_difference",
    "is_record",
    "model_data_structure_hash",
    "read_model_data_provider_ids",
    "read_model_data_structure",
    "read_provider_structure",
    "serialize_json",
    "sha256",
    "sorted_record",
    "validate_generated_model_data",
    "validate_model_data_directory",
    "validate_model_value",
]

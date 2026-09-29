#!/usr/bin/env python3
"""Sync vendored model catalogs from the published `@earendil-works/pi-ai` npm package.

karen-ai's `src/karen_ai/providers/data/*.json` are pi-ai's generated catalogs
(models.dev + OpenRouter + Vercel AI Gateway + Radius, with pi-ai's per-provider
fixups baked in). Rather than re-implementing pi-ai's 3.5k-line generator, this
script refreshes the vendored files from a published pi-ai release, then
re-applies the karen-ai additions that the npm build predates:

- `typesafe.json` (TypeSafe classifier catalog)
- the `cloudflare-workers-ai-system-one` classifier group in `cloudflare-workers-ai.json`
- the `typesafe-system-one` classifier group in `openrouter.json`
- the `openrouter-images` image group in `openrouter.json` (from the package's
  `image-models.generated.js`)

Usage:
    python scripts/sync_model_catalogs.py [--version 0.87.1] [--registry https://registry.npmmirror.com]

Without `--version`, the registry's `latest` tag is used. The script prints a
per-provider added/removed summary and rewrites `.manifest.json`.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
import urllib.request
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "src" / "karen_ai" / "providers" / "data"

DEFAULT_REGISTRY = "https://registry.npmjs.org"
PACKAGE_NAME = "@earendil-works/pi-ai"

TYPESAFE_CATALOG = {
    "typesafe-system-one": {
        "jev-latest": {
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
    }
}

CLOUDFLARE_WORKERS_AI_CLASSIFIER_GROUP = {
    "typesafe/jev": {
        "type": "classifier",
        "id": "typesafe/jev",
        "name": "Jev",
        "api": "cloudflare-workers-ai-system-one",
        "provider": "cloudflare-workers-ai",
        "baseUrl": "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai",
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 32000,
    }
}

OPENROUTER_CLASSIFIER_GROUP = {
    "typesafe/jev": {
        "type": "classifier",
        "id": "typesafe/jev",
        "name": "Jev (OpenRouter)",
        "api": "typesafe-system-one",
        "provider": "openrouter",
        "baseUrl": "https://openrouter.ai/api/v1",
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    }
}


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def resolve_version(registry: str, version: str | None) -> str:
    if version:
        return version
    metadata = fetch_json(f"{registry.rstrip('/')}/{PACKAGE_NAME}")
    return metadata["dist-tags"]["latest"]


def parse_image_models(js: str) -> dict[str, dict]:
    """Parse the generated `IMAGE_MODELS = {...}` literal without executing it."""
    start = js.index("IMAGE_MODELS = ")
    body = js[start + len("IMAGE_MODELS = ") :]
    body = re.sub(r"(?<=[{,])\s*([A-Za-z_]\w*)\s*:", lambda m: f' "{m.group(1)}":', body)
    body = re.sub(r",(\s*[}\]])", r"\1", body)
    data, _ = json.JSONDecoder().raw_decode(body)
    return data


def apply_karen_additions(catalogs: dict[str, dict], image_models_js: str | None) -> list[str]:
    """Re-apply karen-ai-only groups; returns human-readable notes."""
    notes: list[str] = []

    if "typesafe" not in catalogs:
        catalogs["typesafe"] = TYPESAFE_CATALOG
        notes.append("typesafe.json: added TypeSafe classifier catalog")

    cloudflare = catalogs.get("cloudflare-workers-ai")
    if cloudflare is not None and "cloudflare-workers-ai-system-one" not in cloudflare:
        cloudflare["cloudflare-workers-ai-system-one"] = CLOUDFLARE_WORKERS_AI_CLASSIFIER_GROUP
        notes.append("cloudflare-workers-ai.json: added System One classifier group")

    openrouter = catalogs.get("openrouter")
    if openrouter is not None:
        if "typesafe-system-one" not in openrouter:
            openrouter["typesafe-system-one"] = OPENROUTER_CLASSIFIER_GROUP
            notes.append("openrouter.json: added System One classifier group")
        if "openrouter-images" not in openrouter and image_models_js:
            image_models = parse_image_models(image_models_js).get("openrouter", {})
            if image_models:
                openrouter["openrouter-images"] = {
                    model_id: {"type": "image", **model} for model_id, model in image_models.items()
                }
                notes.append(f"openrouter.json: added {len(image_models)} image models")

    return notes


def count_models(catalog: dict) -> int:
    return sum(len(group) for group in catalog.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="pi-ai version to sync from (default: registry latest)")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY, help="npm registry base URL")
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = parser.parse_args()

    version = resolve_version(args.registry, args.version)
    print(f"Syncing model catalogs from {PACKAGE_NAME}@{version}")

    tarball_url = f"{args.registry.rstrip('/')}/{PACKAGE_NAME}/-/pi-ai-{version}.tgz"
    tarball = tarfile.open(fileobj=io.BytesIO(fetch_bytes(tarball_url)), mode="r:gz")

    catalogs: dict[str, dict] = {}
    manifest: bytes | None = None
    image_models_js: str | None = None
    for member in tarball.getmembers():
        name = member.name
        if name.startswith("package/dist/providers/data/") and name.endswith(".json"):
            key = Path(name).name
            content = tarball.extractfile(member)
            if content is None:
                continue
            if key == ".manifest.json":
                manifest = content.read()
            else:
                catalogs[Path(key).stem] = json.loads(content.read().decode("utf-8"))
        elif name == "package/dist/image-models.generated.js":
            content = tarball.extractfile(member)
            if content is not None:
                image_models_js = content.read().decode("utf-8")

    if not catalogs:
        raise SystemExit("No catalogs found in the package; the layout may have changed")

    for note in apply_karen_additions(catalogs, image_models_js):
        print(f"  + {note}")

    changes = 0
    for provider_id in sorted(catalogs):
        path = DATA_DIR / f"{provider_id}.json"
        incoming = catalogs[provider_id]
        if path.is_file():
            current = json.loads(path.read_text(encoding="utf-8"))
            current_count = count_models(current)
            incoming_count = count_models(incoming)
            if current == incoming:
                continue
            current_ids = {mid for group in current.values() for mid in group}
            incoming_ids = {mid for group in incoming.values() for mid in group}
            added = sorted(incoming_ids - current_ids)
            removed = sorted(current_ids - incoming_ids)
            print(
                f"  ~ {provider_id}: {current_count} -> {incoming_count}"
                f" (+{len(added)} -{len(removed)})"
            )
            if added:
                print(f"      added: {', '.join(added[:8])}{' …' if len(added) > 8 else ''}")
            if removed:
                print(f"      removed: {', '.join(removed[:8])}{' …' if len(removed) > 8 else ''}")
        else:
            print(f"  + {provider_id}: new catalog ({count_models(incoming)} models)")
        changes += 1
        if not args.dry_run:
            path.write_text(json.dumps(incoming, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")

    if manifest and not args.dry_run:
        (DATA_DIR / ".manifest.json").write_bytes(manifest)

    print(f"Done. {changes} catalog(s) changed." if not args.dry_run else f"Dry run. {changes} catalog(s) would change.")


if __name__ == "__main__":
    main()

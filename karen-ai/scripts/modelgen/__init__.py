"""Model-catalog generator, a faithful port of pi-ai's `scripts/*.ts`.

pi-ai builds its `providers/data/*.json` catalogs from a handful of upstream
sources (models.dev, OpenRouter, the Vercel AI Gateway, Nvidia NIM, Radius) and
then bakes in a long list of per-provider corrections. This package ports that
pipeline to Python so karen-ai can regenerate its vendored catalogs itself
instead of copying the published npm package.

Layout:

- `reasoning_options`  models.dev reasoning options -> Pi thinking-level maps
- `openrouter`         OpenRouter listing -> chat/image/classifier catalog
- `tables`             every constant table, base URL and static model list
- `compat`             per-api compat detection plus the metadata appliers
- `sources`            the upstream catalog fetchers
- `providers_dev`      models.dev mapping for every provider
- `model_data`         manifest, structure hash and output validation
"""

from __future__ import annotations

from pathlib import Path

#: `karen-ai/` — the directory holding `src/` and `scripts/`.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

__all__ = ["PACKAGE_ROOT"]

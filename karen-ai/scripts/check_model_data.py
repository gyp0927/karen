#!/usr/bin/env python3
"""Validate the vendored model catalogs.

Port of pi-ai's `scripts/check-model-data.ts`: the catalogs under
`src/karen_ai/providers/data/` must carry a manifest whose schema version,
generation stamp and per-file hashes match the data on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modelgen import PACKAGE_ROOT
from modelgen.model_data import validate_generated_model_data


def main() -> int:
    try:
        validate_generated_model_data(PACKAGE_ROOT)
    except Exception as error:  # noqa: BLE001 - the CLI reports any failure the same way
        print(error, file=sys.stderr)
        print(
            "\nModel data is missing or stale. Run `python scripts/generate_models.py`"
            " from karen-ai/.",
            file=sys.stderr,
        )
        return 1
    print("Generated model data is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

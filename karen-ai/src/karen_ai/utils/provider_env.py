"""Provider-scoped environment lookup, mirroring pi-ai's utils/provider-env.ts.

Scoped `ProviderEnv` overrides win over the process environment.
"""

from __future__ import annotations

import os
from typing import Dict, Optional


def get_provider_env_value(name: str, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Resolves `name` from scoped overrides first, then the process environment."""
    if env:
        scoped = env.get(name)
        if scoped:
            return scoped
    return os.environ.get(name) or None


__all__ = ["get_provider_env_value"]

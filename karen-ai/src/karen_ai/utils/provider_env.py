"""Provider-scoped environment lookup, mirroring utils/provider-env.ts."""

from __future__ import annotations

import os
from typing import Optional

from ..types import ProviderEnv


def get_provider_env_value(name: str, env: Optional[ProviderEnv] = None) -> Optional[str]:
    """Provider-scoped env values take precedence over process env."""
    if env is not None:
        value = env.get(name)
        if value:
            return value
    return os.environ.get(name)

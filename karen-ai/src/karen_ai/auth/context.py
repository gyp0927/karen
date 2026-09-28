"""Environment access for auth resolution, mirroring pi-ai's auth/context.ts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class AuthContext:
    """Default environment access: process env vars and filesystem existence checks."""

    async def env(self, name: str) -> Optional[str]:
        return os.environ.get(name)

    async def file_exists(self, path: str) -> bool:
        return Path(path).expanduser().exists()


def default_provider_auth_context() -> AuthContext:
    return AuthContext()

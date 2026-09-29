"""User-Agent header value, mirroring pi-ai's utils/pi-user-agent.ts.

The product token stays `pi`: the Codex backend keys client behavior on it.
"""

from __future__ import annotations

import platform


def get_pi_user_agent() -> str:
    return f"pi ({platform.system().lower()} {platform.release()}; {platform.machine()})"

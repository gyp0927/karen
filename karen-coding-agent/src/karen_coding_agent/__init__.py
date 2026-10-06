"""karen-coding-agent: the karen coding-agent app.

The application layer on top of karen-agent — the karen equivalent of
`@earendil-works/pi`'s `packages/coding-agent`. M1 scope: `AgentSession`
(app-level session: Agent + persistence + auto-compaction + hooks) and the
`karen` CLI (interactive REPL + headless print mode).
"""

from .agent_session import (
    DEFAULT_BRANCH,
    DEFAULT_SESSIONS_ROOT,
    DEFAULT_SYSTEM_PROMPT,
    AgentSession,
    SessionListener,
)

__all__ = [
    "AgentSession",
    "SessionListener",
    "DEFAULT_BRANCH",
    "DEFAULT_SESSIONS_ROOT",
    "DEFAULT_SYSTEM_PROMPT",
]

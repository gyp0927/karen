"""Search service interfaces (pi's `search/index.ts`).

Pure protocol definitions: a session search service indexes session content and
answers session-level and (optionally) entry-level queries. Implementations
live outside karen-agent.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from karen_ai.types import KarenBase

__all__ = [
    "EntrySearchHit",
    "SearchQuery",
    "SessionSearchHit",
    "SessionSearchService",
]


class SearchQuery(KarenBase):
    text: str
    limit: Optional[int] = None


class SessionSearchTopHit(KarenBase):
    entry_id: str
    snippet: Optional[str] = None
    timestamp: float


class SessionSearchHit(KarenBase):
    session_id: str
    score: Optional[float] = None
    top: Optional[SessionSearchTopHit] = None


class EntrySearchHit(KarenBase):
    session_id: str
    entry_id: str
    timestamp: float
    snippet: Optional[str] = None
    score: Optional[float] = None


@runtime_checkable
class SessionSearchService(Protocol):
    """Indexes sessions and answers search queries over them."""

    async def search_sessions(self, query: SearchQuery) -> List[SessionSearchHit]: ...

    async def search_entries(self, query: SearchQuery) -> List[EntrySearchHit]:
        """Optional in pi (`searchEntries?`); implementations may raise NotImplementedError."""
        ...

    async def sync(self) -> None: ...

    def notify(self, session_id: str) -> None: ...

    async def remove(self, session_id: str) -> None: ...

    async def close(self) -> None: ...

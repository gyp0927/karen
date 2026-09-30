"""Search service interfaces (pi's search/index.ts) — shapes + protocol conformance."""

from __future__ import annotations

from typing import List

from karen_agent.search import (
    EntrySearchHit,
    SearchQuery,
    SessionSearchHit,
    SessionSearchService,
)


class _MemorySearchService:
    def __init__(self) -> None:
        self.notified: List[str] = []
        self.removed: List[str] = []
        self.closed = False

    async def search_sessions(self, query: SearchQuery) -> List[SessionSearchHit]:
        return [
            SessionSearchHit(
                session_id="s1",
                score=0.9,
                top={"entry_id": "e1", "snippet": query.text, "timestamp": 123.0},
            )
        ]

    async def search_entries(self, query: SearchQuery) -> List[EntrySearchHit]:
        return [EntrySearchHit(session_id="s1", entry_id="e1", timestamp=123.0, score=1.0)]

    async def sync(self) -> None:
        pass

    def notify(self, session_id: str) -> None:
        self.notified.append(session_id)

    async def remove(self, session_id: str) -> None:
        self.removed.append(session_id)

    async def close(self) -> None:
        self.closed = True


def test_query_and_hit_shapes():
    query = SearchQuery(text="hello", limit=5)
    assert query.text == "hello" and query.limit == 5
    assert SearchQuery(text="x").limit is None

    hit = SessionSearchHit(session_id="s", score=1.5, top={"entry_id": "e", "timestamp": 7})
    assert hit.top is not None and hit.top.entry_id == "e" and hit.top.snippet is None

    entry = EntrySearchHit(session_id="s", entry_id="e", timestamp=7, snippet="snip")
    assert entry.score is None


def test_wire_format_is_camel_case():
    hit = SessionSearchHit(session_id="s", top={"entry_id": "e", "timestamp": 1})
    dumped = hit.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert dumped == {"sessionId": "s", "top": {"entryId": "e", "timestamp": 1}}

    entry = EntrySearchHit(session_id="s", entry_id="e", timestamp=2)
    assert set(entry.model_dump(mode="json", by_alias=True, exclude_none=True)) == {
        "sessionId",
        "entryId",
        "timestamp",
    }


async def test_service_protocol_roundtrip():
    service = _MemorySearchService()
    assert isinstance(service, SessionSearchService)

    hits = await service.search_sessions(SearchQuery(text="q"))
    assert hits[0].top is not None and hits[0].top.snippet == "q"

    entries = await service.search_entries(SearchQuery(text="q", limit=1))
    assert entries[0].entry_id == "e1"

    service.notify("s1")
    await service.remove("s2")
    await service.sync()
    await service.close()
    assert service.notified == ["s1"]
    assert service.removed == ["s2"]
    assert service.closed is True

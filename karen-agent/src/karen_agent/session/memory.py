"""In-memory storage and session repo (pi's `harness/session/memory.ts`).

`MemoryStorage` applies commits to an `InMemoryStorageState` behind a serialized
commit queue. `MemorySessionRepo` hands out `MemorySessionFacade` handles that
track admitted operations so `close()` waits for in-flight work while the
underlying storage (and its entries) survives facade close/reopen.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Callable, Dict, List, Optional

from .ids import uuid7
from .mutation_line import MutationLine
from .session import StorageBackedSession
from .storage_state import InMemoryStorageState
from .types import (
    Branch,
    BranchScan,
    CommitResult,
    Entry,
    EntryScan,
    EntryStructure,
    ForkOptions,
    Session,
    SessionCreateOptions,
    SessionMetadata,
    SessionMutation,
    SessionMutationCallback,
    SessionStats,
    Storage,
    StorageBranchScan,
    UsageRow,
    UsageScan,
    Write,
)
from .values import StoredValue, Value, ValueList


def _now_ms() -> int:
    return int(time.time() * 1000)


class MemoryStorage(Storage):
    def __init__(self, now: Optional[Callable[[], int]] = None) -> None:
        self._now = now or _now_ms
        self._storage_state = InMemoryStorageState()
        self._commit_line = MutationLine()
        self._state = "open"  # "open" | "closing" | "closed"
        self._close_task: Optional[asyncio.Future] = None

    async def commit(self, writes: List[Write]) -> CommitResult:
        if self._state != "open":
            raise RuntimeError("MemoryStorage is closed")

        def apply() -> CommitResult:
            prepared = self._storage_state.prepare_commit(writes, self._now())
            stats = self._storage_state.apply_validated(prepared.writes)
            return CommitResult(
                first_seq=prepared.first_seq,
                seqs=prepared.seqs,
                timestamp=prepared.timestamp,
                stats=stats,
            )

        return await self._commit_line.run(apply)

    async def get_entries(self, ids: List[str]) -> Dict[str, Entry]:
        self._assert_open()
        return self._storage_state.get_entries(ids)

    async def get_value(self, address: Value) -> Optional[StoredValue]:
        self._assert_open()
        return self._storage_state.get_value(address)

    async def scan_values(self, prefix: Value) -> List[StoredValue]:
        self._assert_open()
        return self._storage_state.scan_values(prefix)

    async def read_list(self, address: ValueList, options=None) -> List[Any]:
        self._assert_open()
        return self._storage_state.read_list(address, options)

    async def scan_branch(self, query: StorageBranchScan) -> List[Entry]:
        self._assert_open()
        return self._storage_state.scan_branch(query)

    async def scan_branch_structure(self, query: StorageBranchScan) -> List[EntryStructure]:
        self._assert_open()
        return self._storage_state.scan_branch_structure(query)

    async def scan_entries(self, query: EntryScan) -> List[Entry]:
        self._assert_open()
        return self._storage_state.scan_entries(query)

    async def scan_usage(self, query: UsageScan) -> List[UsageRow]:
        self._assert_open()
        return self._storage_state.scan_usage(query)

    async def get_stats(self) -> SessionStats:
        self._assert_open()
        return self._storage_state.get_stats()

    def fork(self, options: ForkOptions) -> "asyncio.Future[MemoryStorage]":
        """Construct a destination storage at one serialized boundary between source commits."""
        loop = asyncio.get_running_loop()
        if self._state != "open":
            rejected: asyncio.Future = loop.create_future()
            rejected.set_exception(RuntimeError("MemoryStorage is closed"))
            return rejected

        def copy_state() -> "MemoryStorage":
            destination = MemoryStorage(now=self._now)
            destination._storage_state = self._storage_state.create_fork(options)
            return destination

        return self._commit_line.run(copy_state)

    def close(self) -> "asyncio.Future[None]":
        if self._close_task is not None:
            return self._close_task
        self._state = "closing"

        async def finish() -> None:
            await self._commit_line.seal(RuntimeError("MemoryStorage is closed"))
            self._state = "closed"

        self._close_task = asyncio.ensure_future(finish())
        return self._close_task

    def _assert_open(self) -> None:
        if self._state != "open":
            raise RuntimeError("MemoryStorage is closed")


MEMORY_STORAGE_VERSION = 1


class _MemorySessionRecord:
    def __init__(self, metadata: SessionMetadata, storage: MemoryStorage, session: StorageBackedSession) -> None:
        self.metadata = metadata
        self.storage = storage
        self.session = session
        self.open = True


class _FacadeMutation(SessionMutation):
    def __init__(self, facade: "MemorySessionFacade", source: SessionMutation, finished: asyncio.Future) -> None:
        self._facade = facade
        self._source = source
        self._finished = finished
        self._ended = False

    async def commit(self, writes: List[Write]) -> CommitResult:
        return await self._source.commit(writes)

    async def end(self) -> None:
        try:
            await self._source.end()
        finally:
            if not self._ended:
                self._ended = True
                self._facade._admitted.discard(self._finished)
                if not self._finished.done():
                    self._finished.set_result(None)

    async def get_entries(self, ids: List[str]) -> Dict[str, Entry]:
        return await self._source.get_entries(ids)

    async def get_stats(self) -> SessionStats:
        return await self._source.get_stats()

    async def get_value(self, address: Value) -> Optional[StoredValue]:
        return await self._source.get_value(address)

    async def scan_values(self, prefix: Value) -> List[StoredValue]:
        return await self._source.scan_values(prefix)

    async def read_list(self, address: ValueList, options=None) -> List[Any]:
        return await self._source.read_list(address, options)

    async def scan_branch(self, query: StorageBranchScan) -> List[Entry]:
        return await self._source.scan_branch(query)


class _FacadeBranch(Branch):
    def __init__(self, facade: "MemorySessionFacade", branch: Branch) -> None:
        self._facade = facade
        self._branch = branch
        self.name = branch.name

    async def get_tip_id(self) -> Optional[str]:
        return await self._facade._admit(lambda: self._branch.get_tip_id())

    async def find_entries(self, query: Optional[BranchScan] = None) -> List[Entry]:
        return await self._facade._admit(lambda: self._branch.find_entries(query))

    async def find_entry(self, query: Optional[BranchScan] = None) -> Optional[Entry]:
        return await self._facade._admit(lambda: self._branch.find_entry(query))

    async def append_message(self, message: Any) -> str:
        return await self._facade._admit(lambda: self._branch.append_message(message))

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        return await self._facade._admit(lambda: self._branch.append_custom_entry(custom_type, data))


class MemorySessionFacade(Session):
    def __init__(self, session: StorageBackedSession, on_close: Callable[[], None]) -> None:
        self._session = session
        self.metadata = session.metadata
        self.id_generator = session.id_generator
        self._on_close = on_close
        self._admitted: set = set()
        self._closed_error = RuntimeError("Session is closed")
        self._state = "open"
        self._close_task: Optional[asyncio.Future] = None

    async def begin_mutation(self) -> SessionMutation:
        loop = asyncio.get_running_loop()
        finished: asyncio.Future = loop.create_future()
        self._admitted.add(finished)
        try:
            source = await self._admit(lambda: self._session.begin_mutation())
        except Exception:
            self._admitted.discard(finished)
            if not finished.done():
                finished.set_result(None)
            raise
        if self._state != "open":
            await source.end()
            self._admitted.discard(finished)
            if not finished.done():
                finished.set_result(None)
            raise self._closed_error
        return _FacadeMutation(self, source, finished)

    async def mutate(self, mutation: SessionMutationCallback) -> Any:
        # The closed check must run inside the underlying session's mutation line.
        async def checked(mutator: SessionMutation) -> Any:
            if self._state != "open":
                raise self._closed_error
            result = mutation(mutator)
            if inspect.isawaitable(result):
                result = await result
            return result

        return await self._admit(lambda: self._session.mutate(checked))

    # --- delegated reads/writes (each admitted) --------------------------------

    async def get_entries(self, ids: List[str]) -> Dict[str, Entry]:
        return await self._admit(lambda: self._session.get_entries(ids))

    async def get_entry(self, id: str) -> Optional[Entry]:
        return await self._admit(lambda: self._session.get_entry(id))

    async def get_value(self, address: Value) -> Optional[StoredValue]:
        return await self._admit(lambda: self._session.get_value(address))

    async def scan_values(self, prefix: Value) -> List[StoredValue]:
        return await self._admit(lambda: self._session.scan_values(prefix))

    async def read_list(self, address: ValueList, options=None) -> List[Any]:
        return await self._admit(lambda: self._session.read_list(address, options))

    async def scan_branch(self, query: StorageBranchScan) -> List[Entry]:
        return await self._admit(lambda: self._session.scan_branch(query))

    async def get_stats(self) -> SessionStats:
        return await self._admit(lambda: self._session.get_stats())

    async def get_name(self) -> Optional[str]:
        return await self._admit(lambda: self._session.get_name())

    async def get_label(self, target_id: str) -> Optional[str]:
        return await self._admit(lambda: self._session.get_label(target_id))

    async def find_entries(self, query=None) -> List[Entry]:
        return await self._admit(lambda: self._session.find_entries(query))

    async def find_entry(self, query=None) -> Optional[Entry]:
        return await self._admit(lambda: self._session.find_entry(query))

    async def branch(self, name: str) -> Optional[Branch]:
        branch = await self._admit(lambda: self._session.branch(name))
        return None if branch is None else _FacadeBranch(self, branch)

    async def create_branch(self, name: str, at: Optional[str]) -> Branch:
        branch = await self._admit(lambda: self._session.create_branch(name, at))
        return _FacadeBranch(self, branch)

    async def set_value(self, address: Value, next: Any) -> None:
        return await self._admit(lambda: self._session.set_value(address, next))

    async def delete_value(self, address: Value) -> None:
        return await self._admit(lambda: self._session.delete_value(address))

    async def append_list(self, address: ValueList, element: Any) -> None:
        return await self._admit(lambda: self._session.append_list(address, element))

    async def delete_list(self, address: ValueList) -> None:
        return await self._admit(lambda: self._session.delete_list(address))

    async def set_name(self, name: Optional[str]) -> None:
        return await self._admit(lambda: self._session.set_name(name))

    async def set_label(self, target_id: str, label: Optional[str]) -> None:
        return await self._admit(lambda: self._session.set_label(target_id, label))

    def close(self) -> "asyncio.Future[None]":
        if self._close_task is not None:
            return self._close_task
        self._state = "closing"

        async def finish() -> None:
            await asyncio.gather(*self._admitted, return_exceptions=True)
            self._state = "closed"
            self._on_close()

        self._close_task = asyncio.ensure_future(finish())
        return self._close_task

    async def _admit(self, operation: Callable[[], Any]) -> Any:
        if self._state != "open":
            raise self._closed_error
        loop = asyncio.get_running_loop()
        try:
            result = operation()
        except Exception as error:
            rejected: asyncio.Future = loop.create_future()
            rejected.set_exception(error)
            result = rejected
        task = asyncio.ensure_future(result)
        self._admitted.add(task)
        task.add_done_callback(lambda t: self._admitted.discard(t))
        return await task


class MemorySessionRepo:
    """In-memory `SessionRepo`; metadata and state vanish with the process."""

    def __init__(self, now: Optional[Callable[[], int]] = None) -> None:
        self._now = now or _now_ms
        self._sessions: Dict[str, _MemorySessionRecord] = {}
        self._pending_ids: set = set()
        self._closed = False
        self._close_task: Optional[asyncio.Future] = None

    async def create(self, options: Optional[SessionCreateOptions] = None) -> Session:
        self._assert_open()
        options = options or SessionCreateOptions()
        created_at = self._now()
        id = options.id or uuid7(created_at)
        self._reserve_id(id)
        metadata = SessionMetadata(
            id=id,
            created_at=created_at,
            storage_version=MEMORY_STORAGE_VERSION,
            parent_session_id=options.parent_session_id,
        )
        storage = MemoryStorage(now=self._now)
        session = StorageBackedSession(metadata, storage)
        try:
            record = _MemorySessionRecord(metadata, storage, session)
            self._sessions[id] = record
            return self._open_record(record)
        except Exception:
            await session.close()
            raise
        finally:
            self._pending_ids.discard(id)

    async def open(self, metadata: SessionMetadata) -> Session:
        # Memory sessions are always created at the current storage version, so
        # persistent-backend version gating does not apply here.
        self._assert_open()
        record = self._sessions.get(metadata.id)
        if record is None:
            raise ValueError(f"Unknown session: {metadata.id}")
        if record.open:
            raise ValueError(f"Session is already open: {metadata.id}")
        record.open = True
        return self._open_record(record)

    async def list(self, options: Any = None) -> List[SessionMetadata]:
        self._assert_open()
        return [record.metadata for record in self._sessions.values()]

    async def delete(self, metadata: SessionMetadata) -> None:
        self._assert_open()
        record = self._sessions.get(metadata.id)
        if record is None:
            raise ValueError(f"Unknown session: {metadata.id}")
        if record.open:
            raise ValueError(f"Session is open: {metadata.id}")
        await record.session.close()
        del self._sessions[metadata.id]

    async def fork(self, source: SessionMetadata, options: ForkOptions) -> Session:
        self._assert_open()
        source_record = self._sessions.get(source.id)
        if source_record is None:
            raise ValueError(f"Unknown session: {source.id}")
        created_at = self._now()
        id = options.id or uuid7(created_at)
        self._reserve_id(id)
        try:
            storage = await source_record.storage.fork(options)
            metadata = SessionMetadata(
                id=id,
                created_at=created_at,
                storage_version=MEMORY_STORAGE_VERSION,
                parent_session_id=source_record.metadata.id,
            )
            session = StorageBackedSession(metadata, storage)
            record = _MemorySessionRecord(metadata, storage, session)
            self._sessions[id] = record
            return self._open_record(record)
        finally:
            self._pending_ids.discard(id)

    def close(self) -> "asyncio.Future[None]":
        if self._close_task is not None:
            return self._close_task
        self._closed = True

        async def finish() -> None:
            await asyncio.gather(
                *(record.session.close() for record in self._sessions.values()),
                return_exceptions=True,
            )

        self._close_task = asyncio.ensure_future(finish())
        return self._close_task

    def _open_record(self, record: _MemorySessionRecord) -> Session:
        def on_close() -> None:
            record.open = False

        return MemorySessionFacade(record.session, on_close)

    def _reserve_id(self, id: str) -> None:
        if id in self._sessions or id in self._pending_ids:
            raise ValueError(f"Session already exists: {id}")
        self._pending_ids.add(id)

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("MemorySessionRepo is closed")

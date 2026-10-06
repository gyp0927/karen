"""Storage-backed Session and Branch (pi's `harness/session/session.ts`).

`StorageBackedSession` is the package-internal typed boundary shared by concrete
session repositories (memory, JSONL). All writes go through a single
`MutationLine`; readers hit storage directly and are deliberately non-atomic
with respect to concurrent commits.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Callable, Dict, List, Optional

from ..types import AgentMessage
from .commit import insert_entry
from .ids import Uuid7IdGenerator
from .mutation_line import MutationLine
from .types import (
    Branch,
    BranchScan,
    CommitResult,
    CustomEntry,
    Entry,
    EntryQuery,
    EntryScan,
    IdGenerator,
    ListAppendWrite,
    ListDeleteWrite,
    MessageEntry,
    Session,
    SessionMetadata,
    SessionMutation,
    SessionMutationCallback,
    SessionStats,
    Storage,
    StorageBranchScan,
    ValueDeleteWrite,
    ValueSetWrite,
    Write,
)
from .values import (
    MAX_SAFE_INTEGER,
    StoredValue,
    Value,
    ValueList,
    append_list,
    branch_tip,
    delete_list,
    delete_value,
    entry_label,
    session_name,
    set_value,
)


class SessionInvariantError(Exception):
    """Durable session state is internally inconsistent and cannot be safely advanced."""


class SessionInvalidBranchError(Exception):
    """A requested Branch name is invalid."""

    def __init__(self, branch: str, reason: str) -> None:
        super().__init__(f"Invalid branch {json.dumps(branch)}: {reason}")
        self.branch = branch
        self.reason = reason


class SessionBranchExistsError(Exception):
    """A requested branch already exists."""

    def __init__(self, branch: str) -> None:
        super().__init__(f"Branch already exists: {branch}")
        self.branch = branch


class SessionPendingAssistantMessageError(Exception):
    """A pending assistant message cannot be persisted as a session entry."""

    def __init__(self) -> None:
        super().__init__("Cannot persist a pending assistant message")


class SessionUnknownTargetError(Exception):
    """A requested session entry target does not exist."""

    def __init__(self, target_id: str) -> None:
        super().__init__(f"Unknown target: {target_id}")
        self.target_id = target_id


def is_pending_assistant_message(message: AgentMessage) -> bool:
    """Whether a message is an assistant message whose stop_reason is still pending."""
    if isinstance(message, dict):
        return message.get("role") == "assistant" and message.get("stopReason") == "pending"
    return getattr(message, "role", None) == "assistant" and getattr(message, "stop_reason", None) == "pending"


class StorageBackedSessionMutation(SessionMutation):
    def __init__(self, storage: Storage, release: Callable[[], None]) -> None:
        self._storage = storage
        self._release = release
        self._active = True
        self._commit_future: Optional[asyncio.Future] = None
        self._end_future: Optional[asyncio.Future] = None

    async def commit(self, writes: List[Write]) -> CommitResult:
        self._assert_active()
        if self._commit_future is not None:
            raise RuntimeError("SessionMutator commit already attempted")
        loop = asyncio.get_running_loop()
        try:
            for write in writes:
                if (
                    getattr(write, "kind", None) == "entry"
                    and write.entry.type == "message"
                    and is_pending_assistant_message(write.entry.message)
                ):
                    raise SessionPendingAssistantMessageError()
            self._commit_future = loop.create_task(self._storage.commit(writes))
        except Exception as error:
            failed: asyncio.Future = loop.create_future()
            failed.set_exception(error)
            self._commit_future = failed
        return await self._commit_future

    async def end(self) -> None:
        if self._end_future is not None:
            await self._end_future
            return
        self._active = False

        async def finish() -> None:
            if self._commit_future is not None:
                try:
                    await self._commit_future
                except Exception:
                    pass
            self._release()

        self._end_future = asyncio.ensure_future(finish())
        await self._end_future

    async def get_entries(self, ids: List[str]) -> Dict[str, Entry]:
        self._assert_active()
        return await self._storage.get_entries(ids)

    async def get_stats(self) -> SessionStats:
        self._assert_active()
        return await self._storage.get_stats()

    async def get_value(self, address: Value) -> Optional[StoredValue]:
        self._assert_active()
        return await self._storage.get_value(address)

    async def scan_values(self, prefix: Value) -> List[StoredValue]:
        self._assert_active()
        return await self._storage.scan_values(prefix)

    async def read_list(self, address: ValueList, options=None) -> List[Any]:
        self._assert_active()
        return await self._storage.read_list(address, options)

    async def scan_branch(self, query: StorageBranchScan) -> List[Entry]:
        self._assert_active()
        return await self._storage.scan_branch(query)

    def _assert_active(self) -> None:
        if not self._active:
            raise RuntimeError("SessionMutator cannot be used outside its mutation callback")


class StorageBackedBranch(Branch):
    def __init__(self, name: str, session: "StorageBackedSession") -> None:
        self.name = name
        self._session = session

    async def get_tip_id(self) -> Optional[str]:
        return await self._session.get_branch_tip(self.name)

    async def find_entries(self, query: Optional[BranchScan] = None) -> List[Entry]:
        query = query or BranchScan()
        start = query.start if query.start is not None else await self.get_tip_id()
        if start is None:
            return []
        return await self._session.scan_branch(
            StorageBranchScan(
                start=start,
                stop_at_type=query.stop_at_type,
                stop_at_id=query.stop_at_id,
                type=query.type,
                custom_type=query.custom_type,
                order=query.order or "newestFirst",
                limit=query.limit,
                cursor=query.cursor,
            )
        )

    async def find_entry(self, query: Optional[BranchScan] = None) -> Optional[Entry]:
        query = query or BranchScan()
        limit = 1 if query.limit is None else min(query.limit, 1)
        entries = await self.find_entries(query.model_copy(update={"limit": limit}))
        return entries[0] if entries else None

    async def append_message(self, message: AgentMessage) -> str:
        return await self._session.append_message_to_branch(self.name, message)

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        return await self._session.append_custom_entry_to_branch(self.name, custom_type, data)


class StorageBackedSession(Session):
    """Package-internal typed boundary shared by concrete session repositories."""

    def __init__(
        self,
        metadata: SessionMetadata,
        storage: Storage,
        *,
        mutation_line: Optional[MutationLine] = None,
        id_generator: Optional[IdGenerator] = None,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        self.metadata = metadata
        self.id_generator: IdGenerator = id_generator or Uuid7IdGenerator()
        self._storage = storage
        self._mutation_line = mutation_line or MutationLine()
        self._on_close = on_close
        self._branches: Dict[str, StorageBackedBranch] = {}
        self._closed_error = RuntimeError("Session is closed")
        self._state = "open"  # "open" | "closing" | "closed"
        self._close_task: Optional[asyncio.Future] = None

    @property
    def storage(self) -> Storage:
        """The underlying storage (a `JsonlStorage` for file-backed sessions,
        which exposes the session file's `header`)."""
        return self._storage

    # --- mutations ------------------------------------------------------------

    async def begin_mutation(self) -> SessionMutation:
        self._assert_open()
        loop = asyncio.get_running_loop()
        granted: asyncio.Future = loop.create_future()
        finished: asyncio.Future = loop.create_future()

        def release() -> None:
            if not finished.done():
                finished.set_result(None)

        async def hold() -> None:
            granted.set_result(StorageBackedSessionMutation(self._storage, release))
            await finished

        line = self._mutation_line.run(hold)

        def propagate_line_failure(task: asyncio.Future) -> None:
            if granted.done():
                return
            if task.cancelled():
                granted.cancel()
                return
            error = task.exception()
            if error is not None:
                granted.set_exception(error)

        line.add_done_callback(propagate_line_failure)
        return await granted

    async def mutate(self, mutation: SessionMutationCallback) -> Any:
        mutator = await self.begin_mutation()
        try:
            result = mutation(mutator)
            if inspect.isawaitable(result):
                result = await result
            return result
        finally:
            await mutator.end()

    # --- reads ------------------------------------------------------------------

    async def get_entries(self, ids: List[str]) -> Dict[str, Entry]:
        self._assert_open()
        return await self._storage.get_entries(ids)

    async def get_entry(self, id: str) -> Optional[Entry]:
        return (await self.get_entries([id])).get(id)

    async def get_value(self, address: Value) -> Optional[StoredValue]:
        self._assert_open()
        return await self._storage.get_value(address)

    async def scan_values(self, prefix: Value) -> List[StoredValue]:
        self._assert_open()
        return await self._storage.scan_values(prefix)

    async def read_list(self, address: ValueList, options=None) -> List[Any]:
        self._assert_open()
        return await self._storage.read_list(address, options)

    async def scan_branch(self, query: StorageBranchScan) -> List[Entry]:
        self._assert_open()
        return await self._storage.scan_branch(query)

    async def get_stats(self) -> SessionStats:
        self._assert_open()
        return await self._storage.get_stats()

    async def get_name(self) -> Optional[str]:
        stored = await self.get_value(session_name)
        return stored.value if stored is not None else None

    async def get_label(self, target_id: str) -> Optional[str]:
        stored = await self.get_value(entry_label(target_id))
        return stored.value if stored is not None else None

    async def find_entries(self, query: Optional[EntryQuery] = None) -> List[Entry]:
        query = query or EntryQuery()
        self._assert_open()
        order = query.order or "desc"
        from_seq: Optional[int] = None
        to_seq: Optional[int] = None
        if query.cursor is not None:
            if order == "asc":
                if query.cursor.seq == MAX_SAFE_INTEGER:
                    return []
                from_seq = query.cursor.seq + 1
            else:
                if query.cursor.seq <= 1:
                    return []
                to_seq = query.cursor.seq - 1
        return await self._storage.scan_entries(
            EntryScan(
                type=query.type,
                custom_type=query.custom_type,
                order=order,
                limit=query.limit,
                from_seq=from_seq,
                to_seq=to_seq,
            )
        )

    async def find_entry(self, query: Optional[EntryQuery] = None) -> Optional[Entry]:
        query = query or EntryQuery()
        limit = 1 if query.limit is None else min(query.limit, 1)
        entries = await self.find_entries(query.model_copy(update={"limit": limit}))
        return entries[0] if entries else None

    # --- branches -----------------------------------------------------------------

    async def branch(self, name: str) -> Optional[Branch]:
        self._assert_valid_branch_name(name)
        if (await self.get_value(branch_tip(name))) is None:
            return None
        return self._get_or_create_branch_object(name)

    async def create_branch(self, name: str, at: Optional[str]) -> Branch:
        self._assert_open()
        self._assert_valid_branch_name(name)

        async def create(mutator: SessionMutation) -> None:
            if (await mutator.get_value(branch_tip(name))) is not None:
                raise SessionBranchExistsError(name)
            if at is not None and at not in await mutator.get_entries([at]):
                raise SessionUnknownTargetError(at)
            await mutator.commit([set_value(branch_tip(name), at)])

        await self.mutate(create)
        return self._get_or_create_branch_object(name)

    async def get_branch_tip(self, name: str) -> Optional[str]:
        stored = await self.get_value(branch_tip(name))
        if stored is None:
            raise SessionInvariantError(f"Unknown branch: {name}")
        return stored.value

    async def append_message_to_branch(self, name: str, message: AgentMessage) -> str:
        if is_pending_assistant_message(message):
            raise SessionPendingAssistantMessageError()
        return await self._append_to_branch(
            name, lambda id, parent_id: MessageEntry(id=id, parent_id=parent_id, message=message)
        )

    async def append_custom_entry_to_branch(self, name: str, custom_type: str, data: Any = None) -> str:
        return await self._append_to_branch(
            name, lambda id, parent_id: CustomEntry(id=id, parent_id=parent_id, custom_type=custom_type, data=data)
        )

    async def _append_to_branch(self, name: str, build_entry: Callable[[str, Optional[str]], Entry]) -> str:
        self._assert_open()
        id = self.id_generator.next()

        async def append(mutator: SessionMutation) -> None:
            tip = await mutator.get_value(branch_tip(name))
            if tip is None:
                raise SessionInvariantError(f"Unknown branch: {name}")
            await mutator.commit(
                [insert_entry(build_entry(id, tip.value)), set_value(branch_tip(name), id)]
            )

        await self.mutate(append)
        return id

    # --- typed values / lists ---------------------------------------------------

    async def set_value(self, address: Value, next: Any) -> None:
        await self.mutate(lambda mutator: mutator.commit([set_value(address, next)]))

    async def delete_value(self, address: Value) -> None:
        await self.mutate(lambda mutator: mutator.commit([delete_value(address)]))

    async def append_list(self, address: ValueList, element: Any) -> None:
        await self.mutate(lambda mutator: mutator.commit([append_list(address, element)]))

    async def delete_list(self, address: ValueList) -> None:
        await self.mutate(lambda mutator: mutator.commit([delete_list(address)]))

    async def set_name(self, name: Optional[str]) -> None:
        if name is None:
            await self.delete_value(session_name)
        else:
            await self.set_value(session_name, name)

    async def set_label(self, target_id: str, label: Optional[str]) -> None:
        address = entry_label(target_id)
        if label is None:
            await self.delete_value(address)
        else:
            await self.set_value(address, label)

    # --- lifecycle ----------------------------------------------------------------

    def close(self) -> "asyncio.Future[None]":
        if self._close_task is not None:
            return self._close_task
        self._state = "closing"
        self._close_task = asyncio.ensure_future(self._close())
        return self._close_task

    async def _close(self) -> None:
        try:
            await self._mutation_line.seal(self._closed_error)
            await self._storage.close()
        finally:
            self._state = "closed"
            if self._on_close is not None:
                self._on_close()

    # --- internals ------------------------------------------------------------------

    def _get_or_create_branch_object(self, name: str) -> StorageBackedBranch:
        branch = self._branches.get(name)
        if branch is None:
            branch = StorageBackedBranch(name, self)
            self._branches[name] = branch
        return branch

    def _assert_valid_branch_name(self, name: str) -> None:
        if len(name) == 0:
            raise SessionInvalidBranchError(name, "branch name must not be empty")
        if "\x00" in name:
            raise SessionInvalidBranchError(name, "branch name must not contain \\u0000")

    def _assert_open(self) -> None:
        if self._state != "open":
            raise self._closed_error

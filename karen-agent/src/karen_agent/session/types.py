"""Session persistence types, mirroring pi's `harness/session/types.ts`.

Scope: the entry/value/list storage model, scans, stats, fork options, and the
Storage / Session / Branch / SessionRepo abstract interfaces. pi's durable
operation state machine (`OperationState`, `ToolBatch`, `GenerationContext`, ...)
belongs to the harness runtime and is intentionally not ported yet; the reserved
`pi.op.*` value addresses in `values.py` type those payloads as `Any`.

pi threads a chord `Context` through every method; karen drops that parameter
(it carried cancellation/telemetry the session layer never consumed directly).

Unlike pi's `NewEntry = Omit<Entry, "seq"|"timestamp">` type-level split, the
Python models carry `seq`/`timestamp` fields with placeholder defaults; storage
assigns both at commit time (see `commit.py`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Protocol, Union

from karen_ai import Usage
from karen_ai.types import KarenBase
from pydantic import ConfigDict, Field
from typing_extensions import Annotated

from ..types import AgentMessage

#: pi's `JsonValue` (from chord): anything JSON-serializable.
JsonValue = Any

EntryType = Literal["message", "compaction", "branch_summary", "custom"]


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


class EntryBase(KarenBase):
    """`seq` and `timestamp` are assigned by storage at commit time."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    parent_id: Optional[str] = None
    seq: int = 0
    timestamp: int = 0
    custom_type: Optional[str] = None


class MessageEntry(EntryBase):
    type: Literal["message"] = "message"
    message: AgentMessage = None
    terminate: Optional[Literal[True]] = None


class CompactionEntry(EntryBase):
    type: Literal["compaction"] = "compaction"
    summary: str
    retained_tail: List[AgentMessage] = Field(default_factory=list)
    tokens_before: int
    details: Optional[JsonValue] = None
    usage: Optional[Usage] = None
    from_hook: bool


class BranchSummaryEntry(EntryBase):
    type: Literal["branch_summary"] = "branch_summary"
    from_id: Optional[str] = None
    summary: str
    details: Optional[JsonValue] = None
    usage: Optional[Usage] = None
    from_hook: bool


class CustomEntry(EntryBase):
    type: Literal["custom"] = "custom"
    custom_type: str
    data: Optional[JsonValue] = None


Entry = Annotated[
    Union[MessageEntry, CompactionEntry, BranchSummaryEntry, CustomEntry],
    Field(discriminator="type"),
]

#: Convert an application-defined custom entry into model context.
EntryProjector = Callable[[CustomEntry, Any], Any]


# ---------------------------------------------------------------------------
# Lane / inbox state (stored under reserved value addresses)
# ---------------------------------------------------------------------------


class LaneModelRef(KarenBase):
    provider: str
    model_id: str


class LaneConfiguration(KarenBase):
    model: LaneModelRef
    thinking_level: str
    active_tool_names: List[str] = Field(default_factory=list)


InboxItemKind = Literal["steer", "followUp", "nextRun", "write"]


class InboxItem(KarenBase):
    entry_id: str
    kind: InboxItemKind


class LaneState(KarenBase):
    current_operation_id: Optional[str] = None
    last_operation_id: Optional[str] = None
    inbox: List[InboxItem] = Field(default_factory=list)


class OperationError(KarenBase):
    code: str
    message: str
    details: Optional[JsonValue] = None


TerminalStatus = Literal["completed", "declined", "aborted", "failed"]


class OperationResultRecord(KarenBase):
    """Immutable lane-lived observation record written by one terminal transaction."""

    operation_id: str
    kind: Literal["run", "compaction", "navigation"]
    status: TerminalStatus
    error: Optional[OperationError] = None
    from_tip_id: Optional[str] = None
    tip_id: Optional[str] = None
    started_at: int
    ended_at: int


class PendingMessageEntry(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["message"] = "message"
    payload: AgentMessage = None


class PendingCustomEntry(KarenBase):
    type: Literal["custom"] = "custom"
    custom_type: str
    payload: Optional[JsonValue] = None


PendingEntry = Annotated[
    Union[PendingMessageEntry, PendingCustomEntry],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Usage rows and writes
# ---------------------------------------------------------------------------


class UsageRow(KarenBase):
    """`seq` is assigned by storage at commit time."""

    id: str
    seq: int = 0
    usage: Usage
    entry_id: Optional[str] = None
    adjustment: bool
    details: Optional[JsonValue] = None


class EntryWrite(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["entry"] = "entry"
    entry: Entry


class UsageWrite(KarenBase):
    kind: Literal["usage"] = "usage"
    row: UsageRow


class ValueSetWrite(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["value"] = "value"
    op: Literal["set"] = "set"
    seq: int = 0
    namespace: str
    key: str
    value: Any = None


class ValueDeleteWrite(KarenBase):
    kind: Literal["value"] = "value"
    op: Literal["delete"] = "delete"
    seq: int = 0
    namespace: str
    key: str


class ListAppendWrite(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["list"] = "list"
    op: Literal["append"] = "append"
    seq: int = 0
    namespace: str
    key: str
    value: Any = None


class ListDeleteWrite(KarenBase):
    kind: Literal["list"] = "list"
    op: Literal["delete"] = "delete"
    seq: int = 0
    namespace: str
    key: str


ValueWrite = Annotated[Union[ValueSetWrite, ValueDeleteWrite], Field(discriminator="op")]
ListWrite = Annotated[Union[ListAppendWrite, ListDeleteWrite], Field(discriminator="op")]
Write = Union[EntryWrite, UsageWrite, ValueSetWrite, ValueDeleteWrite, ListAppendWrite, ListDeleteWrite]


# ---------------------------------------------------------------------------
# Commits, scans, stats
# ---------------------------------------------------------------------------


class SessionStats(KarenBase):
    message_count: int = 0
    usage: Usage = Field(default_factory=Usage)


class CommitResult(KarenBase):
    first_seq: int
    seqs: List[int]
    timestamp: int
    #: Session totals immediately after this commit was applied.
    stats: SessionStats


class EntryStructure(KarenBase):
    id: str
    parent_id: Optional[str]
    seq: int
    timestamp: int
    type: EntryType
    custom_type: Optional[str] = None


class EntryCursor(KarenBase):
    seq: int


class BranchScan(KarenBase):
    start: Optional[str] = None
    stop_at_type: Optional[EntryType] = None
    stop_at_id: Optional[str] = None
    type: Optional[EntryType] = None
    custom_type: Optional[str] = None
    order: Optional[Literal["newestFirst", "oldestFirst"]] = None
    limit: Optional[int] = None
    cursor: Optional[EntryCursor] = None


class StorageBranchScan(BranchScan):
    start: str  # required for storage-level scans


class EntryScan(KarenBase):
    type: Optional[EntryType] = None
    custom_type: Optional[str] = None
    from_seq: Optional[int] = None
    to_seq: Optional[int] = None
    order: Optional[Literal["asc", "desc"]] = None
    limit: Optional[int] = None


class UsageScan(KarenBase):
    from_seq: Optional[int] = None
    to_seq: Optional[int] = None
    order: Optional[Literal["asc", "desc"]] = None
    limit: Optional[int] = None


class EntryQuery(KarenBase):
    type: Optional[EntryType] = None
    custom_type: Optional[str] = None
    order: Optional[Literal["asc", "desc"]] = None
    limit: Optional[int] = None
    cursor: Optional[EntryCursor] = None


# ---------------------------------------------------------------------------
# Storage and session interfaces
# ---------------------------------------------------------------------------


class SessionMetadata(KarenBase):
    id: str
    created_at: int
    storage_version: int
    cwd: Optional[str] = None
    parent_session_id: Optional[str] = None
    legacy_parent_session_path: Optional[str] = None


class IdGenerator(Protocol):
    def next(self, timestamp_ms: Optional[int] = None) -> str: ...


class Storage(ABC):
    """Durable session state backend (pi's `Storage`)."""

    @abstractmethod
    async def commit(self, writes: List[Write]) -> CommitResult: ...

    @abstractmethod
    async def get_entries(self, ids: List[str]) -> Dict[str, Entry]: ...

    @abstractmethod
    async def get_value(self, address: Any) -> Optional[Any]: ...  # Value[T] -> StoredValue[T]

    @abstractmethod
    async def scan_values(self, prefix: Any) -> List[Any]: ...

    @abstractmethod
    async def read_list(self, address: Any, options: Optional[Any] = None) -> List[Any]: ...

    @abstractmethod
    async def scan_branch(self, query: StorageBranchScan) -> List[Entry]: ...

    @abstractmethod
    async def scan_branch_structure(self, query: StorageBranchScan) -> List[EntryStructure]: ...

    @abstractmethod
    async def scan_entries(self, query: EntryScan) -> List[Entry]: ...

    @abstractmethod
    async def scan_usage(self, query: UsageScan) -> List[UsageRow]: ...

    @abstractmethod
    async def get_stats(self) -> SessionStats: ...

    @abstractmethod
    def close(self) -> Awaitable[None]: ...


class SessionReader(ABC):
    @abstractmethod
    async def get_entries(self, ids: List[str]) -> Dict[str, Entry]: ...

    @abstractmethod
    async def get_stats(self) -> SessionStats: ...

    @abstractmethod
    async def get_value(self, address: Any) -> Optional[Any]: ...

    @abstractmethod
    async def scan_values(self, prefix: Any) -> List[Any]: ...

    @abstractmethod
    async def read_list(self, address: Any, options: Optional[Any] = None) -> List[Any]: ...

    @abstractmethod
    async def scan_branch(self, query: StorageBranchScan) -> List[Entry]: ...


class SessionMutation(SessionReader):
    """Exclusive keyless mutation barrier for one Session."""

    @abstractmethod
    async def commit(self, writes: List[Write]) -> CommitResult:
        """Exactly zero or one commit attempt. A second attempt raises."""

    @abstractmethod
    async def end(self) -> None:
        """Wait for any commit attempt, invalidate the capability, release the barrier."""


SessionMutator = SessionMutation
#: Trusted exclusive callback over the Session mutation line. Use the supplied
#: mutator for the callback's sole commit; awaiting a public Session writer from
#: inside the callback deadlocks (it would be queued behind the callback).
SessionMutationCallback = Callable[[SessionMutator], Any]


class Branch(ABC):
    name: str

    @abstractmethod
    async def get_tip_id(self) -> Optional[str]: ...

    @abstractmethod
    async def find_entries(self, query: Optional[BranchScan] = None) -> List[Entry]: ...

    @abstractmethod
    async def find_entry(self, query: Optional[BranchScan] = None) -> Optional[Entry]: ...

    @abstractmethod
    async def append_message(self, message: AgentMessage) -> str: ...

    @abstractmethod
    async def append_custom_entry(self, custom_type: str, data: Optional[JsonValue] = None) -> str: ...


class Session(SessionReader):
    metadata: SessionMetadata
    id_generator: IdGenerator

    @abstractmethod
    async def get_entry(self, id: str) -> Optional[Entry]: ...

    @abstractmethod
    async def get_name(self) -> Optional[str]: ...

    @abstractmethod
    async def get_label(self, target_id: str) -> Optional[str]: ...

    @abstractmethod
    async def find_entries(self, query: Optional[EntryQuery] = None) -> List[Entry]: ...

    @abstractmethod
    async def find_entry(self, query: Optional[EntryQuery] = None) -> Optional[Entry]: ...

    @abstractmethod
    async def branch(self, name: str) -> Optional[Branch]: ...

    @abstractmethod
    async def create_branch(self, name: str, at: Optional[str]) -> Branch: ...

    @abstractmethod
    async def begin_mutation(self) -> SessionMutation: ...

    @abstractmethod
    async def mutate(self, mutation: SessionMutationCallback) -> Any: ...

    @abstractmethod
    async def set_value(self, address: Any, next: Any) -> None: ...

    @abstractmethod
    async def delete_value(self, address: Any) -> None: ...

    @abstractmethod
    async def append_list(self, address: Any, element: Any) -> None: ...

    @abstractmethod
    async def delete_list(self, address: Any) -> None: ...

    @abstractmethod
    async def set_name(self, name: Optional[str]) -> None: ...

    @abstractmethod
    async def set_label(self, target_id: str, label: Optional[str]) -> None: ...

    @abstractmethod
    def close(self) -> Awaitable[None]: ...


# ---------------------------------------------------------------------------
# Fork and repo
# ---------------------------------------------------------------------------


class BranchForkOptions(KarenBase):
    """Copy one path from a configured source lane under the same branch name."""

    scope: Literal["branch"] = "branch"
    #: Source Branch to copy.
    branch: str
    #: Entry on the source branch's current tip ancestry. Defaults to the tip.
    entry_id: Optional[str] = None
    #: Include the selected entry ("at", default) or stop at its parent ("before").
    position: Literal["before", "at"] = "at"
    #: Optional destination session id.
    id: Optional[str] = None


class TreeForkOptions(KarenBase):
    """Copy the whole conversation tree and every branch tip.

    Each configured lane copies configuration plus fresh idle state; data-only
    branches remain data-only. Operation/pending/result/usage state is excluded.
    """

    scope: Literal["tree"] = "tree"
    id: Optional[str] = None


ForkOptions = Annotated[Union[BranchForkOptions, TreeForkOptions], Field(discriminator="scope")]


class SessionCreateOptions(KarenBase):
    id: Optional[str] = None
    parent_session_id: Optional[str] = None


class SessionRepo(ABC):
    @abstractmethod
    async def create(self, options: SessionCreateOptions) -> Session: ...

    @abstractmethod
    async def open(self, metadata: SessionMetadata) -> Session: ...

    @abstractmethod
    async def list(self, options: Optional[Any] = None) -> List[SessionMetadata]: ...

    @abstractmethod
    async def delete(self, metadata: SessionMetadata) -> None: ...

    @abstractmethod
    async def fork(self, source: SessionMetadata, options: ForkOptions) -> Session: ...


__all__ = [
    "AgentMessage",
    "Branch",
    "BranchForkOptions",
    "BranchScan",
    "BranchSummaryEntry",
    "CommitResult",
    "CompactionEntry",
    "CustomEntry",
    "Entry",
    "EntryBase",
    "EntryCursor",
    "EntryProjector",
    "EntryQuery",
    "EntryScan",
    "EntryStructure",
    "EntryType",
    "EntryWrite",
    "ForkOptions",
    "IdGenerator",
    "InboxItem",
    "InboxItemKind",
    "JsonValue",
    "LaneConfiguration",
    "LaneModelRef",
    "LaneState",
    "ListAppendWrite",
    "ListDeleteWrite",
    "ListWrite",
    "MessageEntry",
    "OperationError",
    "OperationResultRecord",
    "PendingCustomEntry",
    "PendingEntry",
    "PendingMessageEntry",
    "Session",
    "SessionCreateOptions",
    "SessionMetadata",
    "SessionMutation",
    "SessionMutationCallback",
    "SessionMutator",
    "SessionReader",
    "SessionRepo",
    "SessionStats",
    "Storage",
    "StorageBranchScan",
    "TerminalStatus",
    "TreeForkOptions",
    "UsageRow",
    "UsageScan",
    "UsageWrite",
    "ValueDeleteWrite",
    "ValueSetWrite",
    "ValueWrite",
    "Write",
]

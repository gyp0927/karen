"""Session persistence for karen-agent (pi's `harness/session/`).

JSONL-backed durable sessions with branches, typed values/lists, usage rows,
fork (branch/tree scope), and resume-by-reopen. pi's chord `Context` parameter
is dropped throughout, the injected `FileSystem` capability is replaced by
direct `pathlib` I/O, and the legacy v3 migration path is not ported.

Quick start:

    repo = JsonlSessionRepo("~/.karen/sessions")
    session = await repo.create(JsonlSessionCreateOptions(cwd=os.getcwd()))
    branch = await session.create_branch("main", None)
    await branch.append_message(UserMessage(content="hi", timestamp=...))
    await session.close()
    # resume:
    session = await repo.open((await repo.list(JsonlSessionListOptions(cwd=os.getcwd())))[0])
"""

from .commit import (
    CommitValidationState,
    PreparedCommit,
    commit_write,
    insert_entry,
    insert_usage,
    materialize_committed_entry,
    prepare_storage_commit,
    validate_committed_writes,
)
from .fork import ForkDestinationSnapshot, ForkSourceSnapshot, create_fork_snapshot
from .fork_policy import (
    UNDEFINED,
    BranchForkPlan,
    ForkCurrentStatePlan,
    TreeForkPlan,
    project_fork_current_state_write,
    select_branch_fork,
)
from .ids import Uuid7IdGenerator, uuid7
from .memory import MemorySessionRepo, MemoryStorage
from .mutation_line import MutationLine
from .session import (
    SessionBranchExistsError,
    SessionInvariantError,
    SessionInvalidBranchError,
    SessionPendingAssistantMessageError,
    SessionUnknownTargetError,
    StorageBackedSession,
)
from .storage_state import InMemoryStorageState
from .types import *  # noqa: F401,F403 — pi's index.ts does `export * from "./types.ts"`
from .values import (
    ListCursor,
    ListElement,
    ListReadOptions,
    ResolvedListReadOptions,
    StoredValue,
    Value,
    ValueList,
    append_list,
    branch_tip,
    branch_tip_inventory_prefix,
    delete_list,
    delete_value,
    entry_label,
    lane_config,
    lane_state,
    operation_meta,
    operation_preparation,
    operation_preparation_prefix,
    operation_result,
    operation_state,
    operation_tool_args,
    operation_tool_args_prefix,
    operation_tool_memo,
    operation_tool_memo_prefix,
    pending_assistant_frames,
    pending_entry,
    pending_tool_output,
    pending_tool_output_prefix,
    resolve_list_read_options,
    session_name,
    set_value,
    value,
    value_list,
)
from .jsonl import (
    JSONL_FORMAT_VERSION,
    JSONL_STORAGE_VERSION,
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionMetadata,
    JsonlSessionRepo,
    JsonlStorage,
    JsonlStorageHeader,
)

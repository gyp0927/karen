"""Fork planning and current-state projection (pi's `harness/session/fork-policy.ts`)."""

from __future__ import annotations

from typing import Callable, List, Literal, Optional, Union

from karen_ai.types import KarenBase
from typing_extensions import Annotated

from pydantic import Field

from .types import BranchForkOptions, ListAppendWrite, ValueSetWrite


class _Undefined:
    """pi's `undefined` (absent) as distinct from `None` (JSON null)."""

    _instance = None

    def __new__(cls) -> "_Undefined":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNDEFINED"

    def __bool__(self) -> bool:
        return False


UNDEFINED = _Undefined()


class BranchForkPlan(KarenBase):
    scope: Literal["branch"] = "branch"
    branch: str
    destination_tip: Optional[str]


class TreeForkPlan(KarenBase):
    scope: Literal["tree"] = "tree"


ForkCurrentStatePlan = Annotated[Union[BranchForkPlan, TreeForkPlan], Field(discriminator="scope")]


def select_branch_fork(
    options: BranchForkOptions,
    *,
    tip,
    get_parent: Callable[[str], object],
    select_entry: Callable[[str], None],
) -> BranchForkPlan:
    """Validate and select the branch ancestry to copy.

    `tip`: the source branch tip value, or `UNDEFINED` when the branch is unknown.
    `get_parent`: entry id -> parent id, `None` at the root, `UNDEFINED` when missing.
    `select_entry`: marks an entry id for copying.
    """
    if tip is UNDEFINED:
        raise ValueError(f"Unknown source branch: {options.branch}")
    requested = options.entry_id if options.entry_id is not None else tip
    found = requested is None
    destination_tip: Optional[str] = None
    entry_id: Optional[str] = tip
    while entry_id is not None:
        parent_id = get_parent(entry_id)
        if parent_id is UNDEFINED:
            raise ValueError(f"Corrupt source branch: missing parent {entry_id}")
        if entry_id == requested:
            found = True
            destination_tip = parent_id if options.position == "before" else entry_id
            if options.position != "before":
                select_entry(entry_id)
        elif found:
            select_entry(entry_id)
        entry_id = parent_id
    if not found:
        raise ValueError(f"Fork entry {requested} is not on source branch {options.branch!r}")
    return BranchForkPlan(branch=options.branch, destination_tip=destination_tip)


def project_fork_current_state_write(
    write,
    plan: ForkCurrentStatePlan,
    is_entry_copied: Callable[[str], bool],
):
    """Project one current scalar row or surviving list element into destination state.

    Returns the (possibly rewritten) write, or `None` when the fork excludes it.
    `write` is a committed `ValueSetWrite` or `ListAppendWrite`.
    """
    namespace = write.namespace
    if namespace == "pi.session.name":
        return write
    if namespace == "pi.entry.label":
        return write if is_entry_copied(write.key) else None
    if namespace == "pi.branch.tip":
        if isinstance(plan, TreeForkPlan):
            return write
        if write.key == plan.branch:
            return write.model_copy(update={"value": plan.destination_tip})
        return None
    if namespace == "pi.lane.config":
        return write if isinstance(plan, TreeForkPlan) or write.key == plan.branch else None
    if namespace == "pi.lane.state":
        if isinstance(plan, TreeForkPlan) or write.key == plan.branch:
            # Copied lanes restart with fresh idle state (pi's plain-object literal).
            idle = {"currentOperationId": None, "lastOperationId": None, "inbox": []}
            return write.model_copy(update={"value": idle})
        return None
    if namespace == "pi.result":
        return None
    if namespace.startswith("pi.op.") or namespace.startswith("pi.pending."):
        return None
    if namespace == "pi" or namespace.startswith("pi."):
        raise ValueError(f"Unknown reserved fork namespace: {namespace}")
    return write if isinstance(plan, TreeForkPlan) else None

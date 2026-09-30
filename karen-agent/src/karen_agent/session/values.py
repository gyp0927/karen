"""Typed value/list addresses and write constructors (pi's `harness/session/values.ts`).

A `Value` / `ValueList` is a typed *address* (namespace + key); storage keeps one
current value per scalar address and an append-only element log per list address.
The `pi.*` namespaces are reserved by the session/runtime machinery; applications
should use their own namespaces.

The `pi.op.*` / `pi.pending.*` payloads belong to pi's durable runtime (not yet
ported) and are typed as `Any` here.
"""

from __future__ import annotations

from typing import Any, Generic, List, Literal, Optional, TypeVar

from karen_ai.types import KarenBase
from pydantic import ConfigDict

from .types import (
    LaneConfiguration,
    LaneState,
    ListAppendWrite,
    ListDeleteWrite,
    OperationResultRecord,
    PendingEntry,
    ValueDeleteWrite,
    ValueSetWrite,
)

T = TypeVar("T")

MAX_SAFE_INTEGER = (1 << 53) - 1  # JS Number.MAX_SAFE_INTEGER
DEFAULT_LIST_LIMIT = 1_000
MAX_LIST_LIMIT = 10_000


class StoredAddressBase(KarenBase, Generic[T]):
    model_config = ConfigDict(frozen=True)

    namespace: str
    key: str = ""


class Value(StoredAddressBase[T]):
    kind: Literal["value"] = "value"


class ValueList(StoredAddressBase[T]):
    kind: Literal["list"] = "list"


class StoredValue(KarenBase, Generic[T]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    address: Value[T]
    value: T
    seq: int


class ListElement(KarenBase, Generic[T]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    seq: int
    value: T


class ListCursor(KarenBase):
    seq: int


class ListReadOptions(KarenBase):
    cursor: Optional[ListCursor] = None
    order: Optional[Literal["asc", "desc"]] = None
    limit: Optional[int] = None


class ResolvedListReadOptions(KarenBase):
    cursor: Optional[ListCursor] = None
    order: Literal["asc", "desc"] = "asc"
    limit: int = DEFAULT_LIST_LIMIT


def _validate_address(namespace: str, key: str) -> None:
    if len(namespace) == 0:
        raise TypeError("Value namespace must not be empty")
    if "\x00" in namespace:
        raise TypeError("Value namespace must not contain \\u0000")
    if "\x00" in key:
        raise TypeError("Value key must not contain \\u0000")


def value(namespace: str, key: str = "") -> Value[Any]:
    _validate_address(namespace, key)
    return Value(namespace=namespace, key=key)


def value_list(namespace: str, key: str = "") -> ValueList[Any]:
    """pi names this constructor `list`; renamed to avoid shadowing the builtin."""

    _validate_address(namespace, key)
    return ValueList(namespace=namespace, key=key)


def set_value(address: Value[T], next: T) -> ValueSetWrite:
    return ValueSetWrite(namespace=address.namespace, key=address.key, value=next)


def delete_value(address: Value[Any]) -> ValueDeleteWrite:
    return ValueDeleteWrite(namespace=address.namespace, key=address.key)


def append_list(address: ValueList[T], element: T) -> ListAppendWrite:
    return ListAppendWrite(namespace=address.namespace, key=address.key, value=element)


def delete_list(address: ValueList[Any]) -> ListDeleteWrite:
    return ListDeleteWrite(namespace=address.namespace, key=address.key)


def resolve_list_read_options(options: Optional[ListReadOptions] = None) -> ResolvedListReadOptions:
    options = options or ListReadOptions()
    requested_limit = DEFAULT_LIST_LIMIT if options.limit is None else options.limit
    if (
        not isinstance(requested_limit, int)
        or isinstance(requested_limit, bool)
        or requested_limit <= 0
        or requested_limit > MAX_SAFE_INTEGER
    ):
        raise TypeError("List read limit must be a positive safe integer")
    return ResolvedListReadOptions(
        cursor=options.cursor,
        order=options.order or "asc",
        limit=min(requested_limit, MAX_LIST_LIMIT),
    )


# ---------------------------------------------------------------------------
# Reserved addresses
# ---------------------------------------------------------------------------


def branch_tip(branch: str) -> Value[Optional[str]]:
    return Value(namespace="pi.branch.tip", key=branch)


def branch_tip_inventory_prefix() -> Value[Optional[str]]:
    return Value(namespace="pi.branch.tip")


def lane_config(lane: str) -> Value[LaneConfiguration]:
    return Value(namespace="pi.lane.config", key=lane)


def lane_state(lane: str) -> Value[LaneState]:
    return Value(namespace="pi.lane.state", key=lane)


def operation_result(operation_id: str) -> Value[OperationResultRecord]:
    return Value(namespace="pi.result", key=operation_id)


def operation_meta(operation_id: str) -> Value[Any]:
    return Value(namespace="pi.op.meta", key=operation_id)


def operation_state(operation_id: str) -> Value[Any]:
    return Value(namespace="pi.op.state", key=operation_id)


def operation_tool_args(operation_id: str, step_id: str, source_index: int) -> Value[Any]:
    return Value(namespace="pi.op.tool_args", key=f"{operation_id}:{step_id}:{source_index}")


def operation_tool_memo(operation_id: str, invocation_id: str, name: str) -> Value[Any]:
    return Value(namespace="pi.op.tool_memo", key=f"{operation_id}:{invocation_id}:{name}")


def operation_preparation(operation_id: str, task_id: str) -> Value[Any]:
    return Value(namespace="pi.op.preparation", key=f"{operation_id}:{task_id}")


def operation_tool_args_prefix(operation_id: str, step_id: Optional[str] = None) -> Value[Any]:
    key = f"{operation_id}:" if step_id is None else f"{operation_id}:{step_id}:"
    return Value(namespace="pi.op.tool_args", key=key)


def operation_tool_memo_prefix(operation_id: str, invocation_id: Optional[str] = None) -> Value[Any]:
    key = f"{operation_id}:" if invocation_id is None else f"{operation_id}:{invocation_id}:"
    return Value(namespace="pi.op.tool_memo", key=key)


def operation_preparation_prefix(operation_id: str) -> Value[Any]:
    return Value(namespace="pi.op.preparation", key=f"{operation_id}:")


def pending_entry(entry_id: str) -> Value[PendingEntry]:
    return Value(namespace="pi.pending.entry", key=entry_id)


def pending_tool_output(operation_id: str, invocation_id: str) -> Value[Any]:
    return Value(namespace="pi.pending.tool_output", key=f"{operation_id}:{invocation_id}")


def pending_assistant_frames(operation_id: str, response_entry_id: str) -> ValueList[Any]:
    return ValueList(namespace="pi.pending.assistant_frame", key=f"{operation_id}:{response_entry_id}")


def pending_tool_output_prefix(operation_id: str) -> Value[Any]:
    return Value(namespace="pi.pending.tool_output", key=f"{operation_id}:")


#: Address of the optional session display name.
session_name: Value[str] = Value(namespace="pi.session.name")


def entry_label(entry_id: str) -> Value[str]:
    return Value(namespace="pi.entry.label", key=entry_id)

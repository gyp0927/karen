"""Value/list addresses and write constructors (pi's values.ts behaviors)."""

import pytest

from karen_agent.session.types import LaneState
from karen_agent.session.values import (
    ListCursor,
    ListReadOptions,
    append_list,
    branch_tip,
    delete_list,
    delete_value,
    entry_label,
    lane_config,
    lane_state,
    operation_result,
    pending_assistant_frames,
    pending_entry,
    resolve_list_read_options,
    session_name,
    set_value,
    value,
    value_list,
)


def test_address_construction():
    address = value("app.counter")
    assert address.namespace == "app.counter"
    assert address.key == ""
    assert address.kind == "value"
    lst = value_list("app.events", "main")
    assert lst.kind == "list"
    assert lst.key == "main"


def test_address_validation():
    with pytest.raises(TypeError, match="namespace must not be empty"):
        value("")
    with pytest.raises(TypeError, match="namespace must not contain"):
        value("app\x00x")
    with pytest.raises(TypeError, match="key must not contain"):
        value("app", "k\x00")
    with pytest.raises(TypeError):
        value_list("")


def test_write_constructors():
    address = value("app.counter", "c")
    write = set_value(address, 3)
    assert (write.kind, write.op, write.namespace, write.key, write.value) == ("value", "set", "app.counter", "c", 3)
    delete = delete_value(address)
    assert (delete.kind, delete.op, delete.namespace, delete.key) == ("value", "delete", "app.counter", "c")

    lst = value_list("app.events")
    appended = append_list(lst, {"n": 1})
    assert (appended.kind, appended.op, appended.value) == ("list", "append", {"n": 1})
    deleted = delete_list(lst)
    assert (deleted.kind, deleted.op) == ("list", "delete")


def test_resolve_list_read_options():
    resolved = resolve_list_read_options(None)
    assert (resolved.order, resolved.limit, resolved.cursor) == ("asc", 1_000, None)

    resolved = resolve_list_read_options(ListReadOptions(order="desc", limit=5, cursor=ListCursor(seq=3)))
    assert (resolved.order, resolved.limit, resolved.cursor.seq) == ("desc", 5, 3)

    # Limits are capped at 10_000.
    assert resolve_list_read_options(ListReadOptions(limit=999_999)).limit == 10_000

    for bad in (0, -1, 1.5, True, 2**53):
        # model_construct bypasses pydantic validation to reach the runtime check.
        with pytest.raises(TypeError, match="positive safe integer"):
            resolve_list_read_options(ListReadOptions.model_construct(limit=bad))


def test_reserved_addresses():
    assert branch_tip("main").namespace == "pi.branch.tip"
    assert lane_config("main").namespace == "pi.lane.config"
    assert lane_state("main").namespace == "pi.lane.state"
    assert operation_result("op1").key == "op1"
    assert pending_entry("e1").namespace == "pi.pending.entry"
    frames = pending_assistant_frames("op1", "r1")
    assert frames.kind == "list" and frames.key == "op1:r1"
    assert session_name.namespace == "pi.session.name" and session_name.key == ""
    assert entry_label("e1").key == "e1"


def test_lane_state_default_is_idle():
    state = LaneState()
    assert state.current_operation_id is None
    assert state.inbox == []
    assert state.model_dump(mode="json", by_alias=True) == {
        "currentOperationId": None,
        "lastOperationId": None,
        "inbox": [],
    }

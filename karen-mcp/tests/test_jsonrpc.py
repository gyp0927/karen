"""JSON-RPC guards and the error taxonomy (`protocol/jsonrpc.py`)."""

import pytest

from karen_mcp import (
    JSON_RPC_ERROR_CODES,
    McpAbortError,
    McpError,
    McpTimeoutError,
    is_json_rpc_notification,
    is_json_rpc_request,
    is_json_rpc_response,
    parse_json_rpc_message,
)


def test_request_notification_response_predicates():
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    success = {"jsonrpc": "2.0", "id": 1, "result": {}}
    error = {"jsonrpc": "2.0", "id": "a", "error": {"code": -32601, "message": "nope"}}

    assert is_json_rpc_request(request) and not is_json_rpc_notification(request)
    assert is_json_rpc_notification(notification) and not is_json_rpc_request(notification)
    assert is_json_rpc_response(success) and is_json_rpc_response(error)
    # an id without a method is neither a request nor a notification
    assert not is_json_rpc_request({"jsonrpc": "2.0", "id": 1})
    # a response carries exactly one of result/error
    assert not is_json_rpc_response({"jsonrpc": "2.0", "id": 1, "result": 1, "error": {"code": 1, "message": "x"}})
    assert not is_json_rpc_response({"jsonrpc": "2.0", "id": 1, "error": {"code": "1", "message": "x"}})
    # ids are strings or finite numbers, never booleans
    assert not is_json_rpc_request({"jsonrpc": "2.0", "id": True, "method": "ping"})
    assert not is_json_rpc_response({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert not is_json_rpc_request({"id": 1, "method": "ping"})


def test_parse_rejects_anything_else():
    with pytest.raises(McpError) as error:
        parse_json_rpc_message({"jsonrpc": "1.0", "id": 1, "method": "ping"})
    assert error.value.code == JSON_RPC_ERROR_CODES["invalidRequest"]
    assert str(error.value) == "Invalid JSON-RPC message"


def test_error_objects_carry_their_payload():
    error = McpError(1234, "tool failed", {"retryable": False})
    assert (error.code, error.message, error.data) == (1234, "tool failed", {"retryable": False})
    assert error.to_dict() == {"code": 1234, "message": "tool failed", "data": {"retryable": False}}
    assert McpError(-32603, "boom").to_dict() == {"code": -32603, "message": "boom"}
    assert str(McpTimeoutError(50)) == "MCP request timed out after 50ms"
    assert McpTimeoutError(50).timeout_ms == 50
    assert str(McpAbortError()) == "MCP request aborted"

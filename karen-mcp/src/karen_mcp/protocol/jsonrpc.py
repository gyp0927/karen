"""JSON-RPC 2.0 envelopes and the error taxonomy (pi's `protocol/jsonrpc.ts`).

Messages stay plain dicts end to end: pi carries them structurally too (type
guards, no classes), and every layer between the transport and the client just
passes them along — modelling them would only mean unwrapping them again to
write them back out.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Union

__all__ = [
    "JSON_RPC_ERROR_CODES",
    "JsonRpcId",
    "McpAbortError",
    "McpConnectionClosedError",
    "McpError",
    "McpTimeoutError",
    "is_json_rpc_id",
    "is_json_rpc_notification",
    "is_json_rpc_request",
    "is_json_rpc_response",
    "is_object",
    "parse_json_rpc_message",
    "to_error",
]

JsonRpcId = Union[str, int]

#: pi's `JSON_RPC_ERROR_CODES`.
JSON_RPC_ERROR_CODES: Dict[str, int] = {
    "parseError": -32700,
    "invalidRequest": -32600,
    "methodNotFound": -32601,
    "invalidParams": -32602,
    "internalError": -32603,
}


class McpError(Exception):
    """A JSON-RPC error object raised as an exception (pi's `McpError`)."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        """The `error` member of a JSON-RPC error response."""
        error: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error


class McpConnectionClosedError(Exception):
    """The connection is gone: never connected, closed, or dropped."""

    def __init__(self, message: str = "MCP connection closed") -> None:
        super().__init__(message)


class McpTimeoutError(Exception):
    def __init__(self, timeout_ms: float) -> None:
        super().__init__(f"MCP request timed out after {timeout_ms}ms")
        self.timeout_ms = timeout_ms


class McpAbortError(Exception):
    """An aborted request. pi names it `AbortError` so callers can tell an
    abort from a transport failure by name."""

    def __init__(self, message: str = "MCP request aborted") -> None:
        super().__init__(message)


def is_object(value: Any) -> bool:
    return isinstance(value, dict)


def to_error(value: Any) -> Exception:
    return value if isinstance(value, BaseException) else Exception(str(value))


def is_json_rpc_id(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return True
    return isinstance(value, (int, float)) and math.isfinite(value)


def is_json_rpc_request(message: Any) -> bool:
    return (
        is_object(message)
        and message.get("jsonrpc") == "2.0"
        and is_json_rpc_id(message.get("id"))
        and isinstance(message.get("method"), str)
    )


def is_json_rpc_notification(message: Any) -> bool:
    return (
        is_object(message)
        and message.get("jsonrpc") == "2.0"
        and "id" not in message
        and isinstance(message.get("method"), str)
    )


def is_json_rpc_response(message: Any) -> bool:
    if not is_object(message) or message.get("jsonrpc") != "2.0" or not is_json_rpc_id(message.get("id")):
        return False
    if "result" in message:
        return "error" not in message
    if "error" not in message or not is_object(message["error"]):
        return False
    return isinstance(message["error"].get("code"), int) and isinstance(message["error"].get("message"), str)


def parse_json_rpc_message(value: Any) -> Dict[str, Any]:
    """Validate one decoded JSON-RPC message, or raise `McpError(-32600)`."""
    if is_json_rpc_request(value) or is_json_rpc_notification(value) or is_json_rpc_response(value):
        return value
    raise McpError(JSON_RPC_ERROR_CODES["invalidRequest"], "Invalid JSON-RPC message")

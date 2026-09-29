"""Assistant-message diagnostics, mirroring utils/diagnostics.ts."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..types import AssistantMessage, AssistantMessageDiagnostic, DiagnosticErrorInfo


def format_thrown_value(value: Any) -> str:
    if isinstance(value, BaseException):
        return str(value) or type(value).__name__
    if isinstance(value, str):
        return value
    return str(value)


def extract_diagnostic_error(error: Any) -> DiagnosticErrorInfo:
    if not isinstance(error, BaseException):
        return DiagnosticErrorInfo(name="ThrownValue", message=format_thrown_value(error))
    return DiagnosticErrorInfo(
        name=type(error).__name__ or None,
        message=str(error) or type(error).__name__,
    )


def create_assistant_message_diagnostic(
    type: str,
    error: Any,
    details: Optional[Dict[str, Any]] = None,
) -> AssistantMessageDiagnostic:
    return AssistantMessageDiagnostic(
        type=type,
        timestamp=int(time.time() * 1000),
        error=extract_diagnostic_error(error),
        details=details,
    )


def append_assistant_message_diagnostic(
    message: AssistantMessage,
    diagnostic: AssistantMessageDiagnostic,
) -> None:
    message.diagnostics = [*(message.diagnostics or []), diagnostic]

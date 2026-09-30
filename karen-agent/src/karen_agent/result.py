"""Fallible-operation result type (pi's `harness/types.ts` Result helpers + harness errors).

Expected failures are returned as ``Err`` values instead of thrown, matching pi's
``{ ok: false, error }`` convention. ``CompactionError`` and ``BranchSummaryError``
live here alongside the result helpers, as they do in pi's `harness/types.ts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Literal, Optional, TypeVar, Union

__all__ = [
    "Result",
    "Ok",
    "Err",
    "ok",
    "err",
    "get_or_throw",
    "get_or_undefined",
    "to_error",
    "CompactionErrorCode",
    "CompactionError",
    "BranchSummaryErrorCode",
    "BranchSummaryError",
]

TValue = TypeVar("TValue")
TError = TypeVar("TError")


@dataclass(frozen=True)
class Ok(Generic[TValue]):
    """Successful result carrying a value."""

    value: TValue
    ok: ClassVar[Literal[True]] = True


@dataclass(frozen=True)
class Err(Generic[TError]):
    """Failed result carrying an error."""

    error: TError
    ok: ClassVar[Literal[False]] = False


#: Result of a fallible operation: ``Ok(value)`` or ``Err(error)``.
Result = Union[Ok[TValue], Err[TError]]


def ok(value: TValue) -> Result[TValue, Any]:
    """Create a successful :data:`Result`."""
    return Ok(value)


def err(error: TError) -> Result[Any, TError]:
    """Create a failed :data:`Result`."""
    return Err(error)


def get_or_throw(result: Result[TValue, TError]) -> TValue:
    """Return the success value or raise the failure error.

    Intended for tests and explicit adapter boundaries.
    """
    if isinstance(result, Err):
        error = result.error
        if isinstance(error, BaseException):
            raise error
        raise Exception(error)
    return result.value


def get_or_undefined(result: Result[TValue, TError]) -> Optional[TValue]:
    """Return the success value or ``None``."""
    return result.value if isinstance(result, Ok) else None


def to_error(error: Any) -> Exception:
    """Normalize unknown thrown values into Exception instances."""
    if isinstance(error, Exception):
        return error
    if isinstance(error, str):
        return Exception(error)
    return Exception(str(error))


#: Stable compaction error codes returned by compaction helpers.
CompactionErrorCode = Literal["aborted", "summarization_failed"]


class CompactionError(Exception):
    """Error returned by compaction helpers."""

    def __init__(self, code: CompactionErrorCode, message: str, cause: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.code: CompactionErrorCode = code
        if cause is not None:
            self.__cause__ = cause


#: Stable branch-summary error codes returned by branch summarization helpers.
BranchSummaryErrorCode = Literal["aborted", "summarization_failed"]


class BranchSummaryError(Exception):
    """Error returned by branch summarization helpers."""

    def __init__(self, code: BranchSummaryErrorCode, message: str, cause: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.code: BranchSummaryErrorCode = code
        if cause is not None:
            self.__cause__ = cause

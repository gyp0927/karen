"""Error types for karen-ai, mirroring pi-ai's ModelsError."""

from __future__ import annotations

from typing import Literal, Optional

ModelsErrorCode = Literal[
    "auth",
    "oauth",
    "provider",
    "model_source",
    "stream",
]


class ModelsError(Exception):
    """Error raised by the Models registry and auth resolution.

    `code` classifies the failure for status UI and retry policy:
    - "auth": api-key resolution or credential store failure
    - "oauth": OAuth token refresh/derivation failure
    - "provider": unknown provider or unsupported provider operation
    - "model_source": model catalog refresh failure
    - "stream": no API implementation for a model's api
    """

    def __init__(self, code: ModelsErrorCode, message: str, *, cause: Optional[BaseException] = None):
        super().__init__(message)
        self.code: ModelsErrorCode = code
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause


class AbortError(Exception):
    """Raised when an operation is cancelled through an AbortSignal."""

    def __init__(self, message: str = "Request was aborted"):
        super().__init__(message)

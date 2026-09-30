"""Shell capture collector (pi's `harness/utils/shell-output.ts`).

`execute_shell_with_capture` is the compatibility helper for callers that need
one bounded final view: it runs `env.exec` with pi's default tail-retained
limits + spill, accumulates the published views, and folds execution failures
into a single result shape.

Deviation (same as `env/types.py`): karen's capture publishes complete
`ShellOutputView` snapshots instead of pi's replace/append/slide deltas, so the
incremental `on_chunk` text is recovered by prefix-diffing consecutive
snapshots. When the retained window slides (the new view no longer starts with
the previous one) the overlapping bytes cannot be told apart from new bytes, so
that update reports no chunk — matching pi, which also reports nothing for
post-cap replacements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from karen_ai import AbortSignal

from ..env.types import ExecutionEnv, ExecutionError, ShellExecOptions, ShellOutputCaptureOptions, ShellOutputLimits
from ..result import Err, Result, err, ok
from .output_capture import ShellOutputTruncation, ShellOutputView, sanitize_binary_output
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, truncate_tail

__all__ = [
    "ShellCaptureOptions",
    "ShellCaptureProgress",
    "ShellCaptureResult",
    "execute_shell_with_capture",
    "sanitize_binary_output",
]


@dataclass
class ShellCaptureProgress:
    output: str
    truncation: TruncationResult
    full_output_path: Optional[str] = None
    last_line_bytes: int = 0


@dataclass
class ShellCaptureOptions:
    """Subset of ShellExecOptions plus the compatibility chunk callback."""

    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    inherit_env: Optional[bool] = None
    timeout: Optional[float] = None
    #: Called with each incremental output chunk; `get_progress()` returns the current view.
    on_chunk: Optional[Callable[[str, Callable[[], ShellCaptureProgress]], None]] = None
    #: Return shell execution failures with captured output instead of as a failed Result.
    return_execution_errors: bool = False


@dataclass
class ShellCaptureResult(ShellCaptureProgress):
    exit_code: Optional[int] = None
    cancelled: bool = False
    truncated: bool = False
    execution_error: Optional[ExecutionError] = None


def _progress_from(output: ShellOutputView) -> ShellCaptureProgress:
    truncation = output.truncation
    return ShellCaptureProgress(
        output=output.text,
        truncation=TruncationResult(
            content=output.text,
            truncated=truncation.truncated,
            truncated_by=truncation.truncated_by,
            total_lines=truncation.total_lines,
            total_bytes=truncation.total_bytes,
            output_lines=truncation.output_lines,
            output_bytes=truncation.output_bytes,
            last_line_partial=truncation.last_line_partial,
            first_line_exceeds_limit=truncation.first_line_exceeds_limit,
            max_lines=truncation.max_lines,
            max_bytes=truncation.max_bytes,
        ),
        full_output_path=output.spill_path,
        last_line_bytes=output.last_line_bytes or 0,
    )


async def execute_shell_with_capture(
    env: ExecutionEnv,
    command: str,
    options: Optional[ShellCaptureOptions] = None,
    signal: Optional[AbortSignal] = None,
) -> Result[ShellCaptureResult, ExecutionError]:
    """Run `command` through `env.exec` and collect one bounded final view."""
    snapshots: List[ShellOutputView] = []

    def on_update(view: ShellOutputView) -> None:
        previous = snapshots[-1] if snapshots else None
        snapshots.append(view)
        if previous is None:
            chunk = view.text
        elif view.text.startswith(previous.text):
            chunk = view.text[len(previous.text) :]
        else:
            chunk = ""
        if chunk and options is not None and options.on_chunk is not None:
            options.on_chunk(chunk, lambda: _progress_from(snapshots[-1]))

    exec_options = ShellExecOptions(
        cwd=options.cwd if options else None,
        env=options.env if options else None,
        inherit_env=options.inherit_env if options else None,
        timeout=options.timeout if options else None,
        capture=ShellOutputCaptureOptions(
            limits=ShellOutputLimits(max_bytes=DEFAULT_MAX_BYTES, max_lines=DEFAULT_MAX_LINES, retain="tail"),
            spill=True,
        ),
        on_update=on_update,
    )
    result = await env.exec(command, exec_options, signal=signal)

    if not snapshots:
        empty = truncate_tail("")
        snapshots.append(
            ShellOutputView(
                text=empty.content,
                truncation=ShellOutputTruncation(**empty.model_dump(exclude={"content"})),
            )
        )
    progress = _progress_from(snapshots[-1])

    if isinstance(result, Err):
        error = result.error
        if error.code == "aborted" or (signal is not None and signal.aborted):
            return ok(
                ShellCaptureResult(
                    **vars(progress),
                    exit_code=None,
                    cancelled=True,
                    truncated=progress.truncation.truncated,
                )
            )
        if options is not None and options.return_execution_errors:
            return ok(
                ShellCaptureResult(
                    **vars(progress),
                    exit_code=None,
                    cancelled=False,
                    truncated=progress.truncation.truncated,
                    execution_error=error,
                )
            )
        return err(error)

    return ok(
        ShellCaptureResult(
            **vars(progress),
            exit_code=result.value.exit_code,
            cancelled=False,
            truncated=result.value.truncation.truncated,
        )
    )

"""TUI transcript model (pure logic, terminal-agnostic).

Consumes the agent event stream (the same events `cli.py`'s `_on_agent_event`
renders to stdout) and maintains an in-memory transcript the layout renderer
can paint: user prompts, streaming assistant text, tool executions, `!command`
shell runs and session notices. No terminal I/O and no asyncio here — the loop
in `app.py` feeds events in and calls `layout.render`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

#: `!command` output is kept as a bounded tail of this many complete lines
#: (plus the one still being written). The transcript is re-rendered from
#: scratch every frame, so a command that dumps a hundred thousand lines must
#: not make painting quadratic; whatever falls off is counted in
#: `BashExecution.dropped_lines` and shown as a marker.
_BASH_MAX_LINES = 400

#: Longest a single output line may get before its head is dropped (keeping the
#: tail — the newest part is what a progress bar or a minified file is showing).
#: A command with no line breaks at all (`base64 -w0`, a `\r`-redrawn progress
#: bar whose redraws `sanitize_shell_output` folds into one line) would
#: otherwise grow a single line without limit, and `textwrap` on a
#: whitespace-free string of megabytes is quadratic — the renderer would freeze
#: frame after frame. 1024 also keeps the 400-line worst case cheap to wrap.
#: A truncated line is prefixed with an ellipsis so it reads as truncated.
_BASH_MAX_LINE_CHARS = 1024


@dataclass
class ToolExecution:
    name: str
    tool_call_id: str = ""
    args: Any = None
    state: str = "running"  # "running" | "done" | "error"
    error_text: str = ""


@dataclass
class BashExecution:
    """A `!command` shell run (pi's `BashExecutionMessage`, live view).

    Output arrives as sanitized chunks (`AgentSession.execute_bash`'s
    `on_chunk`), so the block streams while the command runs instead of
    appearing all at once. `lines` always holds at least one element, and the
    last one is the line currently being written (it may be partial).
    """

    command: str = ""
    lines: List[str] = field(default_factory=lambda: [""])
    dropped_lines: int = 0
    #: "running" | "done" | "error" | "cancelled"
    state: str = "running"
    exit_code: Optional[int] = None
    #: The executor spilled the full output to a file (see `full_output_path`).
    truncated: bool = False
    full_output_path: Optional[str] = None
    error_text: str = ""

    def append(self, delta: str) -> None:
        parts = delta.split("\n")
        # Index of the line this delta continues (it takes `parts[0]`), kept so
        # the length bound below covers every line the delta touched.
        start = len(self.lines) - 1
        self.lines[-1] += parts[0]
        self.lines.extend(parts[1:])
        # The last element is the line still being written; only the complete
        # lines above it count against the bound, so `dropped_lines` is exactly
        # the number of whole lines that scrolled out of the window.
        over = (len(self.lines) - 1) - _BASH_MAX_LINES
        if over > 0:
            del self.lines[:over]
            self.dropped_lines += over
            start = max(0, start - over)
        for index in range(start, len(self.lines)):
            line = self.lines[index]
            if len(line) > _BASH_MAX_LINE_CHARS:
                self.lines[index] = "…" + line[-_BASH_MAX_LINE_CHARS:]

    @property
    def output(self) -> str:
        return "\n".join(self.lines)


@dataclass
class UserMessage:
    text: str = ""
    #: True from submission until the run actually starts (`agent_start`).
    pending: bool = True


@dataclass
class AssistantMessage:
    text: str = ""
    streaming: bool = True
    error: str = ""
    #: Set when a thinking block was shown because the run had no text yet, so
    #: the placeholder can be dropped as soon as real text streams in.
    thinking_placeholder: bool = False


@dataclass
class Notice:
    text: str = ""
    error: bool = False


@dataclass
class TMessage:
    kind: str  # "user" | "assistant" | "tool" | "bash" | "notice"
    user: Optional[UserMessage] = None
    assistant: Optional[AssistantMessage] = None
    tool: Optional[ToolExecution] = None
    bash: Optional[BashExecution] = None
    notice: Optional[Notice] = None


class Transcript:
    """The message list behind the scrollable area of the chat viewport.

    pi appends a user message to the transcript *before* the model runs and
    marks it pending until the run actually starts; karen does the same, so an
    interrupted prompt shows as "(interrupted)" instead of a bare prompt.

    The run's assistant block is tracked explicitly (`_run_assistant`) rather
    than by scanning: session notices (auto-retry, compaction) are appended
    *between* a failed run and its retry, so "the last message" is not the
    assistant block once a retry is announced.
    """

    def __init__(self) -> None:
        self.messages: List[TMessage] = []
        self._run_assistant: Optional[TMessage] = None
        self._pending_user: Optional[TMessage] = None

    # -- agent events (karen_agent.agent.AgentEvent) --------------------------

    def on_agent_start(self) -> None:
        """The run really started: the submitted prompt is no longer pending."""
        if self._pending_user is not None and self._pending_user.user is not None:
            self._pending_user.user.pending = False
            self._pending_user = None

    def on_user_message(self, text: str) -> TMessage:
        message = TMessage(kind="user", user=UserMessage(text=text))
        self.messages.append(message)
        self._pending_user = message
        # A new prompt starts a new assistant block.
        self._run_assistant = None
        return message

    def begin_assistant(self) -> AssistantMessage:
        """Start (or continue) the streaming assistant block of this run."""
        if self._run_assistant is not None and self._run_assistant.assistant is not None:
            assistant = self._run_assistant.assistant
            assistant.streaming = True
        else:
            assistant = AssistantMessage()
            message = TMessage(kind="assistant", assistant=assistant)
            self.messages.append(message)
            self._run_assistant = message
        return assistant

    def on_text_delta(self, delta: str) -> None:
        assistant = self.begin_assistant()
        if assistant.thinking_placeholder:
            # Real text arrived: drop the "[thinking] …" stand-in.
            assistant.text = ""
            assistant.thinking_placeholder = False
        assistant.text += delta

    def on_thinking_delta(self, delta: str) -> None:
        """Thinking deltas are not shown (pi shows expanded thinking blocks
        only on demand); nothing is recorded here."""

    def on_thinking_block_start(self, text: str) -> None:
        self.begin_assistant()

    def on_thinking_block_end(self, text: str) -> None:
        """Show the block dimmed inline, but only while the run has produced
        no visible text at all. Real text deltas replace the placeholder."""
        assistant = self.begin_assistant()
        if not assistant.text.strip() and text.strip():
            assistant.text = f"[thinking] {text}"
            assistant.thinking_placeholder = True

    def on_tool_execution_start(self, tool_call_id: str, tool_name: str, args: Any) -> None:
        self.messages.append(
            TMessage(
                kind="tool",
                tool=ToolExecution(name=tool_name, tool_call_id=tool_call_id, args=args),
            )
        )

    def on_tool_execution_update(self, tool_call_id: str, partial_result: Any) -> None:
        entry = self._find_tool(tool_call_id)
        if entry is not None and partial_result is not None:
            entry.args = partial_result

    def on_tool_execution_end(
        self, tool_call_id: str, tool_name: str, is_error: bool, result: Any
    ) -> None:
        # Match by call id: parallel tool executions of the same tool emit
        # interleaved end events, so a name-only match would attribute an
        # error to the wrong call.
        entry = self._find_tool(tool_call_id)
        if entry is None:
            return
        if is_error:
            text = "".join(
                getattr(block, "text", "")
                for block in (getattr(result, "content", None) or [])
                if hasattr(block, "text")
            )
            entry.state = "error"
            entry.error_text = text
        else:
            entry.state = "done"

    def _find_tool(self, tool_call_id: str) -> Optional[ToolExecution]:
        for message in reversed(self.messages):
            if message.kind != "tool" or message.tool is None:
                continue
            if tool_call_id and message.tool.tool_call_id == tool_call_id:
                return message.tool
        return None

    def finish_assistant(self) -> None:
        """The model finished producing this message (or the run stopped)."""
        if self._run_assistant is not None and self._run_assistant.assistant is not None:
            self._run_assistant.assistant.streaming = False

    def on_agent_end(self, final_message: Any, will_retry: bool = False) -> None:
        """Run finished (pi's `agent_end`).

        A run that is about to be retried (`will_retry`) is not a failure —
        pi skips reporting it and the same assistant block keeps streaming.
        """
        if will_retry:
            # Nothing is final: the retry continues this run's block, so its
            # streaming cursor and thinking state stay as they are.
            return
        self.finish_assistant()
        stop_reason = getattr(final_message, "stop_reason", None)
        if stop_reason == "error" and self._run_assistant is not None:
            self._run_assistant.assistant.error = (
                getattr(final_message, "error_message", "") or str(stop_reason)
            )
        # Whatever the outcome, the run is over: no user line may keep the
        # "(pending)" marker. A failed/aborted prompt additionally records why.
        for message in reversed(self.messages):
            if message.kind == "user" and message.user is not None:
                message.user.pending = False
                if stop_reason in ("error", "aborted"):
                    message.user.text = f"{message.user.text} ({stop_reason})"
                break

    def mark_user_message_sent(self, message: TMessage) -> None:
        """The prompt run really started: clear `pending` (pi's onCompletion)."""
        if message.user is not None:
            message.user.pending = False
        if message is self._pending_user:
            self._pending_user = None

    def resolve_pending(self) -> None:
        """A submitted line turned out to be a command, not a prompt: it will
        never become a run, so its line must not keep the "(pending)" marker."""
        if self._pending_user is not None and self._pending_user.user is not None:
            self._pending_user.user.pending = False
            self._pending_user = None

    def drop_pending_user(self, text: str) -> None:
        """Remove the just-submitted line the runner echoed.

        A `!command` run renders as a bash block, which already carries the
        command; echoing it as a user line as well would read as if the model
        had been asked to run it. Only the exact, still-pending line is
        dropped (identity match — two identical blocks are not the same one).
        """
        message = self._pending_user
        if message is None or message.user is None or message.user.text != text:
            return
        for index, candidate in enumerate(self.messages):
            if candidate is message:
                del self.messages[index]
                break
        self._pending_user = None

    # -- shell bypass (`!command`) ---------------------------------------------

    def on_bash_start(self, command: str) -> TMessage:
        """The `!command` was submitted: append its block and hand it back, so
        the caller can stream chunks into this exact message."""
        message = TMessage(kind="bash", bash=BashExecution(command=command))
        self.messages.append(message)
        return message

    def on_bash_chunk(self, message: TMessage, delta: str) -> None:
        if message.bash is not None and delta:
            message.bash.append(delta)

    def on_bash_end(
        self, message: TMessage, result: Any = None, error: Optional[str] = None
    ) -> None:
        """The command finished (`result`: a `BashResult`) or failed to run
        (`error`: the spawn/IO failure `execute_bash` raised)."""
        bash = message.bash
        if bash is None:
            return
        if error is not None:
            bash.state = "error"
            bash.error_text = error
            return
        bash.exit_code = getattr(result, "exit_code", None)
        bash.truncated = bool(getattr(result, "truncated", False))
        bash.full_output_path = getattr(result, "full_output_path", None)
        if getattr(result, "cancelled", False):
            bash.state = "cancelled"
        elif bash.exit_code not in (0, None):
            bash.state = "error"
        else:
            bash.state = "done"
        # Defensive: a command whose output never reached `on_chunk` (or whose
        # chunks were all dropped) still shows the tail the session recorded.
        if not bash.output and getattr(result, "output", ""):
            bash.append(result.output)

    # -- session events (the dict events `cli.py`'s `_on_session_event` eats) --

    def on_session_event(self, event: dict) -> None:
        event_type = event.get("type")
        if event_type == "session_opened":
            kind = "resumed" if event["resumed"] else event.get("reason", "new")
            self.add_notice(f"{kind} session {event['session_id']}")
        elif event_type == "compaction_start":
            self.add_notice(f"compacting ({event['reason']}; the model writes a summary)...")
        elif event_type == "compaction_end":
            if event["compacted"]:
                self.add_notice(f"compacted ~{event['tokens_before']} tokens")
            elif event.get("detail") == "nothing_to_compact":
                self.add_notice("nothing to compact")
            else:
                self.add_notice(f"compaction failed: {event.get('detail')}", error=True)
        elif event_type == "overflow_retry":
            self.add_notice("[context overflow: compacted; retrying the turn]")
        elif event_type == "overflow_give_up":
            self.add_notice(
                "[context overflow recovery failed after one compact-and-retry attempt]", error=True
            )
        elif event_type == "auto_retry_start":
            seconds = event["delayMs"] / 1000
            self.add_notice(
                f"[retrying (attempt {event['attempt']}/{event['maxAttempts']}) "
                f"in {seconds:.1f}s: {event['errorMessage']}]"
            )
        elif event_type == "auto_retry_end":
            if event["success"]:
                self.add_notice(f"[retry succeeded on attempt {event['attempt']}]")
            elif event.get("finalError") != "Retry cancelled":
                self.add_notice(f"[auto-retry gave up after {event['attempt']} attempt(s)]", error=True)
        elif event_type == "summarization_retry_scheduled":
            seconds = event["delayMs"] / 1000
            self.add_notice(
                f"[retrying summary (attempt {event['attempt']}/{event['maxAttempts']}) "
                f"in {seconds:.1f}s: {event['errorMessage']}]"
            )

    # -- REPL output -----------------------------------------------------------

    def add_notice(self, text: str, error: bool = False) -> None:
        self.messages.append(TMessage(kind="notice", notice=Notice(text=text, error=error)))

    def reset(self) -> None:
        self.messages = []
        self._run_assistant = None
        self._pending_user = None

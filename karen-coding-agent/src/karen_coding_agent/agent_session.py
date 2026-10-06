"""Application-level agent session for the karen coding agent.

Wires karen_agent's `Agent` to durable session persistence, automatic
compaction (threshold + overflow recovery), and the hook registry — the karen
equivalent of pi coding-agent's `core/agent-session.ts`, greatly simplified:
no settings manager, session projections, context edits, retries, or
extension events yet (those are later milestones).

Persistence model: every message is persisted as its `message_end` event
arrives (crash-safe, like pi). Overflow recovery rewinds the branch tip to
persistently omit the failed attempt (pi's `_omitRecoveryAttempt`) before the
recovery compaction; orphaned entries stay in the JSONL file but are
unreachable from the tip, exactly like branch navigation.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from karen_ai import Model, Models, SystemMessage
from karen_ai.utils.overflow import is_context_overflow, is_recoverable_length
from karen_ai.utils.text import get_system_message_text
from karen_agent import (
    AfterToolEvent,
    Agent,
    AgentInitialState,
    AgentMessage,
    AgentTool,
    BeforeCompactionEvent,
    BeforeToolCallResult,
    BeforeToolEvent,
    HookRegistry,
    StreamFn,
    convert_to_llm,
    models_stream_fn,
)
from karen_agent.compaction import (
    CompactionSettings,
    compact as _compact_impl,
    estimate_context_tokens,
    prepare_compaction,
    should_compact,
)
from karen_agent.messages import message_field
from karen_agent.result import Err
from karen_agent.session import (
    BranchScan,
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionRepo,
    MessageEntry,
    Session,
    UsageRow,
    branch_tip,
    insert_entry,
    insert_usage,
    set_value,
)
from karen_agent.session.context import build_session_context
from karen_agent.session.jsonl import to_jsonable
from karen_agent.session.types import CompactionEntry
from .tools import create_default_tools

DEFAULT_SESSIONS_ROOT = Path.home() / ".karen" / "sessions"
DEFAULT_BRANCH = "main"
DEFAULT_SYSTEM_PROMPT = """You are karen, an AI coding assistant.
The working directory is {cwd}; relative tool paths resolve against it.
Use the read/write/edit tools to inspect and modify files, bash{powershell} to run commands, and find/grep/ls to search.
Keep answers concise."""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _report_hook_error(error: BaseException, hook: str) -> None:
    print(f"\n[hook {hook} error: {error}]", file=sys.stderr)


#: Session lifecycle events forwarded to the optional `listener` callable:
#: {"type": "session_opened", "session_id": str, "resumed": bool}
#: {"type": "compaction_start", "reason": "manual" | "threshold" | "overflow"}
#: {"type": "compaction_end", "reason": ..., "compacted": bool, "detail"?: str, "tokens_before"?: int}
#: {"type": "overflow_retry"}   — overflow compacted; the turn is being retried
#: {"type": "overflow_give_up"} — recovery attempt exhausted; keeping the failure
SessionListener = Callable[[Dict[str, Any]], Any]


class AgentSession:
    """An `Agent` bound to a durable, resumable session on disk."""

    def __init__(
        self,
        *,
        cwd: str,
        models: Models,
        model: Model,
        sessions_root: Optional[str] = None,
        fresh: bool = False,
        branch_name: str = DEFAULT_BRANCH,
        system_prompt: Optional[str] = None,
        system_prompt_sections: Optional[Dict[str, str]] = None,
        tools: Optional[List[AgentTool]] = None,
        shell_path: Optional[str] = None,
        shell_command_prefix: Optional[str] = None,
        hooks: Optional[HookRegistry] = None,
        compaction_settings: Optional[CompactionSettings] = None,
        stream_fn: Optional[StreamFn] = None,
        listener: Optional[SessionListener] = None,
    ) -> None:
        self.cwd = cwd
        self.models = models
        self.model = model
        self.fresh = fresh
        self.branch_name = branch_name
        self.sessions_root = (
            sessions_root or os.environ.get("KAREN_SESSIONS_ROOT") or str(DEFAULT_SESSIONS_ROOT)
        )
        self.repo = JsonlSessionRepo(self.sessions_root)
        self.session: Optional[Session] = None
        self.tools = (
            tools
            if tools is not None
            else create_default_tools(cwd, shell_path=shell_path, shell_command_prefix=shell_command_prefix)
        )
        #: Structured prompt sections (pi's `SystemMessage.sections`), when the
        #: caller assembles the prompt with `karen_coding_agent.prompt`.
        self.system_prompt_sections = system_prompt_sections
        if system_prompt_sections is not None:
            self.system_prompt_text = get_system_message_text(
                SystemMessage(content="", sections=dict(system_prompt_sections), timestamp=0)
            )
        else:
            self.system_prompt_text = (
                system_prompt
                if system_prompt is not None
                else DEFAULT_SYSTEM_PROMPT.format(
                    cwd=cwd, powershell="/powershell" if sys.platform == "win32" else ""
                )
            )
        self.settings = compaction_settings or CompactionSettings()
        self.hooks = hooks if hooks is not None else HookRegistry(_report_hook_error)
        self._listener = listener
        self._stream_fn = stream_fn or models_stream_fn(models)
        self.agent = Agent(
            initial_state=AgentInitialState(
                system_prompt=self.system_prompt_text, model=model, tools=self.tools
            ),
            convert_to_llm=convert_to_llm,
            stream_fn=self._stream_fn,
            before_tool_call=self._before_tool_call,
            after_tool_call=self._after_tool_call,
        )
        self.agent.subscribe(self._on_agent_event)
        self._overflow_recovery_attempted = False
        self._run_entries: List[str] = []
        self._pre_run_tip: Optional[str] = None

    # -- lifecycle -------------------------------------------------------------

    def _emit(self, event: Dict[str, Any]) -> None:
        if self._listener is not None:
            self._listener(event)

    async def open(self) -> None:
        """Create or resume the on-disk session and rebuild the context."""
        resumed = False
        if not self.fresh:
            existing = await self.repo.list(JsonlSessionListOptions(cwd=self.cwd))
            if existing:
                self.session = await self.repo.open(existing[0])
                resumed = True
        if self.session is None:
            self.session = await self.repo.create(JsonlSessionCreateOptions(cwd=self.cwd))
            self.fresh = True
        if await self.session.branch(self.branch_name) is None:
            await self.session.create_branch(self.branch_name, None)
        await self._reload_context()
        self._emit(
            {"type": "session_opened", "session_id": self.session.metadata.id, "resumed": resumed}
        )

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None

    # -- context ---------------------------------------------------------------

    def _system_message(self) -> SystemMessage:
        # pi's initialState pattern: the leading system message carries prompt +
        # tool declarations, so resumed sessions replay tools without extra
        # system messages. Structured prompts carry their sections and leave
        # `content` empty, exactly like pi's transcript.
        if self.system_prompt_sections is not None:
            return SystemMessage(
                content="",
                sections=dict(self.system_prompt_sections),
                tools_added=self.tools,
                timestamp=_now_ms(),
            )
        return SystemMessage(
            content=self.system_prompt_text, tools_added=self.tools, timestamp=_now_ms()
        )

    async def _reload_context(self) -> None:
        branch = await self.session.branch(self.branch_name)
        entries = await branch.find_entries(BranchScan(order="oldestFirst"))
        session_messages = await build_session_context(entries)
        self.agent.state.messages = [self._system_message(), *session_messages]

    def estimate_tokens(self) -> int:
        return estimate_context_tokens(self.agent.state.messages).tokens

    def session_header(self) -> Optional[Dict[str, Any]]:
        """The JSONL session header dict, or None for non-JSONL storage.

        `karen --mode json` emits it as the first stdout line, like pi's
        `session.sessionManager.getHeader()`.
        """
        if self.session is None:
            return None
        header = getattr(self.session.storage, "header", None)
        if header is None:
            return None
        return to_jsonable(header)

    # -- agent passthrough -------------------------------------------------------

    def subscribe(self, listener):
        """Subscribe to agent events; returns an unsubscribe function."""
        return self.agent.subscribe(listener)

    def steer(self, message: AgentMessage) -> None:
        self.agent.steer(message)

    def follow_up(self, message: AgentMessage) -> None:
        self.agent.follow_up(message)

    def abort(self) -> None:
        self.agent.abort()

    async def wait_for_idle(self) -> None:
        await self.agent.wait_for_idle()

    def set_model(self, model: Model) -> None:
        """Switch models mid-session (pi's `AgentSession.setModel`).

        Applies to the next turn onwards: the loop reads `state.model` for
        each request, and compaction/overflow checks read `self.model`.
        """
        self.model = model
        self.agent.state.model = model

    # -- prompting ---------------------------------------------------------------

    async def prompt(self, text: str, images=None, auto_compact: bool = True) -> None:
        """Run one user prompt to completion, including overflow recovery and
        threshold auto-compaction (pi's post-run `_checkCompaction` driver;
        `auto_compact=False` skips the threshold check — the RPC mode's
        `set_auto_compaction` switch routes through here)."""
        self._overflow_recovery_attempted = False
        await self.agent.prompt(text, images)
        if auto_compact:
            await self._post_run()

    async def _post_run(self) -> None:
        while True:
            final = self._last_assistant_message()
            action = self._overflow_action(final)
            if action == "retry" and not self._overflow_recovery_attempted:
                self._overflow_recovery_attempted = True
                await self._omit_final_attempt()
                if await self.run_compaction("overflow"):
                    self._emit({"type": "overflow_retry"})
                    await self.agent.continue_()
                    continue
                return
            if action == "retry":
                # recovery already attempted: keep the failure and give up
                self._emit({"type": "overflow_give_up"})
                return
            if action == "compact_only":
                await self.run_compaction("overflow")
                return
            await self.maybe_auto_compact()
            return

    def _last_assistant_message(self) -> Optional[AgentMessage]:
        for message in reversed(self.agent.state.messages):
            if message_field(message, "role") == "assistant":
                return message
        return None

    def _overflow_action(self, message: Optional[AgentMessage]) -> Optional[str]:
        """The overflow branch of pi's `AgentSession._checkCompaction`:
        "retry" (compact + retry the turn), "compact_only" (compact but keep
        the completed response), or None.

        Simplifications: karen sessions are append-only, so pi's
        projection/context-edit retention guards are trivially true; and only
        fresh post-run messages are checked, so pi's stale pre-compaction
        guard cannot trigger.
        """
        if message is None or message_field(message, "role") != "assistant":
            return None
        if message_field(message, "stop_reason", "stopReason") == "aborted":
            return None
        # pi skips the check when the message came from a different model (the
        # user switched to a larger-context model after the overflow).
        if (
            message_field(message, "provider") != self.model.provider
            or message_field(message, "model") != self.model.id
        ):
            return None
        overflow = is_context_overflow(message, self.model.context_window or None)
        recoverable = is_recoverable_length(message, self.model.max_tokens or 0)
        if not (overflow or recoverable):
            return None
        # pi: willRetry = stopReason !== "stop" — agent.continue_() cannot
        # continue from a completed assistant response.
        return "compact_only" if message_field(message, "stop_reason", "stopReason") == "stop" else "retry"

    async def _omit_final_attempt(self) -> None:
        """Drop the final assistant attempt from the in-memory transcript AND
        rewind the persisted branch tip past its entries (pi's
        `_omitRecoveryAttempt`)."""
        messages = self.agent.state.messages
        index = len(messages) - 1
        while index > 0 and message_field(messages[index], "role") != "assistant":
            index -= 1
        omitted = len(messages) - index
        self.agent.state.messages = list(messages[:index])
        # the omitted messages were the last `omitted` entries persisted this run
        del self._run_entries[len(self._run_entries) - omitted :]
        new_tip = self._run_entries[-1] if self._run_entries else self._pre_run_tip

        async def rewind(mutator):
            await mutator.commit([set_value(branch_tip(self.branch_name), new_tip)])

        await self.session.mutate(rewind)

    # -- persistence --------------------------------------------------------------

    async def _on_agent_event(self, event, signal) -> None:
        event_type = getattr(event, "type", None)
        if event_type == "agent_start":
            self._run_entries = []
            branch = await self.session.branch(self.branch_name)
            self._pre_run_tip = await branch.get_tip_id()
        elif event_type == "message_end":
            await self._persist_message(event.message)

    async def _persist_message(self, message: AgentMessage) -> None:
        entry_id = self.session.id_generator.next()

        async def write_one(mutator):
            tip = await mutator.get_value(branch_tip(self.branch_name))
            writes = [
                insert_entry(MessageEntry(id=entry_id, parent_id=tip.value, message=message)),
                set_value(branch_tip(self.branch_name), entry_id),
            ]
            usage = message_field(message, "usage")
            if message_field(message, "role") == "assistant" and usage is not None:
                if message_field(usage, "total_tokens", "totalTokens", default=0):
                    writes.append(
                        insert_usage(
                            UsageRow(
                                id=self.session.id_generator.next(),
                                usage=usage,
                                entry_id=entry_id,
                                adjustment=False,
                            )
                        )
                    )
            await mutator.commit(writes)

        await self.session.mutate(write_one)
        self._run_entries.append(entry_id)

    # -- compaction --------------------------------------------------------------

    async def compact(self, custom_instructions: Optional[str] = None) -> bool:
        """Manual compaction (`/compact`)."""
        return await self.run_compaction("manual", custom_instructions=custom_instructions)

    async def maybe_auto_compact(self) -> None:
        tokens = self.estimate_tokens()
        window = self.model.context_window or 128000
        if should_compact(tokens, window, self.settings):
            await self.run_compaction("threshold")

    async def run_compaction(self, reason: str, custom_instructions: Optional[str] = None) -> bool:
        branch = await self.session.branch(self.branch_name)
        entries = await branch.find_entries(BranchScan(order="oldestFirst"))
        preparation_result = prepare_compaction(entries, self.settings)
        if isinstance(preparation_result, Err) or preparation_result.value is None:
            self._emit(
                {"type": "compaction_end", "reason": reason, "compacted": False, "detail": "nothing_to_compact"}
            )
            return False
        preparation = preparation_result.value
        hook_result = await self.hooks.run(
            "before_compaction",
            BeforeCompactionEvent(
                reason=reason, preparation=preparation, custom_instructions=custom_instructions
            ),
        )
        if hook_result is not None and hook_result.decline:
            self._emit({"type": "compaction_end", "reason": reason, "compacted": False, "detail": "declined"})
            return False
        self._emit({"type": "compaction_start", "reason": reason})
        result = await _compact_impl(
            preparation, self.models, self.model, custom_instructions=custom_instructions
        )
        if isinstance(result, Err):
            self._emit(
                {"type": "compaction_end", "reason": reason, "compacted": False, "detail": str(result.error)}
            )
            return False
        compacted = result.value
        compaction_id = self.session.id_generator.next()

        async def write_compaction(mutator):
            tip = await mutator.get_value(branch_tip(self.branch_name))
            await mutator.commit(
                [
                    insert_entry(
                        CompactionEntry(
                            id=compaction_id,
                            parent_id=tip.value,
                            summary=compacted.summary,
                            retained_tail=compacted.retained_tail,
                            tokens_before=compacted.tokens_before,
                            details=compacted.details,
                            usage=compacted.usage,
                            from_hook=False,
                        )
                    ),
                    set_value(branch_tip(self.branch_name), compaction_id),
                ]
            )

        await self.session.mutate(write_compaction)
        await self._reload_context()
        self._emit(
            {
                "type": "compaction_end",
                "reason": reason,
                "compacted": True,
                "tokens_before": compacted.tokens_before,
            }
        )
        return True

    # -- hooks bridge --------------------------------------------------------------

    async def _before_tool_call(self, hook_context, signal):
        result = await self.hooks.run(
            "before_tool",
            BeforeToolEvent(
                tool_call_id=hook_context.tool_call.id,
                tool_name=hook_context.tool_call.name,
                args=dict(hook_context.args or {}),
            ),
        )
        if result and result.block:
            return BeforeToolCallResult(
                block=True, reason=result.block.reason, terminate=result.block.terminate
            )
        return None

    async def _after_tool_call(self, hook_context, signal):
        await self.hooks.run(
            "after_tool",
            AfterToolEvent(
                tool_call_id=hook_context.tool_call.id,
                tool_name=hook_context.tool_call.name,
                args=dict(hook_context.args or {}),
                content=hook_context.result.content,
                details=hook_context.result.details,
                is_error=hook_context.is_error,
                usage=hook_context.result.usage,
            ),
        )
        return None

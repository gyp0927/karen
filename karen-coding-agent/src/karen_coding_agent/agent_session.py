"""Application-level agent session for the karen coding agent.

Wires karen_agent's `Agent` to durable session persistence, automatic
compaction (threshold + overflow recovery), auto-retry of transient provider
failures (M8), session navigation (tree, fork, clone, switch — M7), and the
hook registry — the karen equivalent of pi coding-agent's `core/agent-session.ts`,
greatly simplified: no settings manager, session projections, context edits, or
extension events yet (those are later milestones).

Persistence model: every message is persisted as its `message_end` event
arrives (crash-safe, like pi). Recovery rewinds the branch tip to persistently
omit the failed attempt (pi's `_omitRecoveryAttempt`) before the recovery
compaction or the retry's backoff; orphaned entries stay in the JSONL file but
are unreachable from the tip, exactly like branch navigation.
"""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from karen_ai import (
    AbortController,
    AbortError,
    Model,
    Models,
    RetryCallbacks,
    RetryPolicy,
    SystemMessage,
    TextContent,
    abortable_sleep,
)
from karen_ai.utils.overflow import is_context_overflow, is_recoverable_length
from karen_ai.utils.retry import is_retryable_assistant_error, retry_delay_ms
from karen_ai.utils.text import get_system_message_text
from karen_agent import (
    AfterToolCallResult,
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
    GenerateBranchSummaryOptions,
    compact as _compact_impl,
    collect_entries_for_branch_summary,
    estimate_context_tokens,
    generate_branch_summary,
    prepare_compaction,
    should_compact,
)
from karen_agent.messages import message_field
from karen_agent.result import Err
from karen_agent.session import (
    BranchForkOptions,
    BranchScan,
    BranchSummaryEntry,
    CustomEntry,
    Entry,
    EntryQuery,
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionMetadata,
    JsonlSessionRepo,
    LaneConfiguration,
    LaneModelRef,
    LaneState,
    MessageEntry,
    Session,
    UsageRow,
    Value,
    branch_tip,
    branch_tip_inventory_prefix,
    entry_label,
    insert_entry,
    insert_usage,
    lane_config,
    lane_state,
    set_value,
)
from karen_agent.session.context import build_session_context
from karen_agent.session.jsonl import to_jsonable
from karen_agent.session.types import CompactionEntry
from .navigation import TreeNode, build_tree, entry_text, forkable_user_messages
from .bash_executor import BashResult
from .settings import DEFAULT_RETRY_POLICY
from .tools import create_default_tools

#: pi-ai's full thinking-level ordering (karen-ai's `ModelThinkingLevel`).
EXTENDED_THINKING_LEVELS: List[str] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"]

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


def _add_usage(totals: Dict[str, float], usage) -> None:
    """Accumulate one `Usage` object into the running token/cost totals."""
    if usage is None:
        return
    for key in ("input", "output", "cache_read", "cache_write"):
        totals[key] += message_field(usage, key, default=0) or 0
    cost = message_field(usage, "cost")
    if cost is not None:
        totals["cost"] += message_field(cost, "total", default=0.0) or 0.0


#: Session lifecycle events forwarded to the optional `listener` callable:
#: {"type": "session_opened", "session_id": str, "resumed": bool,
#:  "reason"?: "new" | "fork" | "clone" | "switch"}
#: {"type": "compaction_start", "reason": "manual" | "threshold" | "overflow"}
#: {"type": "compaction_end", "reason": ..., "compacted": bool, "detail"?: str, "tokens_before"?: int}
#: {"type": "overflow_retry"}   — overflow compacted; the turn is being retried
#: {"type": "overflow_give_up"} — recovery attempt exhausted; keeping the failure
#: {"type": "auto_retry_start", "attempt": int, "maxAttempts": int, "delayMs": int,
#:  "errorMessage": str}       — a transient failure is about to be retried
#: {"type": "auto_retry_end", "success": bool, "attempt": int, "finalError"?: str}
#: {"type": "summarization_retry_scheduled", "attempt": int, "maxAttempts": int,
#:  "delayMs": int, "errorMessage": str} — same, for a compaction/branch summary call
#: {"type": "summarization_retry_attempt_start", "source": "compaction"|"branchSummary",
#:  "reason"?: "manual" | "threshold" | "overflow"}
#: {"type": "summarization_retry_finished"}
#: {"type": "session_tree", "new_leaf_id": str | None, "old_leaf_id": str | None,
#:  "summary_entry_id": str | None, "editor_text": str | None}
#: {"type": "session_info_changed", "name": str}
SessionListener = Callable[[Dict[str, Any]], Any]

#: pi's placeholder for an image stripped by `images.blockImages`.
BLOCKED_IMAGE_PLACEHOLDER = "Image reading is disabled."

#: Session entry custom type carrying a thinking-level change (pi's
#: `thinking_level_change` entry), so a resumed session comes back at the same
#: level (pi's `getSessionContextSettings`).
THINKING_LEVEL_ENTRY = "thinking_level_change"


def _thinking_level_value(entry: Any) -> Optional[str]:
    """The `thinkingLevel` payload of a `thinking_level_change` custom entry."""
    data = getattr(entry, "data", None)
    if not isinstance(data, dict):
        return None
    level = data.get("thinkingLevel")
    return level if isinstance(level, str) else None


def _block_message_images(message: Any) -> Any:
    """Replace image blocks in a converted user/toolResult message (pi's sdk.ts).

    pi keeps the exact placeholder text and drops a placeholder that directly
    follows another one, so a message with several images renders as one line.
    """
    if getattr(message, "role", None) not in ("user", "toolResult"):
        return message
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return message
    if not any(getattr(block, "type", None) == "image" for block in content):
        return message
    filtered: List[Any] = []
    for block in content:
        if getattr(block, "type", None) != "image":
            filtered.append(block)
            continue
        if (
            filtered
            and getattr(filtered[-1], "type", None) == "text"
            and getattr(filtered[-1], "text", None) == BLOCKED_IMAGE_PLACEHOLDER
        ):
            continue
        filtered.append(TextContent(text=BLOCKED_IMAGE_PLACEHOLDER))
    return message.model_copy(update={"content": filtered})


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
        retry_policy: Optional[RetryPolicy] = None,
        auto_resize_images: bool = True,
        block_images: bool = False,
        stream_fn: Optional[StreamFn] = None,
        listener: Optional[SessionListener] = None,
        mcp_manager: Optional[Any] = None,
    ) -> None:
        self.cwd = cwd
        self.models = models
        self.shell_path = shell_path
        self.shell_command_prefix = shell_command_prefix
        #: pi's `settings.images.autoResize` (default true) — resize inline
        #: images to the model's limits on the prompt and tool-result paths.
        self.auto_resize_images = auto_resize_images
        #: pi's `settings.images.blockImages` (default false) — strip images
        #: from the LLM request in a `convert_to_llm` wrapper (defense-in-depth,
        #: pi's `convertToLlmWithBlockImages`). Read on every call so a
        #: mid-session change takes effect.
        self.block_images = block_images
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
            else create_default_tools(
                cwd,
                shell_path=shell_path,
                shell_command_prefix=shell_command_prefix,
                auto_resize_images=auto_resize_images,
                # Resolved per read, so `set_model` picks up the new profile.
                image_resize_options=lambda: self._image_resize_options(),
            )
        )
        # MCP tools (from a connected `McpToolManager`) extend the default set.
        self.mcp_manager = mcp_manager
        if mcp_manager is not None:
            self.tools = [*self.tools, *mcp_manager.tools()]
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
        #: Assistant-call retry policy (pi's `settings.retry`); toggled live by
        #: `set_auto_retry_enabled`.
        self.retry = retry_policy if retry_policy is not None else DEFAULT_RETRY_POLICY
        self.hooks = hooks if hooks is not None else HookRegistry(_report_hook_error)
        self._listener = listener
        self._stream_fn = stream_fn or models_stream_fn(models)
        self.agent = Agent(
            initial_state=AgentInitialState(
                system_prompt=self.system_prompt_text, model=model, tools=self.tools
            ),
            convert_to_llm=self._convert_to_llm,
            stream_fn=self._stream_fn,
            before_tool_call=self._before_tool_call,
            after_tool_call=self._after_tool_call,
        )
        self.agent.subscribe(self._on_agent_event)
        self._overflow_recovery_attempted = False
        #: Everything appended to the branch during the current run, in append
        #: order, as `(kind, entry_id, entry)` — `entry` is the `CustomEntry`
        #: itself for `"custom"` appends (the rewind has to re-parent it) and
        #: None for `"message"` ones. Because every append moves the branch tip
        #: to the new entry, this list mirrors the chain the run built, which is
        #: what `_omit_final_attempt` has to undo — and only partially.
        self._run_appends: List[Tuple[str, str, Any]] = []
        self._pre_run_tip: Optional[str] = None
        self._retry_attempt = 0
        self._retry_controller: Optional[AbortController] = None
        self._abort_requested = False
        self._bash_controllers: List[AbortController] = []
        #: Side-channel bash results recorded while streaming, flushed when the
        #: run settles and again before a new prompt (pi's `_pendingBashMessages`).
        self._pending_bash_messages: List[Any] = []

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
        await self._ensure_branch(self.session)
        await self._reload_context()
        await self._restore_thinking_level()
        self._emit(
            {"type": "session_opened", "session_id": self.session.metadata.id, "resumed": resumed}
        )

    async def _thinking_level_entries(self) -> List[Any]:
        branch = await self.session.branch(self.branch_name)
        entries = await branch.find_entries(BranchScan(order="oldestFirst"))
        return [
            entry
            for entry in entries
            if getattr(entry, "type", None) == "custom"
            and getattr(entry, "custom_type", None) == THINKING_LEVEL_ENTRY
        ]

    async def _restore_thinking_level(self) -> None:
        """Restore the branch's last `thinking_level_change` (pi's `getSessionContextSettings`).

        Deviation: pi also seeds a `thinking_level_change` entry into every new
        session, and backfills one into sessions that predate the entry type,
        from the settings manager's `defaultThinkingLevel`. karen has no such
        setting — a session starts at the level the caller passed — so a seed
        entry would carry no information while putting a metadata node at the
        root of every tree. karen records actual changes only, and restores
        them the same way pi does, from the last entry on the branch path.
        """
        from karen_ai import clamp_thinking_level

        entries = await self._thinking_level_entries()
        if not entries:
            return
        level = _thinking_level_value(entries[-1])
        if not isinstance(level, str):
            return
        available = self.get_available_thinking_levels()
        self.agent.state.thinking_level = (
            level if level in available else (clamp_thinking_level(self.model, level) if self.model else "off")
        )

    async def _ensure_branch(self, session: Session) -> None:
        """Make sure the branch exists and is a configured lane.

        karen-agent's fork only copies *configured* lanes (`pi.lane.config` +
        `pi.lane.state`); pi's runtime writes the same pair when it binds a
        session to a model. Existing values are left alone, so an old session
        gains forkability without losing its recorded model.
        """
        if await session.branch(self.branch_name) is None:
            await session.create_branch(self.branch_name, None)
        if await session.get_value(lane_config(self.branch_name)) is None:
            configuration = LaneConfiguration(
                model=LaneModelRef(provider=self.model.provider, model_id=self.model.id),
                thinking_level=self.agent.state.thinking_level,
                active_tool_names=[tool.name for tool in self.tools],
            )
            await session.set_value(lane_config(self.branch_name), configuration.model_dump(mode="json", by_alias=True))
        if await session.get_value(lane_state(self.branch_name)) is None:
            await session.set_value(
                lane_state(self.branch_name), LaneState().model_dump(mode="json", by_alias=True)
            )

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None
        if self.mcp_manager is not None:
            await self.mcp_manager.aclose()

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

    # -- session info ------------------------------------------------------------

    async def session_name(self) -> Optional[str]:
        """The session's display name (pi's `getSessionName`)."""
        return await self.session.get_name()

    async def set_session_name(self, name: str) -> None:
        """Name the session (pi's `setSessionName`); empty names are rejected."""
        name = name.strip()
        if not name:
            raise ValueError("Session name cannot be empty")
        await self.session.set_name(name)
        self._emit({"type": "session_info_changed", "name": name})

    async def list_sessions(self) -> List[JsonlSessionMetadata]:
        """Sessions on disk for this cwd, newest first (pi's resume list)."""
        return await self.repo.list(JsonlSessionListOptions(cwd=self.cwd))

    async def entries(self) -> List[Entry]:
        """Every entry in the session file, oldest first (pi's `getEntries`)."""
        return await self.session.find_entries(EntryQuery(order="asc"))

    async def entry_labels(self) -> Dict[str, str]:
        """Label per entry id (pi's `pi.entry.label` values)."""
        stored = await self.session.scan_values(Value(namespace="pi.entry.label"))
        return {item.address.key: item.value for item in stored}

    async def session_tree(self) -> List[TreeNode]:
        """The session tree roots (pi's `getTree`)."""
        return build_tree(await self.entries(), await self.entry_labels())

    async def branch_tip_id(self) -> Optional[str]:
        """The current branch's tip (`leafId` in pi's protocol)."""
        branch = await self.session.branch(self.branch_name)
        return await branch.get_tip_id()

    async def branch_tips(self) -> List[str]:
        """Tip ids of every branch in the session (the tree marks them `○`)."""
        stored = await self.session.scan_values(branch_tip_inventory_prefix())
        return [item.value for item in stored if item.value]

    async def session_stats(self) -> Dict[str, Any]:
        """pi's `getSessionStats`: message breakdown plus token/cost totals.

        Tokens and cost are summed from the entries themselves — assistant and
        tool-result usage, plus the usage carried by compaction and
        branch-summary entries, exactly the set pi sums. (karen's stored usage
        rows hold the same assistant totals, but a fork copies entries and not
        the usage rows, so entries are the durable source.) pi's `contextUsage`
        field is not ported.
        """
        totals: Dict[str, float] = {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "cost": 0.0,
        }
        user_messages = assistant_messages = tool_calls = tool_results = total_messages = 0
        for entry in await self.entries():
            entry_type = getattr(entry, "type", None)
            if entry_type in ("compaction", "branch_summary"):
                _add_usage(totals, getattr(entry, "usage", None))
            if entry_type != "message":
                continue
            total_messages += 1
            message = entry.message
            role = message_field(message, "role")
            if role == "user":
                user_messages += 1
            elif role == "assistant":
                assistant_messages += 1
                for block in message_field(message, "content", default=[]) or []:
                    if message_field(block, "type") == "toolCall":
                        tool_calls += 1
                _add_usage(totals, message_field(message, "usage"))
            elif role == "toolResult":
                tool_results += 1
                _add_usage(totals, message_field(message, "usage"))
        metadata = self.session.metadata
        return {
            "sessionFile": getattr(metadata, "path", None),
            "sessionId": metadata.id,
            "userMessages": user_messages,
            "assistantMessages": assistant_messages,
            "toolCalls": tool_calls,
            "toolResults": tool_results,
            "totalMessages": total_messages,
            "tokens": {
                "input": totals["input"],
                "output": totals["output"],
                "cacheRead": totals["cache_read"],
                "cacheWrite": totals["cache_write"],
                "total": totals["input"] + totals["output"] + totals["cache_read"] + totals["cache_write"],
            },
            "cost": totals["cost"],
        }

    # -- export (M9) --------------------------------------------------------------

    async def export_to_jsonl(self, output_path: Optional[str] = None) -> str:
        """Export the current branch as pi-importable JSONL (pi's `exportSessionToJsonl`).

        The branch is re-parented into one linear chain under a v3
        `{"type":"session",...}` header so pi can re-open the file. Returns the
        written path.
        """
        from .session_export import export_session_to_jsonl

        branch = await self.session.branch(self.branch_name)
        entries = await branch.find_entries(BranchScan(order="oldestFirst"))
        return export_session_to_jsonl(
            self.session.metadata.id,
            self.cwd,
            [to_jsonable(entry) for entry in entries],
            output_path=output_path,
        )

    async def export_to_html(self, output_path: Optional[str] = None) -> str:
        """Export the session as a self-contained HTML report (pi's `exportToHtml`)."""
        from .session_export import export_session_to_html

        header = self.session_header() or {}
        entries = [to_jsonable(entry) for entry in await self.entries()]
        leaf_id = await self.branch_tip_id()
        return export_session_to_html(
            header,
            entries,
            leaf_id,
            output_path=output_path,
            cwd=self.cwd,
            system_prompt=self.system_prompt_text,
        )

    # -- navigation --------------------------------------------------------------

    async def user_messages_for_forking(self) -> List[Dict[str, str]]:
        """pi's `getUserMessagesForForking`: the fork-selector menu.

        Deviation: only user messages on the current branch are listed. karen's
        fork copies the branch path (pi's `createBranchedSession` can copy any
        entry's ancestry), so a message from another branch is not yet a valid
        target — navigate to that branch first.
        """
        branch = await self.session.branch(self.branch_name)
        entries = await branch.find_entries(BranchScan(order="oldestFirst"))
        return [
            {"entryId": entry_id, "text": text}
            for entry_id, text in forkable_user_messages(entries)
        ]

    async def navigate_tree(
        self,
        target_id: str,
        *,
        summarize: bool = False,
        custom_instructions: Optional[str] = None,
        replace_instructions: Optional[bool] = None,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Move the branch tip to another entry (pi's `navigateTree`).

        Staying in the same session file, the tip moves to `target_id` — or, for
        a user message, to its parent, with the message text returned as
        `editorText` so the caller can put it back in the editor. With
        `summarize`, the abandoned branch (old tip back to the common ancestor)
        is condensed into a `branch_summary` entry placed at the new position.
        """
        if self.agent.state.is_streaming:
            raise RuntimeError("Wait for the current response to finish before navigating the session tree.")
        old_leaf = await self.branch_tip_id()
        if target_id == old_leaf:
            return {"cancelled": False, "editorText": None, "summaryEntryId": None}
        target = await self.session.get_entry(target_id)
        if target is None:
            raise ValueError(f"Entry {target_id} not found")

        summary_text: Optional[str] = None
        summary_details: Optional[Dict[str, Any]] = None
        summary_usage = None
        if summarize:
            branch = await self.session.branch(self.branch_name)
            collected = await collect_entries_for_branch_summary(branch, self.session, old_leaf, target_id)
            if collected.entries:
                options = GenerateBranchSummaryOptions(
                    models=self.models,
                    model=self.model,
                    custom_instructions=custom_instructions,
                    replace_instructions=replace_instructions,
                    retry=self.retry,
                    callbacks=self._summarization_retry_callbacks({"source": "branchSummary"}),
                )
                result = await generate_branch_summary(collected.entries, options)
                if isinstance(result, Err):
                    raise RuntimeError(str(result.error))
                summary_text = result.value.summary
                summary_usage = result.value.usage
                summary_details = {
                    "readFiles": list(result.value.read_files),
                    "modifiedFiles": list(result.value.modified_files),
                }

        # pi: a user message navigates to its parent and hands the text back
        editor_text: Optional[str] = None
        if target.type == "message" and message_field(target.message, "role") == "user":
            new_leaf = target.parent_id
            editor_text = entry_text(target)
        else:
            new_leaf = target_id

        summary_entry_id: Optional[str] = None
        labeled_entry = target_id
        if summary_text is not None:
            summary_entry_id = self.session.id_generator.next()
            labeled_entry = summary_entry_id

        async def apply(mutator) -> None:
            writes = []
            leaf = new_leaf
            if summary_text is not None:
                writes.append(
                    insert_entry(
                        BranchSummaryEntry(
                            id=summary_entry_id,
                            parent_id=new_leaf,
                            summary=summary_text,
                            details=summary_details,
                            usage=summary_usage,
                            from_hook=False,
                        )
                    )
                )
                leaf = summary_entry_id
            writes.append(set_value(branch_tip(self.branch_name), leaf))
            if label:
                writes.append(set_value(entry_label(labeled_entry), label))
            await mutator.commit(writes)

        await self.session.mutate(apply)
        await self._reload_context()
        self._emit(
            {
                "type": "session_tree",
                "new_leaf_id": await self.branch_tip_id(),
                "old_leaf_id": old_leaf,
                "summary_entry_id": summary_entry_id,
                "editor_text": editor_text,
            }
        )
        return {"cancelled": False, "editorText": editor_text, "summaryEntryId": summary_entry_id}

    async def fork(self, entry_id: Optional[str] = None, *, position: str = "before") -> Dict[str, Any]:
        """Copy this branch into a new session file and switch to it.

        `position="before"` (pi's fork default) requires `entry_id` to be a user
        message on the current branch: everything up to its parent is copied and
        the message text is returned as `selectedText`, ready to edit.
        `position="at"` copies through the selected entry; with no `entry_id` it
        copies the whole branch (pi's `clone`).
        """
        if position not in ("before", "at"):
            raise ValueError(f"Invalid fork position: {position!r} (expected 'before' or 'at')")
        if self.agent.state.is_streaming:
            raise RuntimeError("Wait for the current response to finish before forking the session.")
        selected_text: Optional[str] = None
        if position == "before":
            entry = await self.session.get_entry(entry_id) if entry_id else None
            if entry is None or entry.type != "message" or message_field(entry.message, "role") != "user":
                raise ValueError("Invalid entry ID for forking")
            selected_text = entry_text(entry)
        forked = await self.repo.fork(
            self.session.metadata,
            BranchForkOptions(branch=self.branch_name, entry_id=entry_id, position=position),
        )
        await self._rebind(forked, reason="fork")
        return {"cancelled": False, "selectedText": selected_text}

    async def clone(self) -> Dict[str, Any]:
        """Copy the current branch into a new session file (pi's `clone`)."""
        if await self.branch_tip_id() is None:
            raise ValueError("Cannot clone session: no current entry selected")
        return await self.fork(None, position="at")

    async def switch_session(self, metadata: JsonlSessionMetadata) -> None:
        """Open another on-disk session for this cwd and switch to it."""
        if self.agent.state.is_streaming:
            raise RuntimeError("Wait for the current response to finish before switching sessions.")
        if metadata.id == self.session.metadata.id:
            return  # already there: reopening an open session file is not allowed
        await self._rebind(await self.repo.open(metadata), reason="switch")

    async def set_label(self, entry_id: str, label: Optional[str]) -> None:
        """Attach or clear a tree label on an entry (pi's `appendLabelChange`)."""
        if await self.session.get_entry(entry_id) is None:
            raise ValueError(f"Entry {entry_id} not found")

        async def apply(mutator) -> None:
            await mutator.commit([set_value(entry_label(entry_id), label)])

        await self.session.mutate(apply)

    async def _rebind(self, session: Session, *, reason: str = "new") -> None:
        """Adopt a different session handle (fork/clone/switch) and reload context."""
        previous = self.session
        self.session = session
        if previous is not None and previous is not session:
            await previous.close()
        await self._ensure_branch(session)
        await self._reload_context()
        # The level belongs to the session, not the runtime (pi rebuilds its
        # AgentSession on every switch and restores the level there), so a
        # switch/fork/clone must re-read the new branch's last change.
        await self._restore_thinking_level()
        branch = await session.branch(self.branch_name)
        self._pre_run_tip = await branch.get_tip_id()
        self._run_appends = []
        self._emit(
            {
                "type": "session_opened",
                "session_id": session.metadata.id,
                "resumed": False,
                "reason": reason,
            }
        )

    # -- agent passthrough -------------------------------------------------------

    def subscribe(self, listener):
        """Subscribe to agent events; returns an unsubscribe function.

        The forwarded `agent_end` carries `willRetry` (pi sets the same flag
        before dispatching it), so consumers can tell a run that failed and is
        about to be retried from one that is really over.
        """

        def forwarded(event, signal):
            if getattr(event, "type", None) == "agent_end":
                event = event.model_copy(
                    update={"will_retry": self.will_retry_after_agent_end(event)}
                )
            return listener(event, signal)

        return self.agent.subscribe(forwarded)

    def steer(self, message: AgentMessage) -> None:
        self.agent.steer(message)

    def follow_up(self, message: AgentMessage) -> None:
        self.agent.follow_up(message)

    def abort(self) -> None:
        self._abort_requested = True
        self.abort_retry()
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

    # -- thinking level ---------------------------------------------------------

    def supports_thinking(self) -> bool:
        """Whether the current model supports thinking/reasoning (pi's `supportsThinking`)."""
        return bool(getattr(self.model, "reasoning", False))

    def get_available_thinking_levels(self) -> List[str]:
        """Thinking levels the current model supports (pi's `getAvailableThinkingLevels`)."""
        from karen_ai import get_supported_thinking_levels

        if not self.model:
            return list(EXTENDED_THINKING_LEVELS)
        return list(get_supported_thinking_levels(self.model))

    async def set_thinking_level(self, level: str) -> None:
        """Set the thinking level, clamped to the model's capabilities (pi's `setThinkingLevel`).

        Records a `thinking_level_change` session entry and emits
        `thinking_level_changed`, both only when the level actually changes.
        """
        from karen_ai import clamp_thinking_level

        available = self.get_available_thinking_levels()
        effective = level if level in available else (clamp_thinking_level(self.model, level) if self.model else "off")
        previous = self.agent.state.thinking_level
        self.agent.state.thinking_level = effective
        if effective != previous:
            await self._append_custom_entry(THINKING_LEVEL_ENTRY, {"thinkingLevel": effective})
            self._emit({"type": "thinking_level_changed", "level": effective})

    async def cycle_thinking_level(self) -> Optional[str]:
        """Cycle to the next supported thinking level (pi's `cycleThinkingLevel`).

        Returns the new level, or None when the model doesn't support thinking.
        """
        if not self.supports_thinking():
            return None
        levels = self.get_available_thinking_levels()
        try:
            index = levels.index(self.agent.state.thinking_level)
        except ValueError:
            index = -1
        next_level = levels[(index + 1) % len(levels)]
        await self.set_thinking_level(next_level)
        return next_level

    # -- bash (RPC side channel) -------------------------------------------------

    async def execute_bash(
        self,
        command: str,
        on_chunk: Optional[Callable[[str], None]] = None,
        *,
        exclude_from_context: bool = False,
        command_id: Optional[str] = None,
    ) -> "BashResult":
        """Run a shell command outside the agent loop (pi's `AgentSession.executeBash`).

        Applies the configured shell path and command prefix, streams sanitized
        output to `on_chunk` and `bash_execution_update` events, records the
        result as a `BashExecutionMessage` in the session (pi's
        `recordBashResult`, so the model sees it next turn; `exclude_from_context`
        keeps it out of the LLM conversion), and reports pi's `BashResult`.
        `abort_bash()` cancels it.
        """
        from .bash_executor import execute_bash

        controller = AbortController()
        self._bash_controllers.append(controller)

        def _on_chunk(delta: str) -> None:
            if on_chunk is not None:
                on_chunk(delta)
            self._emit({"type": "bash_execution_update", "id": command_id, "delta": delta})

        try:
            result = await execute_bash(
                command,
                self.cwd,
                shell_path=self.shell_path,
                shell_command_prefix=self.shell_command_prefix,
                on_chunk=_on_chunk,
                signal=controller.signal,
            )
            await self.record_bash_result(command, result, exclude_from_context=exclude_from_context)
            return result
        finally:
            if controller in self._bash_controllers:
                self._bash_controllers.remove(controller)

    async def record_bash_result(
        self, command: str, result: "BashResult", *, exclude_from_context: bool = False
    ) -> None:
        """Record a side-channel bash run in the transcript (pi's `recordBashResult`).

        The message goes into the session so the model sees the command and its
        output on the next turn; while the agent is streaming it is queued
        instead, so appending it can't break the tool_use/tool_result ordering.
        """
        from karen_agent.messages import BashExecutionMessage

        message = BashExecutionMessage(
            command=command,
            output=result.output,
            exit_code=result.exit_code,
            cancelled=result.cancelled,
            truncated=result.truncated,
            full_output_path=result.full_output_path,
            timestamp=int(time.time() * 1000),
            exclude_from_context=exclude_from_context or None,
        )
        if getattr(self.agent.state, "is_streaming", False):
            self._pending_bash_messages.append(message)
            return
        await self._persist_message(message)
        self.agent.state.messages.append(message)

    def abort_bash(self) -> None:
        """Cancel any in-flight `execute_bash` (pi's `AgentSession.abortBash`)."""
        for controller in self._bash_controllers:
            controller.abort()
        self._bash_controllers.clear()

    # -- prompting ---------------------------------------------------------------

    async def prompt(self, text: str, images=None, auto_compact: bool = True) -> None:
        """Run one user prompt to completion, including auto-retry of transient
        provider failures, overflow recovery and threshold auto-compaction (pi's
        post-run `_handlePostAgentRun` driver; `auto_compact=False` skips the
        compaction and overflow branches — the RPC mode's `set_auto_compaction`
        switch routes through here — while auto-retry keeps running, like pi).

        `images` are normalized first (pi's `_normalizePromptImages`): each is
        converted/resized to the model's inline limits, failures are dropped and
        their messages appended to the text as hints. `steer()`/`follow_up()`
        keep images raw, exactly like pi.
        """
        self._overflow_recovery_attempted = False
        self._abort_requested = False
        # pi's second flush point (the first is the run-settle flush in the
        # `finally` below): a bash result recorded in the window after that
        # flush but before the run fully settled is still queued here, and the
        # model has to see it in *this* turn rather than at the end of the next
        # one. pi skips this on the streaming path (a steer/followUp returns
        # before the flush), so a mid-run call leaves the queue alone.
        if not getattr(self.agent.state, "is_streaming", False):
            await self._flush_pending_bash_messages()
        normalized_images, hints = await self._normalize_prompt_images(images)
        if hints:
            text = f"{text}\n\n" + "\n".join(hints)
        try:
            await self.agent.prompt(text, normalized_images)
            await self._post_run(auto_compact)
        finally:
            # pi flushes pending side-channel results in the `finally` of
            # `_runAgentPrompt` — once the whole run is over, including
            # auto-retry and overflow recovery. Flushing at each attempt's
            # `agent_end` instead would append the result *behind* the failed
            # attempt that `_omit_final_attempt` then casts away, so a retried
            # turn would silently lose it from the transcript and the branch.
            # A run still in flight (a nested `prompt()` returns through here
            # after the agent refused to re-enter) keeps the queue for its own
            # settle flush rather than having a message spliced into it.
            if not getattr(self.agent.state, "is_streaming", False):
                await self._flush_pending_bash_messages()

    @staticmethod
    def _resize_options_for(model):
        """A model's image resize profile (pi's `model.inputLimits.images.resize`)."""
        input_limits = getattr(model, "input_limits", None)
        images = getattr(input_limits, "images", None) if input_limits is not None else None
        return getattr(images, "resize", None) if images is not None else None

    def _convert_to_llm(self, messages: List[AgentMessage]) -> Any:
        """`convert_to_llm` + pi's `images.blockImages` filter (pi's sdk.ts wrapper).

        The setting is read on every call, so toggling it mid-session applies to
        the next request without rebuilding the agent.
        """
        converted = convert_to_llm(messages)
        if not self.block_images:
            return converted
        return [_block_message_images(message) for message in converted]

    def _image_resize_options(self):
        """The current model's image resize profile (pi's `model.inputLimits.images.resize`)."""
        return self._resize_options_for(self.model)

    async def _normalize_prompt_images(self, images):
        """pi's `_normalizePromptImages`: process each image, drop failures into hints."""
        if not images:
            return images, []
        from karen_ai import ImageContent

        from .utils import process_image

        resize_options = self._image_resize_options()
        normalized: List[Any] = []
        hints: List[str] = []
        for image in images:
            processed = await process_image(
                base64.b64decode(image.data),
                image.mime_type,
                auto_resize_images=self.auto_resize_images,
                resize_options=resize_options,
            )
            if not processed.ok:
                hints.append(processed.message)
                continue
            normalized.append(ImageContent(data=processed.data, mime_type=processed.mime_type))
            hints.extend(processed.hints)
        return normalized, hints

    async def _post_run(self, auto_compact: bool = True) -> None:
        while True:
            final = self._last_assistant_message()
            if await self._retry_or_keep_failure(final):
                await self.agent.continue_()
                continue
            if not auto_compact:
                return
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
        `_omitRecoveryAttempt`).

        pi edits the failed message out of the *context* and leaves every
        session entry parented where it was; karen's storage has no context-edit
        entry, so it rewinds the branch tip instead. That is only equivalent as
        long as the rewind does not take unrelated entries off the branch: a
        recovery can be triggered while the run has already recorded something
        else — a thinking-level change, a label, a model switch — and those
        entries must survive (a lost `thinking_level_change` silently reverts
        the level on the next resume).
        """
        messages = self.agent.state.messages
        index = len(messages) - 1
        while index > 0 and message_field(messages[index], "role") != "assistant":
            index -= 1
        omitted = len(messages) - index
        self.agent.state.messages = list(messages[:index])
        # Each append made its entry the branch tip, so the list mirrors the
        # chain the run built and the failed attempt's entries are its last
        # `omitted` message appends.
        message_appends = [
            position
            for position, (kind, _id, _entry) in enumerate(self._run_appends)
            if kind == "message"
        ]
        cut_index = len(message_appends) - omitted
        cut = message_appends[cut_index] if cut_index >= 0 else 0
        kept, dropped = self._run_appends[:cut], self._run_appends[cut:]
        self._run_appends = kept
        new_tip = kept[-1][1] if kept else self._pre_run_tip

        async def rewind(mutator):
            await mutator.commit([set_value(branch_tip(self.branch_name), new_tip)])

        await self.session.mutate(rewind)
        # Everything after the new tip that is not part of the failed attempt
        # (i.e. every custom entry the rewind just took off the branch) goes
        # back on, in order, as a child of the tip we rewound to.
        for kind, _id, entry in dropped:
            if kind == "custom":
                await self._append_custom_entry(entry.custom_type, entry.data)

    # -- auto-retry ---------------------------------------------------------------

    async def _retry_or_keep_failure(self, message: Optional[AgentMessage]) -> bool:
        """pi's `_handlePostAgentRun` retry branch: schedule a retry for a
        transiently failed turn (returns True when the caller should continue
        the agent), otherwise report an exhausted retry budget."""
        if self._is_retryable_error(message) and await self._prepare_retry(message):
            return True
        if (
            message is not None
            and message_field(message, "stop_reason", "stopReason") == "error"
            and self._retry_attempt > 0
        ):
            attempt = self._retry_attempt
            self._retry_attempt = 0
            self._emit(
                {
                    "type": "auto_retry_end",
                    "success": False,
                    "attempt": attempt,
                    "finalError": message_field(message, "error_message", "errorMessage"),
                }
            )
        return False

    def _is_retryable_error(self, message: Optional[AgentMessage]) -> bool:
        """Whether a failed assistant message is worth retrying. Context
        overflow is handled by compaction instead (pi's `_isRetryableError`)."""
        if message is None:
            return False
        if is_context_overflow(message, self.model.context_window or None):
            return False
        return is_retryable_assistant_error(message)

    async def _prepare_retry(self, message: AgentMessage) -> bool:
        """Back off, omit the failed attempt, and let the caller re-run the turn.

        Returns False when auto-retry is disabled, the budget is exhausted, or
        the backoff was cancelled — the failure is then kept as the run's result.
        """
        if not self.retry.enabled:
            return False
        self._retry_attempt += 1
        if self._retry_attempt > self.retry.max_retries:
            # Preserve the completed attempt count so post-run handling can emit the final failure.
            self._retry_attempt -= 1
            return False
        delay_ms = retry_delay_ms(self.retry, self._retry_attempt)
        self._emit(
            {
                "type": "auto_retry_start",
                "attempt": self._retry_attempt,
                "maxAttempts": self.retry.max_retries,
                "delayMs": delay_ms,
                "errorMessage": message_field(message, "error_message", "errorMessage") or "Unknown error",
            }
        )
        # Keep the failed attempt in raw history while durably omitting it from model projection.
        await self._omit_final_attempt()
        self._retry_controller = AbortController()
        try:
            await abortable_sleep(delay_ms / 1000, self._retry_controller.signal)
        except AbortError:
            # Aborted during the backoff: emit the end event so listeners can clean up.
            self._finish_cancelled_retry()
            return False
        finally:
            self._retry_controller = None
        return True

    def _finish_cancelled_retry(self) -> None:
        if self._retry_attempt == 0:
            return
        attempt = self._retry_attempt
        self._retry_attempt = 0
        self._emit(
            {
                "type": "auto_retry_end",
                "success": False,
                "attempt": attempt,
                "finalError": "Retry cancelled",
            }
        )

    def abort_retry(self) -> None:
        """Cancel an in-progress retry backoff (pi's `abortRetry`)."""
        if self._retry_controller is not None:
            self._retry_controller.abort()

    @property
    def is_retrying(self) -> bool:
        """Whether an auto-retry is currently in progress."""
        return self._retry_controller is not None

    @property
    def auto_retry_enabled(self) -> bool:
        return self.retry.enabled

    def set_auto_retry_enabled(self, enabled: bool) -> None:
        self.retry.enabled = enabled

    def will_retry_after_agent_end(self, event) -> bool:
        """pi's `_willRetryAfterAgentEnd`: whether the run that just ended is
        about to be retried, for the `willRetry` flag on the forwarded event."""
        if (
            self._abort_requested
            or not self.retry.enabled
            or self._retry_attempt >= self.retry.max_retries
        ):
            return False
        for message in reversed(getattr(event, "messages", None) or []):
            if message_field(message, "role") == "assistant":
                return self._is_retryable_error(message)
        return False

    def _summarization_retry_callbacks(self, source: Dict[str, Any]) -> RetryCallbacks:
        """Retry reporting shared by compaction and branch-summary calls (pi's
        `_summarizationRetryCallbacks`); `source` recreates the indicator."""
        return RetryCallbacks(
            on_retry_scheduled=lambda attempt, max_attempts, delay_ms, error_message: self._emit(
                {
                    "type": "summarization_retry_scheduled",
                    "attempt": attempt,
                    "maxAttempts": max_attempts,
                    "delayMs": delay_ms,
                    "errorMessage": error_message,
                }
            ),
            on_retry_attempt_start=lambda: self._emit(
                {"type": "summarization_retry_attempt_start", **source}
            ),
            on_retry_finished=lambda *args: self._emit({"type": "summarization_retry_finished"}),
        )

    # -- persistence --------------------------------------------------------------

    async def _on_agent_event(self, event, signal) -> None:
        event_type = getattr(event, "type", None)
        if event_type == "agent_start":
            self._abort_requested = False
            self._run_appends = []
            branch = await self.session.branch(self.branch_name)
            self._pre_run_tip = await branch.get_tip_id()
        elif event_type == "message_end":
            await self._persist_message(event.message)
            message = event.message
            if message_field(message, "role") == "assistant":
                # Reset the retry counter immediately on a successful assistant
                # response, so it cannot accumulate across a turn's LLM calls.
                if (
                    message_field(message, "stop_reason", "stopReason") != "error"
                    and self._retry_attempt > 0
                ):
                    attempt = self._retry_attempt
                    self._retry_attempt = 0
                    self._emit({"type": "auto_retry_end", "success": True, "attempt": attempt})

    async def _flush_pending_bash_messages(self) -> None:
        """Append side-channel bash results queued during the run (pi's
        `_runAgentPrompt` settle flush)."""
        if not self._pending_bash_messages:
            return
        pending, self._pending_bash_messages = list(self._pending_bash_messages), []
        for message in pending:
            await self._persist_message(message)
            self.agent.state.messages.append(message)

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
        self._run_appends.append(("message", entry_id, None))

    async def _append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        """Append a non-context custom entry to the current branch tip.

        pi's session entries of this kind (labels, model/thinking changes) carry
        metadata the context builder ignores.
        """
        entry_id = self.session.id_generator.next()
        written: Dict[str, Any] = {}

        async def write_one(mutator):
            tip = await mutator.get_value(branch_tip(self.branch_name))
            entry = CustomEntry(
                id=entry_id,
                parent_id=tip.value,
                custom_type=custom_type,
                data=data,
            )
            written["entry"] = entry
            await mutator.commit(
                [
                    insert_entry(entry),
                    set_value(branch_tip(self.branch_name), entry_id),
                ]
            )

        await self.session.mutate(write_one)
        # Recorded so `_omit_final_attempt` can re-parent it if a recovery
        # rewinds the tip past it (the entry carries no model context, but it
        # carries state a resume reads back — the thinking level, a label).
        self._run_appends.append(("custom", entry_id, written["entry"]))
        return entry_id

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
            preparation,
            self.models,
            self.model,
            custom_instructions=custom_instructions,
            retry=self.retry,
            callbacks=self._summarization_retry_callbacks(
                {"source": "compaction", "reason": reason}
            ),
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
        # pi runs image normalization AFTER the extension tool_result hook so
        # hook-injected images are normalized too. Oversized images from tools
        # (screenshots, MCP bridges) would otherwise make the provider reject
        # the whole conversation, so normalize them once here.
        content = hook_context.result.content
        if content and any(getattr(block, "type", None) == "image" for block in content):
            from .utils import normalize_tool_result_images

            normalized = await normalize_tool_result_images(
                content,
                auto_resize_images=self.auto_resize_images,
                resize_options=self._image_resize_options(),
            )
            if normalized != list(content):
                return AfterToolCallResult(content=normalized)
        return None

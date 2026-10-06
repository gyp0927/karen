"""karen demo CLI — the full M0–M3 stack as an interactive terminal agent.

- agent loop with the built-in read/write/edit/bash tools, streaming output
- sessions persisted under ~/.karen/sessions (resume per working directory)
- automatic + manual compaction, persisted as compaction entries; a context
  overflow (or recoverable length stop) triggers "overflow" compaction and one
  bounded compact-and-retry attempt (pi's AgentSession overflow recovery)
- prompt templates from .karen/prompts (project) and ~/.karen/prompts (user),
  invocable as /name args...
- a HookRegistry wired into the loop's tool hooks (demo: a path guard +
  a tool-call counter)

Usage:
    python examples/karen_cli.py [--cwd PATH] [--model ID] [--new]

Piping works too, which is how the automated smoke drives it:
    printf 'hello\n/quit\n' | python examples/karen_cli.py --new
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from karen_ai import CreateModelsOptions, JsonFileCredentialStore, SystemMessage, UserMessage, create_models
from karen_ai.providers import deepseek_provider
from karen_ai.utils.overflow import is_context_overflow, is_recoverable_length
from karen_agent import (
    AfterToolEvent,
    AgentContext,
    AgentLoopConfig,
    BeforeCompactionEvent,
    BeforeToolCallResult,
    BeforeToolEvent,
    BeforeToolResult,
    HookRegistry,
    PromptTemplate,
    ToolBlock,
    agent_loop,
    agent_loop_continue,
    convert_to_llm,
    format_prompt_template_invocation,
    load_prompt_templates,
    models_stream_fn,
    parse_command_args,
)
from karen_agent.compaction import (
    CompactionSettings,
    compact,
    estimate_context_tokens,
    prepare_compaction,
    should_compact,
)
from karen_agent.result import Err
from karen_agent.session import (
    BranchScan,
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionRepo,
    MessageEntry,
    UsageRow,
    branch_tip,
    insert_entry,
    insert_usage,
    set_value,
)
from karen_agent.session.context import build_session_context
from karen_agent.session.types import CompactionEntry
from karen_agent.tools import create_builtin_tools

DEFAULT_MODEL_ID = "deepseek-v4-pro"
DEFAULT_CREDENTIALS = Path.home() / ".karen" / "credentials.json"
SESSIONS_ROOT = Path(os.environ.get("KAREN_SESSIONS_ROOT", Path.home() / ".karen" / "sessions"))
SYSTEM_PROMPT = """You are karen, an AI coding assistant running in a demo CLI.
The working directory is {cwd}; relative tool paths resolve against it.
Use the read/write/edit/bash tools to inspect and modify files.
Keep answers concise."""

HELP_TEXT = """Commands:
  /help                 show this help
  /new                  start a fresh session
  /compact [focus]      compact the context now (optional extra instructions)
  /templates            list available prompt templates
  /quit                 exit
  /<template> [args]    invoke a prompt template ($1, $@, ${@:N:L} supported)
Anything else is sent to the model."""


def parse_command(line: str, template_names):
    """Route one input line. Returns (kind, name, arg_string).

    kind: "quit" | "help" | "new" | "compact" | "templates" | "template" | "prompt"
    """
    if not line.startswith("/"):
        return "prompt", None, line
    body = line[1:]
    name, _, rest = body.partition(" ")
    name = name.strip()
    rest = rest.strip()
    if name in ("quit", "exit", "q"):
        return "quit", None, ""
    if name == "help":
        return "help", None, ""
    if name == "new":
        return "new", None, ""
    if name == "compact":
        return "compact", None, rest
    if name == "templates":
        return "templates", None, ""
    if name in template_names:
        return "template", name, rest
    return "prompt", None, line  # unknown slash command: send verbatim


def format_args_preview(args, max_len: int = 72) -> str:
    """One-line preview of tool call arguments for the terminal."""
    parts = []
    for key, value in args.items():
        if isinstance(value, str):
            parts.append(f"{key}={value.replace(chr(10), chr(92) + 'n')!r}")
        else:
            parts.append(f"{key}={repr(value)}")
    preview = ", ".join(parts)
    if len(preview) > max_len:
        preview = preview[: max_len - 1] + "…"
    return preview


def build_models():
    override = os.environ.get("KAREN_CREDENTIALS_PATH")
    path = Path(override).expanduser() if override else DEFAULT_CREDENTIALS
    if path.exists():
        return create_models(CreateModelsOptions(credentials=JsonFileCredentialStore(path))), f"credentials file {path}"
    return create_models(), "environment"


def load_all_templates(cwd: str):
    result = load_prompt_templates(
        [str(Path(cwd) / ".karen" / "prompts"), str(Path.home() / ".karen" / "prompts")]
    )
    return result.prompt_templates, result.diagnostics


def now_ms() -> int:
    return int(time.time() * 1000)


class KarenCli:
    def __init__(self, cwd: str, model_id: str, fresh: bool) -> None:
        self.cwd = cwd
        self.models, self.auth_source = build_models()
        self.models.set_provider(deepseek_provider())
        self.model = self.models.get_model("deepseek", model_id)
        self.fresh = fresh
        self.repo = JsonlSessionRepo(str(SESSIONS_ROOT))
        self.session = None
        self.branch_name = "main"
        self.context = AgentContext(messages=[], tools=create_builtin_tools(cwd))
        self.settings = CompactionSettings()
        self.templates, self.template_diagnostics = load_all_templates(cwd)
        self.tool_counts = {}

        def report_hook_error(error, hook):
            print(f"\n[hook {hook} error: {error}]", file=sys.stderr)

        self.hooks = HookRegistry(report_hook_error)
        self._register_demo_hooks()

    # -- hooks ----------------------------------------------------------------

    def _register_demo_hooks(self) -> None:
        cwd_root = os.path.normcase(os.path.realpath(self.cwd))

        def guard_paths(event):
            """Demo before_tool hook: refuse writes outside the working directory."""
            if event.tool_name in ("write", "edit") and isinstance(event.args.get("path"), str):
                raw = event.args["path"]
                resolved = os.path.normcase(
                    os.path.realpath(raw if os.path.isabs(raw) else os.path.join(self.cwd, raw))
                )
                if not (resolved == cwd_root or resolved.startswith(cwd_root + os.sep)):
                    return BeforeToolResult(
                        block=ToolBlock(reason=f"path escapes the working directory: {raw}")
                    )
            return None

        def count_tools(event):
            """Demo after_tool hook: count completed tool calls by name."""
            self.tool_counts[event.tool_name] = self.tool_counts.get(event.tool_name, 0) + 1
            return None

        self.hooks.on("before_tool", guard_paths, id="demo.path-guard")
        self.hooks.on("after_tool", count_tools, id="demo.counter")

    async def _before_tool_call(self, hook_context, signal):
        """Bridge: loop before_tool_call -> registry before_tool."""
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
        """Bridge: loop after_tool_call -> registry after_tool."""
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

    # -- session --------------------------------------------------------------

    async def open_session(self) -> None:
        if not self.fresh:
            existing = await self.repo.list(JsonlSessionListOptions(cwd=self.cwd))
            if existing:
                self.session = await self.repo.open(existing[0])
                print(f"resumed session {self.session.metadata.id} ({existing[0].path})")
        if self.session is None:
            self.session = await self.repo.create(JsonlSessionCreateOptions(cwd=self.cwd))
            print(f"new session {self.session.metadata.id}")
        if await self.session.branch(self.branch_name) is None:
            await self.session.create_branch(self.branch_name, None)
        await self.reload_context()
        restored = len(self.context.messages) - 1  # minus the seeded system message
        if restored:
            print(f"context restored: {restored} messages, {self.estimate_tokens()} tokens")

    def _system_message(self) -> SystemMessage:
        # pi's initialState seeds the leading system message the same way: prompt
        # plus the tool declaration, so the loop's tool-delta against the committed
        # transcript is empty and providers resolve tools from the head message.
        return SystemMessage(
            content=SYSTEM_PROMPT.format(cwd=self.cwd),
            tools_added=self.context.tools,
            timestamp=now_ms(),
        )

    async def reload_context(self) -> None:
        branch = await self.session.branch(self.branch_name)
        entries = await branch.find_entries(BranchScan(order="oldestFirst"))
        session_messages = await build_session_context(entries)
        self.context.messages = [self._system_message(), *session_messages]

    def estimate_tokens(self) -> int:
        return estimate_context_tokens(self.context.messages).tokens

    async def persist_messages(self, messages) -> None:
        for message in messages:
            entry_id = self.session.id_generator.next()

            async def write_one(mutator, message=message, entry_id=entry_id):
                tip = await mutator.get_value(branch_tip(self.branch_name))
                writes = [
                    insert_entry(MessageEntry(id=entry_id, parent_id=tip.value, message=message)),
                    set_value(branch_tip(self.branch_name), entry_id),
                ]
                usage = getattr(message, "usage", None)
                if getattr(message, "role", None) == "assistant" and usage is not None and usage.total_tokens:
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

    # -- compaction -------------------------------------------------------------

    async def run_compaction(self, reason: str, custom_instructions=None) -> bool:
        branch = await self.session.branch(self.branch_name)
        entries = await branch.find_entries(BranchScan(order="oldestFirst"))
        preparation_result = prepare_compaction(entries, self.settings)
        if isinstance(preparation_result, Err) or preparation_result.value is None:
            print("nothing to compact")
            return False
        preparation = preparation_result.value
        hook_result = await self.hooks.run(
            "before_compaction",
            BeforeCompactionEvent(
                reason=reason, preparation=preparation, custom_instructions=custom_instructions
            ),
        )
        if hook_result is not None and hook_result.decline:
            print("compaction declined by hook")
            return False
        print("compacting (the model writes a summary)...")
        result = await compact(preparation, self.models, self.model, custom_instructions=custom_instructions)
        if isinstance(result, Err):
            print(f"compaction failed: {result.error}", file=sys.stderr)
            return False
        compacted = result.value
        compaction_id = self.session.id_generator.next()

        async def write_compaction(mutator):
            tip = await mutator.get_value(branch_tip(self.branch_name))
            await mutator.commit([
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
            ])

        await self.session.mutate(write_compaction)
        await self.reload_context()
        print(f"compacted ~{compacted.tokens_before} tokens -> {len(self.context.messages)} context messages")
        return True

    async def maybe_auto_compact(self) -> None:
        tokens = self.estimate_tokens()
        window = self.model.context_window or 128000
        if should_compact(tokens, window, self.settings):
            print(f"[context ~{tokens} tokens exceeds {window} window - reserve]")
            await self.run_compaction("threshold")

    # -- the agent turn -----------------------------------------------------------

    async def _drive(self, stream):
        """Consume one agent stream, printing text deltas and tool calls, and
        return the run's NEW messages (prompt first on the initial drive)."""
        async for event in stream:
            if event.type == "message_update" and event.assistant_message_event.type == "text_delta":
                print(event.assistant_message_event.delta, end="", flush=True)
            elif event.type == "tool_execution_start":
                print(f"\n[tool ->] {event.tool_name}({format_args_preview(event.args)})")
            elif event.type == "tool_execution_end":
                if event.is_error:
                    text = "".join(getattr(c, "text", "") for c in event.result.content)
                    print(f"[tool <-] {event.tool_name} ERROR: {text[:200]}")
        print()
        return await stream.result()

    async def _persist_and_extend(self, messages) -> None:
        await self.persist_messages(messages)
        # the loop works on its own copy of the context, so extend ours to stay in sync
        self.context.messages = [*self.context.messages, *messages]

    @staticmethod
    def _omit_final_attempt(messages):
        """Drop the final assistant attempt (the failed message plus any tool
        results it produced) — pi's `_omitRecoveryAttempt`."""
        index = len(messages) - 1
        while index > 0 and getattr(messages[index], "role", None) != "assistant":
            index -= 1
        return list(messages[:index])

    def _overflow_action(self, message):
        """pi's `AgentSession._checkCompaction` overflow branch: "retry"
        (compact + retry the turn), "compact_only" (compact but keep the
        completed response), or None.

        Simplifications (the CLI transcript is append-only, so pi's
        projection/context-edit retention guards are trivially true, and only
        fresh post-run messages are checked, so pi's stale pre-compaction
        guard cannot trigger): no projection comparison, no compaction-
        boundary timestamp check.
        """
        if getattr(message, "role", None) != "assistant" or message.stop_reason == "aborted":
            return None
        # pi skips the check when the message came from a different model (the
        # user switched to a larger-context model after the overflow).
        if message.provider != self.model.provider or message.model != self.model.id:
            return None
        overflow = is_context_overflow(message, self.model.context_window or None)
        recoverable = is_recoverable_length(message, self.model.max_tokens or 0)
        if not (overflow or recoverable):
            return None
        # pi: willRetry = stopReason !== "stop" — agent.continue() cannot
        # continue from a completed assistant response.
        return "compact_only" if message.stop_reason == "stop" else "retry"

    async def run_turn(self, prompt_text: str) -> None:
        prompt = UserMessage(content=prompt_text, timestamp=now_ms())
        config = AgentLoopConfig(
            model=self.model,
            convert_to_llm=convert_to_llm,
            before_tool_call=self._before_tool_call,
            after_tool_call=self._after_tool_call,
        )
        stream_fn = models_stream_fn(self.models)
        new_messages = await self._drive(agent_loop([prompt], self.context, config, None, stream_fn))

        # Overflow recovery: one bounded compact-and-retry attempt per prompt.
        recovery_attempted = False
        while True:
            final = new_messages[-1] if new_messages else None
            action = self._overflow_action(final)
            if action == "retry" and not recovery_attempted:
                recovery_attempted = True
                # the failed attempt never reaches the transcript
                await self._persist_and_extend(self._omit_final_attempt(new_messages))
                if await self.run_compaction("overflow"):
                    print("[context overflow: compacted; retrying the turn]")
                    new_messages = await self._drive(agent_loop_continue(self.context, config, None, stream_fn))
                    continue
                new_messages = []
                break
            if action == "retry":  # recovery already attempted: keep the failure and give up
                print(
                    "[context overflow recovery failed after one compact-and-retry attempt; "
                    "try reducing context or switching to a larger-context model]",
                    file=sys.stderr,
                )
            await self._persist_and_extend(new_messages)
            if action == "compact_only":
                await self.run_compaction("overflow")
            break

        final = new_messages[-1] if new_messages else None
        if getattr(final, "stop_reason", None) == "error":
            print(f"[run failed: {final.error_message}]", file=sys.stderr)
        print(f"[context ~{self.estimate_tokens()} tokens; tools used: {self.tool_counts or '{}'}]")
        await self.maybe_auto_compact()

    # -- main loop ---------------------------------------------------------------

    async def repl(self) -> int:
        await self.open_session()
        print(f"model: deepseek/{self.model.id} | cwd: {self.cwd} | tools: read write edit bash")
        print("type /help for commands")
        if self.template_diagnostics:
            for diagnostic in self.template_diagnostics:
                print(f"[template warning: {diagnostic.code} {diagnostic.path}]", file=sys.stderr)
        while True:
            try:
                line = input("karen> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            kind, name, rest = parse_command(line, {t.name for t in self.templates})
            if kind == "quit":
                break
            if kind == "help":
                print(HELP_TEXT)
                continue
            if kind == "templates":
                if not self.templates:
                    print("no templates (add .md files to .karen/prompts or ~/.karen/prompts)")
                for template in self.templates:
                    print(f"  /{template.name}  {template.description or ''}")
                continue
            if kind == "new":
                await self.session.close()
                self.session = None
                self.fresh = True
                await self.open_session()
                continue
            if kind == "compact":
                await self.run_compaction("manual", custom_instructions=rest or None)
                continue
            if kind == "template":
                template = next(t for t in self.templates if t.name == name)
                line = format_prompt_template_invocation(template, parse_command_args(rest))
            try:
                await self.run_turn(line)
            except KeyboardInterrupt:
                print("\n[interrupted]")
            except Exception as error:
                print(f"[error: {error}]", file=sys.stderr)
        await self.session.close()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="karen demo CLI")
    parser.add_argument("--cwd", default=os.getcwd(), help="working directory for tools and session resume")
    parser.add_argument("--model", default=os.environ.get("KAREN_MODEL", DEFAULT_MODEL_ID))
    parser.add_argument("--new", action="store_true", help="start a fresh session instead of resuming")
    args = parser.parse_args()
    cli = KarenCli(cwd=os.path.abspath(args.cwd), model_id=args.model, fresh=args.new)
    return asyncio.run(cli.repl())


if __name__ == "__main__":
    sys.exit(main())

"""`karen` — the CLI coding assistant (M1: REPL + print mode; M3: JSON mode;
M5: RPC mode; M6: structured system prompt, context files and skills; M7:
session navigation — tree, fork, clone, switch; M8: auto-retry).

Interactive:
    karen [--cwd PATH] [--model ID] [--new]

Headless (print mode, pi's runPrintMode):
    karen -p "summarize this repo" [--cwd PATH]   # final reply text on stdout
    karen "one prompt" "another prompt"           # prompts run sequentially
    karen --mode json "prompt"                    # JSON event stream on stdout
    karen --mode rpc                              # JSON command protocol (see rpc.py)

Piping into the REPL works too (that's how the smokes drive it):
    printf 'hello\n/quit\n' | karen --new

Deviation from pi: karen does not auto-switch to print mode when stdin/stdout
is not a TTY (the REPL is designed to be pipe-driven), and positional prompts
imply print mode instead of becoming an interactive initial message.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from karen_ai import CreateModelsOptions, JsonFileCredentialStore, create_models
from karen_ai.providers import deepseek_provider
from karen_agent import (
    format_prompt_template_invocation,
    format_skill_invocation,
    load_prompt_templates,
    parse_command_args,
)

from .agent_session import AgentSession
from .json_events import to_json_event
from .navigation import render_tree
from .prompt import build_system_prompt_sections
from .resources import (
    DEFAULT_AGENT_DIR,
    discover_append_system_prompt_file,
    discover_system_prompt_file,
    load_project_context_files,
    load_project_skills,
    read_text_file,
)
from .rpc import run_rpc_mode
from .settings import (
    DEFAULT_SETTINGS_PATH,
    LoadedSettings,
    compaction_settings_from_wire,
    image_auto_resize,
    image_block_images,
    load_settings,
    resolve_default_tool_names,
    retry_policy_from_wire,
    update_settings,
)
from .tools import create_default_tools

DEFAULT_MODEL_ID = "deepseek-v4-pro"
DEFAULT_CREDENTIALS = Path.home() / ".karen" / "credentials.json"

HELP_TEXT = """Commands:
  /help                 show this help
  /new                  start a fresh session
  /compact [focus]      compact the context now (optional extra instructions)
  /retry [on|off]       show or toggle auto-retry of transient provider failures
  /tree [id]            show the session tree; with an id, move the branch tip there
  /tree --summarize <id>  move there and summarize the abandoned branch
  /fork [n|id]          fork from a user message (lists them when no argument)
  /clone                copy the current branch into a new session
  /sessions             list saved sessions for this directory
  /resume <n|id>        switch to a listed session
  /name [text]          show or set the session name
  /session              show session info (id, name, file, messages, tokens, cost)
  /thinking [level|cycle]  show or set the reasoning level for the current model
  /export [jsonl|html] [path]  export the session (default: html)
  /settings [show | global|project key=value ...]  inspect or edit settings files
  /templates            list available prompt templates
  /skills               list available skills
  /quit                 exit
  /<name> [args]        invoke a prompt template or skill from the current project
Anything else is sent to the model."""


def parse_command(line: str, template_names, skill_names=()):
    """Route one input line. Returns (kind, name, arg_string).

    kind: "quit" | "help" | "new" | "compact" | "retry" | "tree" | "fork"
          | "clone" | "sessions" | "resume" | "name" | "session"
          | "templates" | "skills" | "template" | "skill" | "prompt"

    Templates and skills share one `/name` namespace, like pi's slash commands;
    templates win when a name is defined as both.
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
    if name == "retry":
        return "retry", None, rest
    if name == "tree":
        return "tree", None, rest
    if name == "fork":
        return "fork", None, rest
    if name == "clone":
        return "clone", None, ""
    if name == "sessions":
        return "sessions", None, rest
    if name == "resume":
        return "resume", None, rest
    if name == "name":
        return "name", None, rest
    if name == "session":
        return "session", None, ""
    if name == "thinking":
        return "thinking", None, rest
    if name == "export":
        return "export", None, rest
    if name == "settings":
        return "settings", None, rest
    if name == "templates":
        return "templates", None, ""
    if name == "skills":
        return "skills", None, ""
    if name in template_names:
        return "template", name, rest
    if name in skill_names:
        return "skill", name, rest
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


def load_all_templates(cwd: str, extra_paths=None):
    result = load_prompt_templates(
        [
            str(Path(cwd) / ".karen" / "prompts"),
            str(Path.home() / ".karen" / "prompts"),
            *(extra_paths or []),
        ]
    )
    return result.prompt_templates, result.diagnostics


class KarenCli:
    """Terminal front-end over `AgentSession`: event printing + the REPL."""

    def __init__(self, cwd: str, model_id: str, fresh: bool, quiet_tools: bool = False,
                 provider: str = "deepseek", output_mode: str = "text",
                 loaded_settings: "LoadedSettings | None" = None) -> None:
        self.cwd = cwd
        self.provider = provider
        self.model_id = model_id
        self.fresh = fresh
        self.quiet_tools = quiet_tools  # headless: tool chatter goes to stderr
        self.output_mode = output_mode  # "text" (final reply) | "json" (event stream)
        self.loaded_settings = loaded_settings if loaded_settings is not None else load_settings(cwd)
        self.settings = self.loaded_settings.settings
        self.models, self.auth_source = build_models()
        if provider == "deepseek":
            self.models.set_provider(deepseek_provider())
        self.model = self.resolve_model()
        self.agent_dir = DEFAULT_AGENT_DIR
        self.templates, self.template_diagnostics = load_all_templates(cwd, self.settings.prompts)
        skills_result = load_project_skills(cwd, self.agent_dir)
        self.skills = skills_result.skills
        self.skill_diagnostics = skills_result.diagnostics
        self.session: AgentSession | None = None
        self.tool_counts = {}
        self._print_settings_diagnostics()
        for diagnostic in self.skill_diagnostics:
            print(f"[skill warning: {diagnostic.code} {diagnostic.path}: {diagnostic.message}]", file=sys.stderr)

    def _print_settings_diagnostics(self) -> None:
        for diagnostic in self.loaded_settings.diagnostics:
            print(
                f"[settings warning: Invalid settings file {diagnostic.path}: {diagnostic.message}]",
                file=sys.stderr,
            )

    def resolve_model(self):
        model = self.models.get_model(self.provider, self.model_id)
        if model is None:
            raise SystemExit(f"unknown model: {self.provider}/{self.model_id}")
        return model

    # -- printing ---------------------------------------------------------------

    def _tool_line(self, text: str) -> None:
        print(text, file=sys.stderr if self.quiet_tools else sys.stdout)

    def _on_agent_event(self, event, signal) -> None:
        if self.output_mode == "json":
            print(json.dumps(to_json_event(event), ensure_ascii=False), flush=True)
            return
        if (
            not self.quiet_tools
            and event.type == "message_update"
            and event.assistant_message_event.type == "text_delta"
        ):
            print(event.assistant_message_event.delta, end="", flush=True)
        elif event.type == "tool_execution_start":
            self.tool_counts[event.tool_name] = self.tool_counts.get(event.tool_name, 0) + 1
            self._tool_line(f"\n[tool ->] {event.tool_name}({format_args_preview(event.args)})")
        elif event.type == "tool_execution_end" and event.is_error:
            text = "".join(getattr(c, "text", "") for c in event.result.content)
            self._tool_line(f"[tool <-] {event.tool_name} ERROR: {text[:200]}")
        elif event.type == "agent_end" and not self.quiet_tools:
            # A run that is about to be retried is not a failure (pi's `willRetry`).
            if getattr(event, "will_retry", None):
                return
            print()
            final = self.session.agent.state.messages[-1] if self.session else None
            if getattr(final, "stop_reason", None) == "error":
                print(f"[run failed: {final.error_message}]", file=sys.stderr)

    def _on_session_event(self, event) -> None:
        event_type = event.get("type")
        if self.output_mode == "json":
            # the session header line covers session_opened
            if event_type != "session_opened":
                print(json.dumps(event, ensure_ascii=False), flush=True)
            return
        if event_type == "session_opened":
            kind = "resumed" if event["resumed"] else event.get("reason", "new")
            self._tool_line(f"{kind} session {event['session_id']}")
        elif event_type == "compaction_start":
            self._tool_line(f"compacting ({event['reason']}; the model writes a summary)...")
        elif event_type == "compaction_end":
            if event["compacted"]:
                self._tool_line(f"compacted ~{event['tokens_before']} tokens")
            elif event.get("detail") == "nothing_to_compact":
                self._tool_line("nothing to compact")
            else:
                print(f"compaction failed: {event.get('detail')}", file=sys.stderr)
        elif event_type == "overflow_retry":
            self._tool_line("[context overflow: compacted; retrying the turn]")
        elif event_type == "overflow_give_up":
            print(
                "[context overflow recovery failed after one compact-and-retry attempt; "
                "try reducing context or switching to a larger-context model]",
                file=sys.stderr,
            )
        elif event_type == "auto_retry_start":
            seconds = event["delayMs"] / 1000
            self._tool_line(
                f"[retrying (attempt {event['attempt']}/{event['maxAttempts']}) "
                f"in {seconds:.1f}s: {event['errorMessage']}]"
            )
        elif event_type == "auto_retry_end":
            if event["success"]:
                self._tool_line(f"[retry succeeded on attempt {event['attempt']}]")
            elif event.get("finalError") == "Retry cancelled":
                pass  # Ctrl-C already reported the interruption
            else:
                # the failed turn itself is reported by the agent_end handler
                print(f"[auto-retry gave up after {event['attempt']} attempt(s)]", file=sys.stderr)
        elif event_type == "summarization_retry_scheduled":
            seconds = event["delayMs"] / 1000
            self._tool_line(
                f"[retrying summary (attempt {event['attempt']}/{event['maxAttempts']}) "
                f"in {seconds:.1f}s: {event['errorMessage']}]"
            )

    # -- session ---------------------------------------------------------------

    def _select_default_tools(self, entries):
        """Apply a settings `defaultTools` list (pi's resolveDefaultTools)."""
        all_tools = create_default_tools(
            self.cwd,
            shell_path=self.settings.shell_path,
            shell_command_prefix=self.settings.shell_command_prefix,
            # pi captures `autoResizeImages` when it builds the read tool too.
            auto_resize_images=image_auto_resize(self.settings.images),
        )
        all_names = [tool.name for tool in all_tools]
        # no settings entry at all -> pi's built-in default tool set
        names = resolve_default_tool_names(entries, all_names) if entries else all_names
        by_name = {tool.name: tool for tool in all_tools}
        selected = []
        for name in names:
            tool = by_name.get(name)
            if tool is None:
                print(f"[settings warning: unknown tool in defaultTools: {name}]", file=sys.stderr)
                continue
            selected.append(tool)
        return selected

    def _build_prompt_sections(self, tools):
        """Assemble the structured system prompt (pi's buildSystemPrompt call).

        Sources: `SYSTEM.md` replaces the default preamble, `APPEND_SYSTEM.md`
        adds the `addendum` section, AGENTS.md/CLAUDE.md ancestors become
        `project_context`, skills become `skills`, and the session's tool set
        drives `tools`/`rules`.
        """
        system_prompt_file = discover_system_prompt_file(self.cwd, self.agent_dir)
        append_prompt_file = discover_append_system_prompt_file(self.cwd, self.agent_dir)
        custom_prompt = self._read_optional(system_prompt_file)
        append_prompt = self._read_optional(append_prompt_file) or ""
        return build_system_prompt_sections(
            cwd=self.cwd,
            selected_tools=[tool.name for tool in tools],
            custom_prompt=custom_prompt,
            append_system_prompt=append_prompt,
            context_files=load_project_context_files(self.cwd, self.agent_dir),
            skills=self.skills,
        )

    def _read_optional(self, path):
        if path is None:
            return None
        try:
            return read_text_file(path)
        except OSError as error:
            print(f"[warning: could not read {path}: {error}]", file=sys.stderr)
            return None

    async def _open_session(self, fresh: bool) -> None:
        if self.session is not None:
            await self.session.close()
        settings = self.settings
        sessions_root = None
        if settings.session_dir and not os.environ.get("KAREN_SESSIONS_ROOT"):
            sessions_root = settings.session_dir
        tools = self._select_default_tools(settings.default_tools)
        retry_policy = retry_policy_from_wire(settings.retry)
        self.session = AgentSession(
            cwd=self.cwd,
            models=self.models,
            model=self.model,
            fresh=fresh,
            listener=self._on_session_event,
            sessions_root=sessions_root,
            tools=tools,
            system_prompt_sections=self._build_prompt_sections(tools),
            shell_path=settings.shell_path,
            shell_command_prefix=settings.shell_command_prefix,
            auto_resize_images=image_auto_resize(settings.images),
            block_images=image_block_images(settings.images),
            compaction_settings=compaction_settings_from_wire(settings.compaction)
            if settings.compaction is not None
            else None,
            retry_policy=retry_policy,
        )
        await self.session.open()
        self.session.subscribe(self._on_agent_event)
        self.tool_counts = {}
        restored = len(self.session.agent.state.messages) - 1  # minus the seeded system message
        if restored:
            self._tool_line(f"context restored: {restored} messages, ~{self.session.estimate_tokens()} tokens")

    # -- navigation --------------------------------------------------------------

    async def _handle_navigation(self, kind: str, rest: str) -> None:
        """Route the `/tree`, `/fork`, `/clone`, `/sessions`, `/resume`, `/name`
        and `/session` REPL commands (pi's tree/fork/session TUI actions)."""
        if kind == "tree":
            await self._handle_tree(rest)
        elif kind == "fork":
            await self._handle_fork(rest)
        elif kind == "clone":
            await self._handle_clone()
        elif kind in ("sessions", "resume"):
            await self._handle_sessions(rest)
        elif kind == "name":
            await self._handle_name(rest)
        else:
            await self._print_session_info()

    async def _resolve_entry(self, token: str) -> str:
        """Resolve a full entry id, or a unique prefix/suffix of one.

        Tree lines print the id's *last* 8 characters: karen's uuid7 ids share
        their timestamp prefix, so the head is not distinguishing.
        """
        matches = [
            entry.id
            for entry in await self.session.entries()
            if entry.id.startswith(token) or entry.id.endswith(token)
        ]
        if not matches:
            raise ValueError(f"no entry matches {token!r}")
        if len(matches) > 1:
            raise ValueError(f"ambiguous entry prefix {token!r} ({len(matches)} entries)")
        return matches[0]

    async def _show_tree(self) -> None:
        roots = await self.session.session_tree()
        if not roots:
            print("(empty session tree)")
            return
        print(
            render_tree(
                roots,
                leaf_id=await self.session.branch_tip_id(),
                tips=await self.session.branch_tips(),
            )
        )
        print("● current tip   ○ other branch tips   [label] entry label")

    async def _handle_tree(self, rest: str) -> None:
        tokens = rest.split()
        summarize = bool(tokens) and tokens[0] == "--summarize"
        if summarize:
            tokens = tokens[1:]
        if not tokens:
            await self._show_tree()
            return
        target = await self._resolve_entry(tokens[0])
        result = await self.session.navigate_tree(target, summarize=summarize)
        if result["summaryEntryId"]:
            print(f"summarized the abandoned branch into {result['summaryEntryId'][-8:]}")
        await self._show_tree()
        if result["editorText"]:
            print("[message left the branch; send it again to branch off here]")
            print(result["editorText"])

    async def _handle_fork(self, rest: str) -> None:
        messages = await self.session.user_messages_for_forking()
        if not messages:
            print("nothing to fork from: this session has no user messages yet")
            return
        token = rest.split()[0] if rest.split() else ""
        if not token:
            for index, item in enumerate(messages, start=1):
                preview = " ".join(item["text"].split())
                print(f"  {index}. [{item['entryId'][-8:]}] {preview[:72]}")
            print("fork from one of these with /fork <n|id>")
            return
        if token.isdigit():
            index = int(token)
            if not 1 <= index <= len(messages):
                print(f"[no fork target {index}: choose 1..{len(messages)}]")
                return
            entry_id = messages[index - 1]["entryId"]
        else:
            entry_id = await self._resolve_entry(token)
        result = await self.session.fork(entry_id)
        print(f"forked into session {self.session.session.metadata.id}")
        if result["selectedText"]:
            print("[edit and send it again to branch off here]")
            print(result["selectedText"])

    async def _handle_clone(self) -> None:
        if await self.session.branch_tip_id() is None:
            print("[nothing to clone yet: send a message first]")
            return
        await self.session.clone()
        print(f"cloned into session {self.session.session.metadata.id}")

    async def _handle_sessions(self, rest: str) -> None:
        sessions = await self.session.list_sessions()
        if not sessions:
            print(f"no saved sessions for {self.cwd}")
            return
        token = rest.split()[0] if rest.split() else ""
        if not token:
            current = self.session.session.metadata.id
            for index, metadata in enumerate(sessions, start=1):
                marker = "*" if metadata.id == current else " "
                stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(metadata.created_at / 1000))
                print(f"{marker} {index}. {metadata.id[-8:]}  {stamp}  {metadata.path}")
            print("switch with /resume <n|id>")
            return
        metadata = None
        if token.isdigit():
            index = int(token)
            if 1 <= index <= len(sessions):
                metadata = sessions[index - 1]
        else:
            matches = [m for m in sessions if m.id.startswith(token) or m.id.endswith(token)]
            if len(matches) > 1:
                print(f"[ambiguous session id {token!r}]")
                return
            metadata = matches[0] if matches else None
        if metadata is None:
            print(f"[no session matches {token!r}]")
            return
        await self.session.switch_session(metadata)
        restored = len(self.session.agent.state.messages) - 1
        print(f"switched to session {metadata.id}")
        print(f"context restored: {restored} messages, ~{self.session.estimate_tokens()} tokens")

    async def _handle_name(self, rest: str) -> None:
        if rest:
            await self.session.set_session_name(rest)
        name = await self.session.session_name()
        print(f"session name: {name or '(unnamed)'}")

    def _handle_retry(self, rest: str) -> None:
        """`/retry [on|off]` — pi's `setAutoRetryEnabled` toggle."""
        argument = rest.strip().lower()
        if argument in ("on", "off"):
            self.session.set_auto_retry_enabled(argument == "on")
        elif argument:
            print("usage: /retry [on|off]", file=sys.stderr)
            return
        policy = self.session.retry
        print(
            f"auto-retry: {'on' if policy.enabled else 'off'} "
            f"(maxRetries {policy.max_retries}, baseDelayMs {policy.base_delay_ms})"
        )

    async def _handle_thinking(self, rest: str) -> None:
        """`/thinking [level]` — show or set the reasoning level (pi's thinking commands)."""
        argument = rest.strip().lower()
        available = self.session.get_available_thinking_levels()
        if not argument:
            print(f"thinking: {self.session.agent.state.thinking_level} (available: {', '.join(available)})")
            return
        if argument == "cycle":
            level = await self.session.cycle_thinking_level()
            if level is None:
                print("this model does not support thinking", file=sys.stderr)
                return
            print(f"thinking: {level}")
            return
        if argument not in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
            print(f"usage: /thinking [{'|'.join(available)}|cycle]", file=sys.stderr)
            return
        await self.session.set_thinking_level(argument)
        print(f"thinking: {self.session.agent.state.thinking_level}")

    async def _handle_export(self, rest: str) -> None:
        """`/export [jsonl|html] [path]` — write the session out (pi's export commands)."""
        parts = rest.split()
        fmt = parts[0].lower() if parts else "html"
        output = parts[1] if len(parts) > 1 else None
        if fmt in ("jsonl", "json"):
            path = await self.session.export_to_jsonl(output)
        elif fmt == "html":
            path = await self.session.export_to_html(output)
        else:
            print("usage: /export [jsonl|html] [path]", file=sys.stderr)
            return
        print(f"exported: {path}")

    def _handle_settings(self, rest: str) -> None:
        """`/settings [show]` or `/settings scope key=value ...` — inspect/edit settings files.

        Values parse as JSON when possible (so `maxRetries=5`,
        `enabled=true`, `compaction={"enabled":false}` all work), else as bare
        strings. Nested objects merge into the file, like pi's nested-field
        persistence.
        """
        parts = rest.split()
        if not parts or parts[0] == "show":
            settings = load_settings(self.cwd)
            if settings.diagnostics:
                for diagnostic in settings.diagnostics:
                    print(f"[settings warning ({diagnostic.scope}): {diagnostic.message}]", file=sys.stderr)
            global_path = os.environ.get("KAREN_SETTINGS_PATH") or str(DEFAULT_SETTINGS_PATH)
            project_path = str(Path(self.cwd) / ".karen" / "settings.json")
            for label, path in (("global", global_path), ("project", project_path)):
                print(f"{label}: {path}")
                try:
                    print(Path(path).read_text(encoding="utf-8-sig").rstrip())
                except OSError:
                    print("  (missing)")
            return

        scope = parts[0]
        if scope not in ("global", "project"):
            print("usage: /settings [show | global|project key=value ...]", file=sys.stderr)
            return

        updates = {}
        for token in parts[1:]:
            key, separator, raw_value = token.partition("=")
            if not separator or not key:
                print(f"usage: /settings {scope} key=value ...", file=sys.stderr)
                return
            try:
                updates[key] = json.loads(raw_value)
            except json.JSONDecodeError:
                updates[key] = raw_value
        if not updates:
            print(f"usage: /settings {scope} key=value ...", file=sys.stderr)
            return
        path = update_settings(updates, scope=scope, cwd=self.cwd)
        print(f"updated {path}: {', '.join(f'{k}={v!r}' for k, v in updates.items())}")
        self._apply_live_settings()

    def _apply_live_settings(self) -> None:
        """Re-read settings and apply the ones a live session honors immediately.

        pi's settings manager is read on every use, so `images.autoResize` and
        `images.blockImages` take effect without a restart; the rest (model,
        shell, compaction) is applied when the session is next opened. Keeping
        `self.settings` current is part of that: `/new` builds its tool set and
        session from the snapshot, so patching only the live session would let
        the next session silently revert the write.
        """
        self.loaded_settings = load_settings(self.cwd)
        settings = self.loaded_settings.settings
        self.settings = settings
        self.templates, self.template_diagnostics = load_all_templates(self.cwd, settings.prompts)
        self._print_settings_diagnostics()
        if self.session is not None:
            self.session.auto_resize_images = image_auto_resize(settings.images)
            self.session.block_images = image_block_images(settings.images)

    async def _print_session_info(self) -> None:
        stats = await self.session.session_stats()
        name = await self.session.session_name()
        tip = await self.session.branch_tip_id()
        tokens = stats["tokens"]
        print(f"id: {stats['sessionId']}")
        print(f"name: {name or '(unnamed)'}")
        print(f"file: {stats['sessionFile']}")
        print(f"cwd: {self.cwd}")
        print(f"branch tip: {tip[-8:] if tip else '(none)'}")
        print(
            f"messages: {stats['totalMessages']} "
            f"({stats['userMessages']} user, {stats['assistantMessages']} assistant, "
            f"{stats['toolCalls']} tool calls, {stats['toolResults']} tool results)"
        )
        print(
            f"tokens: in {tokens['input']} out {tokens['output']} "
            f"cache read/write {tokens['cacheRead']}/{tokens['cacheWrite']} "
            f"total {tokens['total']} | cost {stats['cost']:.4f}"
        )
        print(f"context: ~{self.session.estimate_tokens()} tokens")

    # -- modes ------------------------------------------------------------------

    async def run_print(self, prompts) -> int:
        """Headless mode (pi's runPrintMode): run the prompt(s) sequentially.

        text mode: stdout gets the final assistant message's text only (exit 1
        on an error/aborted final message). json mode: stdout gets the session
        header followed by one JSON event per line (exit 1 only on exceptions).
        """
        await self._open_session(self.fresh)
        if self.output_mode == "json":
            header = self.session.session_header()
            if header is not None:
                print(json.dumps(header, ensure_ascii=False), flush=True)
        exit_code = 0
        try:
            for prompt in prompts:
                await self.session.prompt(prompt)
        except Exception as error:
            print(str(error), file=sys.stderr)
            exit_code = 1
        if exit_code == 0 and self.output_mode == "text":
            messages = self.session.agent.state.messages
            final = messages[-1] if messages else None
            if getattr(final, "role", None) == "assistant":
                if final.stop_reason in ("error", "aborted"):
                    print(final.error_message or f"Request {final.stop_reason}", file=sys.stderr)
                    exit_code = 1
                else:
                    for block in final.content:
                        text = getattr(block, "text", None)
                        if text is not None:
                            sys.stdout.write(f"{text}\n")
            sys.stdout.flush()
        await self.session.close()
        return exit_code

    async def repl(self) -> int:
        await self._open_session(self.fresh)
        tool_names = " ".join(tool.name for tool in self.session.tools)
        print(f"model: {self.provider}/{self.model.id} | cwd: {self.cwd} | tools: {tool_names}")
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
            kind, name, rest = parse_command(
                line, {t.name for t in self.templates}, {s.name for s in self.skills}
            )
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
            if kind == "skills":
                if not self.skills:
                    print("no skills (add SKILL.md files to .karen/skills or ~/.karen/skills)")
                for skill in self.skills:
                    print(f"  /{skill.name}  {skill.description}")
                continue
            if kind == "new":
                await self._open_session(fresh=True)
                continue
            if kind == "compact":
                await self.session.run_compaction("manual", custom_instructions=rest or None)
                continue
            if kind == "retry":
                self._handle_retry(rest)
                continue
            if kind in ("tree", "fork", "clone", "sessions", "resume", "name", "session"):
                try:
                    await self._handle_navigation(kind, rest)
                except Exception as error:  # bad ids, failed forks — keep the REPL alive
                    print(f"[error: {error}]", file=sys.stderr)
                continue
            if kind == "thinking":
                try:
                    await self._handle_thinking(rest)
                except Exception as error:  # bad level, transcript write failure
                    print(f"[error: {error}]", file=sys.stderr)
                continue
            if kind == "export":
                try:
                    await self._handle_export(rest)
                except (OSError, ValueError) as error:  # unwritable path — keep the REPL alive
                    print(f"[error: {error}]", file=sys.stderr)
                continue
            if kind == "settings":
                try:
                    self._handle_settings(rest)
                except (OSError, ValueError) as error:  # malformed file, unwritable path
                    print(f"[error: {error}]", file=sys.stderr)
                continue
            if kind == "template":
                template = next(t for t in self.templates if t.name == name)
                line = format_prompt_template_invocation(template, parse_command_args(rest))
            elif kind == "skill":
                skill = next(s for s in self.skills if s.name == name)
                line = format_skill_invocation(skill, rest or None)
            try:
                await self.session.prompt(line)
                print(f"[context ~{self.session.estimate_tokens()} tokens; tools used: {self.tool_counts or '{}'}]")
            except KeyboardInterrupt:
                self.session.abort()
                await self.session.wait_for_idle()
                print("\n[interrupted]")
            except Exception as error:
                print(f"[error: {error}]", file=sys.stderr)
        await self.session.close()
        return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="karen", description="karen — AI coding assistant")
    parser.add_argument("-p", "--print", dest="print_flag", action="store_true",
                        help="headless print mode: run the prompt(s), print the reply, exit")
    parser.add_argument("--mode", choices=["text", "json", "rpc"], default="text",
                        help="headless output mode: text (default, final reply only), "
                             "json (event stream), or rpc (interactive JSON command protocol)")
    parser.add_argument("--cwd", default=os.getcwd(), help="working directory for tools and session resume")
    parser.add_argument("--model", default=None,
                        help="model id (default: KAREN_MODEL, then settings defaultModel, then deepseek-v4-pro)")
    parser.add_argument("--provider", default=None,
                        help="model provider (default: KAREN_PROVIDER, then settings defaultProvider, then deepseek)")
    parser.add_argument("--new", action="store_true", help="start a fresh session instead of resuming")
    parser.add_argument("--export", metavar="FILE", dest="export_file",
                        help="export a session file to HTML and exit (an optional PROMPT arg is the output path)")
    parser.add_argument("messages", nargs="*", metavar="PROMPT",
                        help="prompt(s) for headless mode; several run sequentially")
    args = parser.parse_args(argv)
    cwd = os.path.abspath(args.cwd)
    if args.export_file:
        # pi's `--export <file> [output]`: no model or credentials needed.
        from .session_export import export_html_from_file

        try:
            result = export_html_from_file(args.export_file, args.messages[0] if args.messages else None)
        except Exception as error:  # missing/unreadable file, bad output path
            print(f"Error: {error}", file=sys.stderr)
            return 1
        print(f"Exported to: {result}")
        return 0
    loaded_settings = load_settings(cwd)
    provider = (
        args.provider
        or os.environ.get("KAREN_PROVIDER")
        or loaded_settings.settings.default_provider
        or "deepseek"
    )
    model_id = (
        args.model
        or os.environ.get("KAREN_MODEL")
        or loaded_settings.settings.default_model
        or DEFAULT_MODEL_ID
    )
    if args.mode == "rpc":
        # RPC mode is interactive: commands arrive on stdin over a long-lived
        # session, so positional prompts don't apply and stdout is reserved
        # for the JSON stream.
        cli = KarenCli(cwd=cwd, model_id=model_id, fresh=args.new, quiet_tools=True,
                       provider=provider, output_mode="rpc", loaded_settings=loaded_settings)
        session = None

        async def _run_rpc() -> int:
            nonlocal session
            await cli._open_session(args.new)
            session = cli.session
            return await run_rpc_mode(session)

        try:
            return asyncio.run(_run_rpc())
        finally:
            async def _close() -> None:
                if session is not None:
                    await session.close()

            asyncio.run(_close())
    headless = args.print_flag or bool(args.messages) or args.mode == "json"
    cli = KarenCli(cwd=cwd, model_id=model_id, fresh=args.new,
                   quiet_tools=headless, provider=provider, output_mode=args.mode,
                   loaded_settings=loaded_settings)
    if headless:
        return asyncio.run(cli.run_print(args.messages))
    return asyncio.run(cli.repl())


if __name__ == "__main__":
    sys.exit(main())

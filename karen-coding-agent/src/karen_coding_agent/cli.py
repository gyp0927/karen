"""`karen` — the CLI coding assistant (M1: REPL + print mode; M3: JSON mode;
M5: RPC mode; M6: structured system prompt, context files and skills; M7:
session navigation — tree, fork, clone, switch; M8: auto-retry; M10:
extensions — `--extension`, the settings `extensions` list and `/name`
extension commands).

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
from typing import Any, Optional, Tuple

from karen_ai import CreateModelsOptions, JsonFileCredentialStore, create_models
from karen_ai.providers import deepseek_provider
from karen_agent import (
    format_prompt_template_invocation,
    format_skill_invocation,
    load_prompt_templates,
    parse_command_args,
)

from .agent_session import AgentSession
from .extensions import (
    CommandContext,
    ExtensionHost,
    load_extensions_async,
)
from .extensions import events as extension_events
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
    mcp_server_configs_from_settings,
    resolve_default_tool_names,
    retry_policy_from_wire,
    update_settings,
)
from .tools import create_default_tools
from .tui import LiveTui, supports_tty

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
  /<name> [args]        invoke a prompt template, skill, or extension command
Anything else is sent to the model."""


#: How often `!command` output may repaint the alt screen (seconds). The
#: command's own output is the only thing that streams between keystrokes, and
#: a full-screen repaint per chunk of a flooded stdout would be wasteful.
_BASH_PAINT_INTERVAL_SECONDS = 0.05


def parse_command(line: str, template_names, skill_names=(), extension_names=()):
    """Route one input line. Returns (kind, name, arg_string).

    kind: "quit" | "help" | "new" | "compact" | "retry" | "tree" | "fork"
          | "clone" | "sessions" | "resume" | "name" | "session"
          | "templates" | "skills" | "extension" | "template" | "skill" | "prompt"

    One `/name` namespace, like pi's slash commands; the precedence is
    built-ins, then extension commands, then prompt templates, then skills
    (templates win over skills when a name is defined as both).
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
    if name in extension_names:
        return "extension", name, rest
    if name in template_names:
        return "template", name, rest
    if name in skill_names:
        return "skill", name, rest
    return "prompt", None, line  # unknown slash command: send verbatim


def resolve_interactive_mode(
    *,
    tui_flag: bool,
    repl_flag: bool,
    setting: Optional[bool],
    interactive: bool,
) -> Tuple[bool, str]:
    """Decide whether `karen` opens the alt-screen TUI or the plain REPL.

    Precedence: an explicit `--tui`/`--repl` wins, then the settings file's
    `tui` key, then the default (TUI). The TUI needs a real terminal — the
    alt-screen escapes go to stdout and the Windows key path wants a native
    console handle — so without one the caller gets the plain REPL plus a
    `notice` explaining why. Returns `(use_tui, notice)`; `notice` is empty
    when there is nothing to explain.
    """
    if tui_flag:
        wanted = True
    elif repl_flag:
        wanted = False
    elif setting is not None:
        wanted = setting
    else:
        wanted = True
    if not wanted:
        return False, ""
    if not interactive:
        return False, "TUI needs an interactive terminal; using the plain REPL"
    return True, ""


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
                 loaded_settings: "LoadedSettings | None" = None,
                 extension_paths=()) -> None:
        self.cwd = cwd
        self.provider = provider
        self.model_id = model_id
        self.fresh = fresh
        self.quiet_tools = quiet_tools  # headless: tool chatter goes to stderr
        self.output_mode = output_mode  # "text" (final reply) | "json" (event stream) | "rpc"
        self.loaded_settings = loaded_settings if loaded_settings is not None else load_settings(cwd)
        self.settings = self.loaded_settings.settings
        self.models, self.auth_source = build_models()
        if provider == "deepseek":
            self.models.set_provider(deepseek_provider())
        self.model = self.resolve_model()
        self.agent_dir = DEFAULT_AGENT_DIR
        self.tui: "Any" = None  # set when running in alt-screen TUI mode
        # Extension runtime state. Loading is deferred to the first
        # `_open_session` (factories may be `async def`, and `__init__` may run
        # inside an event loop in tests, where the sync loader is illegal);
        # the host is built now, with a notify sink that reads the face
        # dynamically — `self.tui` appears when the TUI starts.
        self.extension_paths = list(extension_paths)
        self.extension_host = ExtensionHost(
            settings=self.settings,
            mode=self._face(),
            has_ui=self._face() in ("repl", "tui"),
            notify=self._extension_notify,
        )
        self.loaded_extensions = None
        self.extension_runner = None
        self._extension_prompt_paths: list = []
        self._extension_skill_paths: list = []
        self.templates, self.template_diagnostics = self._load_templates()
        self._reload_skills()
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

    def _face(self) -> str:
        """The extension-facing mode name for the current output configuration."""
        if self.output_mode == "rpc":
            return "rpc"
        if self.quiet_tools:
            return "print"
        return "tui" if self.tui is not None else "repl"

    def _load_templates(self):
        """Prompt templates: conventional dirs, settings `prompts`, extension paths."""
        extra = [*(self.settings.prompts or []), *self._extension_prompt_paths]
        return load_all_templates(self.cwd, extra)

    def _reload_skills(self) -> None:
        result = load_project_skills(self.cwd, self.agent_dir, extra_dirs=self._extension_skill_paths)
        self.skills = result.skills
        self.skill_diagnostics = result.diagnostics

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
        if self.tui is not None:
            self.tui.app.ingest_agent_event(event, self.session)
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
        if self.tui is not None:
            # Every session event renders inside the viewport; printing to
            # stdout/stderr here would paint over the alt-screen frame.
            self.tui.app.ingest_session_event(event)
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

    # -- extensions (M10) -------------------------------------------------------

    def _extension_notify(self, message: str, level: str = "info") -> None:
        """Route an extension's user-facing message by face (pi's `ui.notify`).

        The TUI gets a transcript notice; the plain REPL prints info to stdout
        and anything graver to stderr; headless modes keep stdout clean for
        the reply / event stream / protocol, so all of it goes to stderr.
        """
        if self.tui is not None:
            self.tui.app.add_notice(message, error=(level != "info"))
            return
        interactive = self.output_mode == "text" and not self.quiet_tools
        print(message, file=sys.stdout if interactive and level == "info" else sys.stderr)

    def _on_extension_error(self, error) -> None:
        """The runner's error sink: a handler or factory failed at runtime."""
        self._extension_notify(
            f"[extension error: {error.event} ({error.extension_path}): {error.message}]",
            level="error",
        )

    async def _ensure_extensions(self) -> None:
        """Load extensions once, on the first session open.

        Deferred out of `__init__` because factories may be `async def` (the
        loader awaits them) and a CLI built inside a running loop — the tests
        do this — cannot drive the sync loader's `asyncio.run`. The runner
        then outlives sessions: `/new` re-opens rebind it, never reload it.
        """
        if self.loaded_extensions is not None:
            return
        self.loaded_extensions = await load_extensions_async(
            cwd=self.cwd,
            agent_dir=self.agent_dir,
            settings_paths=self.settings.extensions or (),
            cli_paths=self.extension_paths,
            host=self.extension_host,
            settings=self.settings,
            on_error=self._on_extension_error,
        )
        self.extension_runner = self.loaded_extensions.runner
        for diagnostic in self.loaded_extensions.diagnostics:
            level = "error" if diagnostic.severity == "error" else "warning"
            for line in f"[extension {diagnostic.severity}: {diagnostic.path}: {diagnostic.message}]".splitlines():
                self._extension_notify(line, level=level)
        # resources_discover is the CLI's own event — nobody else fires it, and
        # its contributions feed the same loaders templates/skills come from.
        runner = self.extension_runner
        if runner.has_handlers("resources_discover"):
            discovered = await runner.emit_resources_discover(
                extension_events.ResourcesDiscoverEvent(cwd=self.cwd, reason="startup")
            )
            self._extension_prompt_paths = discovered.get("prompt_paths") or []
            self._extension_skill_paths = discovered.get("skill_paths") or []
            # theme_paths are accepted for pi parity; karen has no theme loader.
            if self._extension_prompt_paths:
                self.templates, self.template_diagnostics = self._load_templates()
            if self._extension_skill_paths:
                self._reload_skills()

    def _extension_command_names(self):
        # getattr: test doubles build a bare KarenCli (object.__new__, no
        # __init__) carrying only what the input router reads.
        runner = getattr(self, "extension_runner", None)
        return set(runner.commands()) if runner is not None else set()

    def _extension_command_context(self) -> CommandContext:
        """A command handler's reach: the live session, notify, and composition."""
        host = self.extension_host

        async def run_extension(name: str, args: Optional[str] = None) -> None:
            await self._run_extension_command(name, args or "")

        return CommandContext(
            session=self.session,
            notify=lambda message: host.notify(message),
            run_extension=run_extension,
        )

    async def _run_extension_command(self, name: str, rest: str) -> bool:
        """Dispatch `/name args` to an extension (pi's `_tryExecuteExtensionCommand`).

        The runner reports a failing handler to the error sink and still counts
        the command as handled; False means the name was never registered, so
        the caller falls back to ordinary prompt handling, like pi.
        """
        runner = getattr(self, "extension_runner", None)
        if runner is None:
            return False
        return await runner.run_command(name, rest or None, self._extension_command_context())

    def _extension_summary_line(self) -> "str | None":
        """The one-line startup summary of what the extension runtime loaded."""
        loaded = self.loaded_extensions
        if not loaded or not loaded.extensions:
            return None
        line = f"extensions: {len(loaded.extensions)} loaded"
        commands = self._extension_command_names()
        if commands:
            line += f" | commands: {' '.join('/' + name for name in sorted(commands))}"
        return line

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
        replacing = self.session is not None
        if replacing:
            # The only path that swaps a live session out is /new, which is
            # also the shutdown reason pi reports to extension handlers.
            await self.session.close(reason="new" if fresh else "quit")
        await self._ensure_extensions()
        settings = self.settings
        sessions_root = None
        if settings.session_dir and not os.environ.get("KAREN_SESSIONS_ROOT"):
            sessions_root = settings.session_dir
        tools = self._select_default_tools(settings.default_tools)
        retry_policy = retry_policy_from_wire(settings.retry)

        # MCP servers from settings: connect each, expose their tools to the
        # agent. A server that fails to connect is skipped (see McpToolManager).
        mcp_manager = None
        if settings.mcp_servers:
            from karen_coding_agent.mcp import McpToolManager

            mcp_configs = mcp_server_configs_from_settings(settings)
            if mcp_configs:
                mcp_manager = McpToolManager(configs=mcp_configs)
                await mcp_manager.connect()

        self.session = AgentSession(
            cwd=self.cwd,
            models=self.models,
            model=self.model,
            fresh=fresh,
            listener=self._on_session_event,
            sessions_root=sessions_root,
            tools=tools,
            system_prompt_sections=self._build_prompt_sections(
                [*tools, *mcp_manager.tools()] if mcp_manager else tools
            ),
            shell_path=settings.shell_path,
            shell_command_prefix=settings.shell_command_prefix,
            auto_resize_images=image_auto_resize(settings.images),
            block_images=image_block_images(settings.images),
            compaction_settings=compaction_settings_from_wire(settings.compaction)
            if settings.compaction is not None
            else None,
            retry_policy=retry_policy,
            mcp_manager=mcp_manager,
            extensions=self.extension_runner,
        )
        await self.session.open(reason="new" if replacing and fresh else None)
        self.session.subscribe(self._on_agent_event)
        self.tool_counts = {}
        restored = len(self.session.agent.state.messages) - 1  # minus the seeded system message
        if restored:
            self._tool_line(f"context restored: {restored} messages, ~{self.session.estimate_tokens()} tokens")

    async def _shutdown_session(self, *, reason: str = "quit") -> None:
        """Close the live session, surviving cancellation (Ctrl-C).

        Every mode calls this in a `finally`, because the interesting exits are
        the ones that never reach an ordinary statement: `asyncio.run` cancels
        the main task on Ctrl-C, and an exception in the TUI's key loop skips
        the rest of `repl_tui`. Cleanup still has to run — `close()` dispatches
        the extension `session_shutdown` and flushes the transcript — so a
        cancellation that lands *during* cleanup is absorbed and the original
        one re-raised once the session is down. The cancellation state is
        cleared first so the awaits inside `close()` actually get to run.
        """
        session = self.session
        if session is None:
            return
        task = asyncio.current_task()
        cancelled = task.cancelling() if task is not None and hasattr(task, "cancelling") else 0
        if cancelled and hasattr(task, "uncancel"):
            for _ in range(cancelled):
                task.uncancel()
        try:
            await session.close(reason=reason)
        except asyncio.CancelledError:
            # A second Ctrl-C while shutting down wins: propagate it.
            raise
        finally:
            if cancelled and task is not None:
                for _ in range(cancelled):
                    task.cancel()

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
        self.templates, self.template_diagnostics = self._load_templates()
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
        The session is closed on the way out of either — including a cancelled
        run (Ctrl-C), which is what the `run_print` repro in `.scratch` pinned.
        """
        await self._open_session(self.fresh)
        if self.output_mode == "json":
            header = self.session.session_header()
            if header is not None:
                print(json.dumps(header, ensure_ascii=False), flush=True)
        exit_code = 0
        try:
            try:
                for prompt in prompts:
                    await self.session.prompt(prompt)
            except Exception as error:
                print(str(error), file=sys.stderr)
                exit_code = 1
        finally:
            await self._shutdown_session()
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
        return exit_code

    async def repl(self) -> int:
        return await self._run_until_closed(self._repl_loop)

    async def _run_until_closed(self, body) -> int:
        """Run a REPL/TUI body with the session closed on every exit.

        Both faces reach their end in more than one way: the loop returns, the
        user hits Ctrl-C (the main task is cancelled), or something inside the
        view raises. The plain `repl()` finishes with two statements that a
        cancellation never reaches, so the close lives here — see
        `_shutdown_session` for the cancellation handling.
        """
        await self._open_session(self.fresh)
        try:
            return await body()
        finally:
            await self._shutdown_session()

    async def _repl_loop(self) -> int:
        tool_names = " ".join(tool.name for tool in self.session.tools)
        print(f"model: {self.provider}/{self.model.id} | cwd: {self.cwd} | tools: {tool_names}")
        print("type /help for commands")
        summary = self._extension_summary_line()
        if summary:
            print(summary)
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
                line,
                {t.name for t in self.templates},
                {s.name for s in self.skills},
                self._extension_command_names(),
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
            if kind == "extension":
                try:
                    handled = await self._run_extension_command(name, rest)
                except Exception as error:  # keep the REPL alive
                    print(f"[error: {error}]", file=sys.stderr)
                    continue
                if handled:
                    continue
                # The name was never registered after all: fall through and
                # send the line to the model, exactly like pi.
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
        return 0

    async def repl_tui(self) -> int:
        """Alt-screen TUI (pi's interactive mode, hand-rolled viewport).

        Opens the session exactly like the plain REPL, then hands control to
        the TUI. Agent + session events are rerouted into the `Transcript`
        (via `self.tui`), the editor's submitted text is dispatched through
        the same command router the REPL uses, and the async run loop keeps
        reading keys while a prompt streams.
        """
        if not supports_tty():
            print("TUI needs an interactive terminal; using the plain REPL", file=sys.stderr)
            return await self.repl()
        # Create the TUI before opening the session so the session-opened event
        # lands in the transcript instead of on stdout (redraw is gated on
        # `start`, which happens once the alt screen is entered).
        self.tui = LiveTui()
        # The face is known now: extension contexts and notifications are TUI's.
        self.extension_host.mode = "tui"
        self.extension_host.has_ui = True
        try:
            await self._open_session(self.fresh)
            tool_count = len(self.session.tools)
            self.tui.app.set_status_provider(
                lambda: f"model: {self.provider}/{self.model.id} | tools: {tool_count} | cwd: {self.cwd}"
            )
            summary = self._extension_summary_line()
            if summary:
                self.tui.app.add_notice(summary)

            async def _prompt_handler(app, text: str) -> None:
                await self._dispatch_tui_input(app, text)

            try:
                await self.tui.run(_prompt_handler)
            finally:
                self.tui = None
        finally:
            # The TUI is down by now; the session still needs its shutdown on
            # the paths that skip the tail of this method (Ctrl-C, a raise).
            await self._shutdown_session()
        return 0

    async def _dispatch_tui_input(self, app, text: str) -> None:
        """Route one submitted editor line: shell bypass, prompt, or slash command."""
        if text.startswith("!"):
            # pi's `!command` shell bypass. The runner echoes every submitted
            # line as a user message before dispatch; a shell run renders as a
            # bash block instead (it carries the command itself), so drop the
            # echo rather than showing the command twice.
            app.transcript.drop_pending_user(text)
            await self._run_tui_bash(app, text[1:])
            return
        kind, name, rest = parse_command(
            text,
            {t.name for t in self.templates},
            {s.name for s in self.skills},
            self._extension_command_names(),
        )
        if kind == "quit":
            app.transcript.resolve_pending()
            app.running = False
            return
        if kind in ("prompt", "template", "skill"):
            if kind == "template":
                template = next(t for t in self.templates if t.name == name)
                prompt_text = format_prompt_template_invocation(template, parse_command_args(rest))
            elif kind == "skill":
                skill = next(s for s in self.skills if s.name == name)
                prompt_text = format_skill_invocation(skill, rest or None)
            else:
                prompt_text = text
            await self._run_tui_prompt(app, prompt_text)
            return
        # Everything below is a command, not a prompt: it never becomes a run,
        # so its transcript line must not keep the "(pending)" marker waiting
        # for an `agent_start` that will not come.
        app.transcript.resolve_pending()
        if kind == "new":
            # Reset first: the fresh session's session_opened notice should
            # survive the reset, not be erased by it.
            app.transcript.reset()
            try:
                await self._open_session(fresh=True)
            except Exception as error:
                self.tui.app.add_notice(f"[error: {error}]", error=True)
            return
        try:
            await self._handle_navigation_or_settings(kind, name, rest)
        except Exception as error:
            self.tui.app.add_notice(f"[error: {error}]", error=True)

    async def _run_tui_prompt(self, app, prompt_text: str) -> None:
        """Run one prompt, keeping the TUI alive and responsive throughout."""
        try:
            await self.session.prompt(prompt_text)
        except KeyboardInterrupt:
            self.session.abort()
            await self.session.wait_for_idle()
        except Exception as error:
            self.tui.app.add_notice(f"[error: {error}]", error=True)

    async def _run_tui_bash(self, app, raw_command: str) -> None:
        """`!command` in the TUI: run it through `AgentSession.execute_bash`.

        The command runs outside the agent loop, its output streams into the
        transcript block as it arrives, and the result is recorded in the
        session so the model sees it on the next turn (pi's `recordBashResult`).
        """
        command = raw_command.strip()
        if not command:
            app.add_notice("usage: !<shell command>")
            return
        transcript = app.transcript
        entry = transcript.on_bash_start(command)
        last_paint = 0.0

        def _on_chunk(delta: str) -> None:
            nonlocal last_paint
            transcript.on_bash_chunk(entry, delta)
            # A command that floods stdout would otherwise repaint the whole
            # screen per chunk; throttle, and always paint again at the end.
            now = time.monotonic()
            if now - last_paint >= _BASH_PAINT_INTERVAL_SECONDS:
                last_paint = now
                app.redraw()

        try:
            result = await self.session.execute_bash(command, on_chunk=_on_chunk)
        except asyncio.CancelledError:
            # The TUI is going away (ctrl+c). Killing the child is not this
            # frame's job: the cancel has already unwound through
            # `execute_bash`, which drops its controller, so the kill happens in
            # `run_shell_command`'s own `finally` (nothing else may outlive the
            # app). The loose end left here is the block an answer was
            # streaming into — no `agent_end` is coming to close it.
            transcript.finish_assistant()
            raise
        except Exception as error:
            transcript.on_bash_end(entry, error=str(error))
        else:
            transcript.on_bash_end(entry, result=result)
        app.redraw()

    async def _handle_navigation_or_settings(self, kind: str, name: str, rest: str) -> None:
        """TUI-side dispatcher for slash commands that aren't prompt-shaped.

        Every branch's output is captured and reposted as transcript notice
        lines: printing while the alt screen is live would paint over the
        frame.
        """
        if kind in ("tree", "fork", "clone", "sessions", "resume", "name", "session"):
            await self._run_captured(
                lambda: self._handle_navigation(kind, rest)
            )
        elif kind == "extension":
            await self._run_captured(lambda: self._run_extension_command(name, rest))
        elif kind == "retry":
            await self._run_captured(lambda: self._handle_retry(rest))
        elif kind == "thinking":
            await self._run_captured(lambda: self._handle_thinking(rest))
        elif kind == "export":
            await self._run_captured(lambda: self._handle_export(rest))
        elif kind == "settings":
            await self._run_captured(lambda: self._handle_settings(rest))
        elif kind == "compact":
            await self._run_captured(
                lambda: self.session.run_compaction("manual", custom_instructions=rest or None)
            )
        elif kind == "templates":
            if not self.templates:
                self.tui.app.add_notice("no templates (add .md files to .karen/prompts)")
            for template in self.templates:
                self.tui.app.add_notice(f"/{template.name}  {template.description or ''}")
        elif kind == "skills":
            if not self.skills:
                self.tui.app.add_notice("no skills (add SKILL.md files to .karen/skills)")
            for skill in self.skills:
                self.tui.app.add_notice(f"/{skill.name}  {skill.description}")
        elif kind == "help":
            for line in HELP_TEXT.splitlines():
                self.tui.app.add_notice(line)
            # TUI-only: the shell bypass has no plain-REPL counterpart.
            self.tui.app.add_notice("  !<command>            run a shell command in this session")

    async def _run_captured(self, produce) -> None:
        """Run `produce()` (sync or async) and post its stdout/stderr as
        transcript notices, one per non-empty line."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                await _maybe_await(produce())
        except Exception as error:
            self.tui.app.add_notice(f"[error: {error}]", error=True)
            return
        for line in buffer.getvalue().splitlines():
            if line.strip():
                self.tui.app.add_notice(line)


async def _maybe_await(value):
    """Await `value` when it is awaitable, else return it (sync handlers)."""
    if hasattr(value, "__await__"):
        return await value
    return value


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
    ui = parser.add_mutually_exclusive_group()
    ui.add_argument("--tui", action="store_true",
                    help="interactive alt-screen TUI (the default when stdin and stdout are a terminal)")
    ui.add_argument("--repl", action="store_true",
                    help="plain line-based REPL instead of the alt-screen TUI")
    parser.add_argument("--export", metavar="FILE", dest="export_file",
                        help="export a session file to HTML and exit (an optional PROMPT arg is the output path)")
    parser.add_argument("--extension", dest="extensions", action="append", default=[], metavar="PATH",
                        help="load an extension file or directory (repeatable); loads after the "
                             "conventional directories and the settings `extensions` list")
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
                       provider=provider, output_mode="rpc", loaded_settings=loaded_settings,
                       extension_paths=args.extensions)
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
                # `run_rpc_mode` closes whatever session was live at EOF itself
                # — a `new_session` command rebinds its holder, so the startup
                # session is not necessarily the live one. This is the guard for
                # a failure before that loop was reached; when the loop already
                # closed its session this is a no-op.
                if session is not None:
                    await session.close()

            asyncio.run(_close())
    headless = args.print_flag or bool(args.messages) or args.mode == "json"
    cli = KarenCli(cwd=cwd, model_id=model_id, fresh=args.new,
                   quiet_tools=headless, provider=provider, output_mode=args.mode,
                   loaded_settings=loaded_settings, extension_paths=args.extensions)
    if headless:
        return asyncio.run(cli.run_print(args.messages))
    use_tui, notice = resolve_interactive_mode(
        tui_flag=args.tui,
        repl_flag=args.repl,
        setting=loaded_settings.settings.tui,
        interactive=supports_tty(),
    )
    if notice:
        print(notice, file=sys.stderr)
    if use_tui:
        return asyncio.run(cli.repl_tui())
    return asyncio.run(cli.repl())


if __name__ == "__main__":
    sys.exit(main())

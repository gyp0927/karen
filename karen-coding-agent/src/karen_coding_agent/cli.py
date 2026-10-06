"""`karen` — the CLI coding assistant (M1: REPL + print mode; M3: JSON mode).

Interactive:
    karen [--cwd PATH] [--model ID] [--new]

Headless (print mode, pi's runPrintMode):
    karen -p "summarize this repo" [--cwd PATH]   # final reply text on stdout
    karen "one prompt" "another prompt"           # prompts run sequentially
    karen --mode json "prompt"                    # JSON event stream on stdout

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
from pathlib import Path

from karen_ai import CreateModelsOptions, JsonFileCredentialStore, create_models
from karen_ai.providers import deepseek_provider
from karen_agent import format_prompt_template_invocation, load_prompt_templates, parse_command_args

from .agent_session import AgentSession
from .json_events import to_json_event
from .settings import (
    LoadedSettings,
    compaction_settings_from_wire,
    load_settings,
    resolve_default_tool_names,
)
from .tools import create_default_tools

DEFAULT_MODEL_ID = "deepseek-v4-pro"
DEFAULT_CREDENTIALS = Path.home() / ".karen" / "credentials.json"

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
        self.templates, self.template_diagnostics = load_all_templates(cwd, self.settings.prompts)
        self.session: AgentSession | None = None
        self.tool_counts = {}
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
            kind = "resumed" if event["resumed"] else "new"
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

    # -- session ---------------------------------------------------------------

    def _select_default_tools(self, entries):
        """Apply a settings `defaultTools` list (pi's resolveDefaultTools)."""
        all_tools = create_default_tools(
            self.cwd,
            shell_path=self.settings.shell_path,
            shell_command_prefix=self.settings.shell_command_prefix,
        )
        names = resolve_default_tool_names(entries, [tool.name for tool in all_tools])
        by_name = {tool.name: tool for tool in all_tools}
        selected = []
        for name in names:
            tool = by_name.get(name)
            if tool is None:
                print(f"[settings warning: unknown tool in defaultTools: {name}]", file=sys.stderr)
                continue
            selected.append(tool)
        return selected

    async def _open_session(self, fresh: bool) -> None:
        if self.session is not None:
            await self.session.close()
        settings = self.settings
        sessions_root = None
        if settings.session_dir and not os.environ.get("KAREN_SESSIONS_ROOT"):
            sessions_root = settings.session_dir
        self.session = AgentSession(
            cwd=self.cwd,
            models=self.models,
            model=self.model,
            fresh=fresh,
            listener=self._on_session_event,
            sessions_root=sessions_root,
            tools=self._select_default_tools(settings.default_tools)
            if settings.default_tools is not None
            else None,
            shell_path=settings.shell_path,
            shell_command_prefix=settings.shell_command_prefix,
            compaction_settings=compaction_settings_from_wire(settings.compaction)
            if settings.compaction is not None
            else None,
        )
        await self.session.open()
        self.session.subscribe(self._on_agent_event)
        self.tool_counts = {}
        restored = len(self.session.agent.state.messages) - 1  # minus the seeded system message
        if restored:
            self._tool_line(f"context restored: {restored} messages, ~{self.session.estimate_tokens()} tokens")

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
                await self._open_session(fresh=True)
                continue
            if kind == "compact":
                await self.session.run_compaction("manual", custom_instructions=rest or None)
                continue
            if kind == "template":
                template = next(t for t in self.templates if t.name == name)
                line = format_prompt_template_invocation(template, parse_command_args(rest))
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
    parser.add_argument("--mode", choices=["text", "json"], default="text",
                        help="headless output mode: text (default, final reply only) or json (event stream)")
    parser.add_argument("--cwd", default=os.getcwd(), help="working directory for tools and session resume")
    parser.add_argument("--model", default=None,
                        help="model id (default: KAREN_MODEL, then settings defaultModel, then deepseek-v4-pro)")
    parser.add_argument("--provider", default=None,
                        help="model provider (default: KAREN_PROVIDER, then settings defaultProvider, then deepseek)")
    parser.add_argument("--new", action="store_true", help="start a fresh session instead of resuming")
    parser.add_argument("messages", nargs="*", metavar="PROMPT",
                        help="prompt(s) for headless mode; several run sequentially")
    args = parser.parse_args(argv)
    cwd = os.path.abspath(args.cwd)
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
    headless = args.print_flag or bool(args.messages) or args.mode == "json"
    cli = KarenCli(cwd=cwd, model_id=model_id, fresh=args.new,
                   quiet_tools=headless, provider=provider, output_mode=args.mode,
                   loaded_settings=loaded_settings)
    if headless:
        return asyncio.run(cli.run_print(args.messages))
    return asyncio.run(cli.repl())


if __name__ == "__main__":
    sys.exit(main())

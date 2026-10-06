"""`karen` — the CLI coding assistant (M1: interactive REPL + print mode).

Interactive:
    karen [--cwd PATH] [--model ID] [--new]

Headless (print mode):
    karen -p "summarize this repo" [--cwd PATH]

Piping into the REPL works too (that's how the smokes drive it):
    printf 'hello\n/quit\n' | karen --new
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from karen_ai import CreateModelsOptions, JsonFileCredentialStore, create_models
from karen_ai.providers import deepseek_provider
from karen_agent import format_prompt_template_invocation, load_prompt_templates, parse_command_args

from .agent_session import AgentSession

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


def load_all_templates(cwd: str):
    result = load_prompt_templates(
        [str(Path(cwd) / ".karen" / "prompts"), str(Path.home() / ".karen" / "prompts")]
    )
    return result.prompt_templates, result.diagnostics


class KarenCli:
    """Terminal front-end over `AgentSession`: event printing + the REPL."""

    def __init__(self, cwd: str, model_id: str, fresh: bool, quiet_tools: bool = False,
                 provider: str = "deepseek") -> None:
        self.cwd = cwd
        self.provider = provider
        self.model_id = model_id
        self.fresh = fresh
        self.quiet_tools = quiet_tools  # print mode: tool chatter goes to stderr
        self.models, self.auth_source = build_models()
        if provider == "deepseek":
            self.models.set_provider(deepseek_provider())
        self.model = self.resolve_model()
        self.templates, self.template_diagnostics = load_all_templates(cwd)
        self.session: AgentSession | None = None
        self.tool_counts = {}

    def resolve_model(self):
        model = self.models.get_model(self.provider, self.model_id)
        if model is None:
            raise SystemExit(f"unknown model: {self.provider}/{self.model_id}")
        return model

    # -- printing ---------------------------------------------------------------

    def _tool_line(self, text: str) -> None:
        print(text, file=sys.stderr if self.quiet_tools else sys.stdout)

    def _on_agent_event(self, event, signal) -> None:
        if event.type == "message_update" and event.assistant_message_event.type == "text_delta":
            print(event.assistant_message_event.delta, end="", flush=True)
        elif event.type == "tool_execution_start":
            self.tool_counts[event.tool_name] = self.tool_counts.get(event.tool_name, 0) + 1
            self._tool_line(f"\n[tool ->] {event.tool_name}({format_args_preview(event.args)})")
        elif event.type == "tool_execution_end" and event.is_error:
            text = "".join(getattr(c, "text", "") for c in event.result.content)
            self._tool_line(f"[tool <-] {event.tool_name} ERROR: {text[:200]}")
        elif event.type == "agent_end":
            print()
            final = self.session.agent.state.messages[-1] if self.session else None
            if getattr(final, "stop_reason", None) == "error":
                print(f"[run failed: {final.error_message}]", file=sys.stderr)

    def _on_session_event(self, event) -> None:
        event_type = event.get("type")
        if event_type == "session_opened":
            kind = "resumed" if event["resumed"] else "new"
            print(f"{kind} session {event['session_id']}")
        elif event_type == "compaction_start":
            print(f"compacting ({event['reason']}; the model writes a summary)...")
        elif event_type == "compaction_end":
            if event["compacted"]:
                print(f"compacted ~{event['tokens_before']} tokens")
            elif event.get("detail") == "nothing_to_compact":
                print("nothing to compact")
            else:
                print(f"compaction failed: {event.get('detail')}", file=sys.stderr)
        elif event_type == "overflow_retry":
            print("[context overflow: compacted; retrying the turn]")
        elif event_type == "overflow_give_up":
            print(
                "[context overflow recovery failed after one compact-and-retry attempt; "
                "try reducing context or switching to a larger-context model]",
                file=sys.stderr,
            )

    # -- session ---------------------------------------------------------------

    async def _open_session(self, fresh: bool) -> None:
        if self.session is not None:
            await self.session.close()
        self.session = AgentSession(
            cwd=self.cwd,
            models=self.models,
            model=self.model,
            fresh=fresh,
            listener=self._on_session_event,
        )
        await self.session.open()
        self.session.subscribe(self._on_agent_event)
        self.tool_counts = {}
        restored = len(self.session.agent.state.messages) - 1  # minus the seeded system message
        if restored:
            print(f"context restored: {restored} messages, ~{self.session.estimate_tokens()} tokens")

    # -- modes ------------------------------------------------------------------

    async def run_print(self, prompt_text: str) -> int:
        """Headless single prompt: stream the reply, exit."""
        await self._open_session(self.fresh)
        try:
            await self.session.prompt(prompt_text)
        finally:
            await self.session.close()
        return 0

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
    parser.add_argument("-p", "--print", dest="print_prompt", metavar="PROMPT",
                        help="headless mode: run one prompt, print the reply, exit")
    parser.add_argument("--cwd", default=os.getcwd(), help="working directory for tools and session resume")
    parser.add_argument("--model", default=os.environ.get("KAREN_MODEL", DEFAULT_MODEL_ID))
    parser.add_argument("--provider", default=os.environ.get("KAREN_PROVIDER", "deepseek"))
    parser.add_argument("--new", action="store_true", help="start a fresh session instead of resuming")
    args = parser.parse_args(argv)
    cli = KarenCli(cwd=os.path.abspath(args.cwd), model_id=args.model, fresh=args.new,
                   quiet_tools=bool(args.print_prompt), provider=args.provider)
    if args.print_prompt is not None:
        return asyncio.run(cli.run_print(args.print_prompt))
    return asyncio.run(cli.repl())


if __name__ == "__main__":
    sys.exit(main())

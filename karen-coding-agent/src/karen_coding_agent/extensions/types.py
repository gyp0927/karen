"""Extension-system types, ported from pi (`core/extensions/types.ts`, f29ea3d).

An extension is a Python module exposing a factory::

    from karen_coding_agent.extensions import ToolResult

    def extension(pi):
        pi.register_command("hello", handler=lambda args, ctx: ctx.notify("hi"))

The factory — synchronous or ``async def`` — receives an :class:`ExtensionAPI`
(:mod:`karen_coding_agent.extensions.loader`) and registers capabilities with it
before the session starts. Everything here is the contract an author writes
against: the events (:mod:`…extensions.events` re-exports them), the tool
definition, and the two context objects a handler or command is handed.

Only a subset of pi's surface is ported — tools, commands, lifecycle events, and
the session-control methods an extension can call. UI primitives, shortcuts and
flags, provider/virtual-model registration, cache warming and
``before_provider_*`` events are deliberately absent; the runner emits only what
karen can supply (see ``runner.py`` for the per-event semantics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

__all__ = [
    "Command",
    "CommandContext",
    "CommandHandler",
    "Extension",
    "ExtensionContext",
    "ExtensionError",
    "TOOL_EVENT_NAMES",
    "ToolCallEventResult",
    "ToolDefinition",
    "ToolExecute",
    "ToolResult",
    "ToolResultDict",
    "ToolResultEventResult",
    "ToolError",
    "as_mapping",
    "is_awaitable",
    "maybe_await",
]

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def is_awaitable(value: Any) -> bool:
    """Whether *value* is something an ``await`` can consume."""
    return hasattr(value, "__await__")


async def maybe_await(value: Any) -> Any:
    """Await *value* when it is awaitable, otherwise return it unchanged.

    Handlers may be plain functions or coroutine functions and either may return
    a plain value from an ``async`` body; every dispatch site funnels through
    here so both spellings work everywhere.
    """
    if is_awaitable(value):
        return await value
    return value


def as_mapping(value: Any) -> Optional[Dict[str, Any]]:
    """Normalise a handler's return value to a dict, or ``None``.

    A handler says something by returning a mapping (``{"block": True}``,
    ``{"messages": [...]}``, ...); returning anything else — including ``None``,
    the common case — means "nothing to add".
    """
    if isinstance(value, dict):
        return value
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """A tool run that failed: the model sees the message as an error result.

    Return it from ``execute`` (the wrapper raises it into karen's normal
    tool-error path) or simply ``raise ToolError("…")`` — or any other
    exception — instead.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass
class ToolResultDict:
    """The fields of a successful tool execution (pi's tool result object).

    ``content`` is what the model sees; ``details`` is structured data for
    rendering or later reconstruction; ``usage`` is the cost of any nested model
    calls the tool made, so session totals stay accurate; ``terminate`` asks the
    agent to skip its automatic follow-up once every tool in the finishing batch
    agrees.
    """

    content: List[Any] = field(default_factory=list)
    details: Any = None
    usage: Any = None
    terminate: Optional[bool] = None


#: What ``execute`` may return. A mapping with the :class:`ToolResultDict`
#: fields works too, as does karen-agent's own ``AgentToolResult``.
ToolResult = Union[ToolResultDict, Dict[str, Any], Any]

#: ``execute(tool_call_id, args, signal, on_update)`` — the same four arguments
#: karen-agent hands its own tools, so an extension tool is indistinguishable
#: from a built-in one at the call site.
ToolExecute = Callable[..., Any]


@dataclass
class ToolDefinition:
    """A tool an extension registers for the model (pi's ``ToolDefinition``)."""

    name: str
    description: str
    #: JSON Schema for the arguments, the same shape karen's built-in tools use.
    parameters: Dict[str, Any]
    execute: ToolExecute
    #: Human-facing label for transcripts; defaults to *name*.
    label: Optional[str] = None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@dataclass
class CommandContext:
    """What a command handler can reach (pi's ``ExtensionCommandContext``).

    ``session`` is the live ``AgentSession``; ``notify`` reports to the user
    without entering the transcript; ``run_extension`` runs another extension
    command by name (mirroring ``/name args``), so commands compose.
    """

    session: Any
    notify: Callable[[str], None]
    run_extension: Optional[Callable[[str, Optional[str]], Awaitable[None]]] = None


#: ``handler(args_text, ctx)`` — *args_text* is everything after the command
#: name, unparsed: ``/hello world`` arrives as ``"world"``.
CommandHandler = Callable[[Optional[str], CommandContext], Any]


@dataclass
class Command:
    """A ``/`` command an extension registered."""

    name: str
    handler: CommandHandler
    description: str = ""


# ---------------------------------------------------------------------------
# Registration records
# ---------------------------------------------------------------------------


@dataclass
class ToolCallEventResult:
    """Result of a ``tool_call`` handler: block execution, or stand aside."""

    block: bool = False
    reason: str = ""
    terminate: Optional[bool] = None


@dataclass
class ToolResultEventResult:
    """Field-wise override of a tool result: omitted fields stay as they are.

    The field set is pi's; there is no ``terminate``, because a tool result's
    terminate hint is not something a ``tool_result`` handler can change.
    """

    content: Optional[List[Any]] = None
    details: Any = None
    usage: Any = None
    is_error: Optional[bool] = None


@dataclass
class Extension:
    """One loaded extension module and everything it registered."""

    #: Path as discovered (the file, or the directory entry that resolved).
    path: str
    #: Absolute path of the module file that was executed.
    module_path: str
    #: Which discovery rule found it: "project" | "user" | "settings" | "cli".
    source: str = "user"
    handlers: Dict[str, List[Callable[..., Any]]] = field(default_factory=dict)
    tools: Dict[str, ToolDefinition] = field(default_factory=dict)
    commands: Dict[str, Command] = field(default_factory=dict)


@dataclass
class ExtensionError:
    """A handler (or factory) failure, reported and then stepped over."""

    extension_path: str
    event: str
    message: str


@dataclass
class ExtensionContext:
    """What a handler sees while it runs.

    *mode* is which face the session shows — ``"tui"``, ``"repl"``, ``"print"``
    or ``"rpc"`` — and *has_ui* is true when anything can render at all.
    *signal* is the abort signal of the run in flight (``None`` for commands and
    idle events, exactly like pi); *is_idle* answers whether a run is active.
    *cwd* is the session's working directory.
    """

    session: Any = None
    mode: str = "repl"
    has_ui: bool = False
    cwd: str = "."
    signal: Any = None
    is_idle: Callable[[], bool] = lambda: True
    has_pending_messages: Callable[[], bool] = lambda: False


#: Events that gate a tool call or rewrite its result.
TOOL_EVENT_NAMES = ("tool_call", "tool_result")

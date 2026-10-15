"""karen's extension system — pi's ``core/extensions/`` ported to Python.

An extension is a Python module in ``<cwd>/.karen/extensions/``,
``~/.karen/extensions/``, or wherever ``settings.extensions`` / ``--extension``
points, exposing a factory::

    def extension(pi):
        pi.register_tool(name="...", description="...", parameters={...}, execute=run)
        pi.register_command("hello", handler=lambda args, ctx: ctx.notify("hi"))
        pi.on("tool_call", lambda event, ctx: ...)

The factory receives an :class:`ExtensionAPI` and runs before the session
starts. Handlers and factories may be sync or ``async``.

The pieces below are the whole public surface: the events handlers receive, the
result shapes they return, the API they register with, and — for the host side —
the loader, the runner, and the session-bound :class:`ExtensionHost`.

What pi can do that this does not (yet): terminal-UI primitives, shortcuts and
flags, provider/virtual-model registration, ``before_provider_*`` and cache
events, nested tool calls, and ``project_trust``. The runner only emits events
karen can supply; see ``events.py`` and ``runner.py`` for the exact contract.
"""

from .events import (
    EVENT_NAMES,
    AgentEndEvent,
    AgentErrorEvent,
    AgentStartEvent,
    BeforeAgentStartEvent,
    ContextEvent,
    ContextWithSystemEvent,
    InputEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ResourcesDiscoverEvent,
    SessionInfoChangedEvent,
    SessionShutdownEvent,
    SessionStartEvent,
    ToolCallEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
    UserBashEvent,
)
from .host import ExtensionHost, bash_result_from, make_agent_tool, tool_result_from
from .loader import (
    ExtensionAPI,
    ExtensionEvents,
    ExtensionLoadDiagnostic,
    LoadedExtensions,
    discover_extension_entries,
    extension_search_paths,
    load_extensions,
    load_extensions_async,
)
from .runner import ExtensionRunner, InputOutcome, UserBashOutcome
from .types import (
    Command,
    CommandContext,
    Extension,
    ExtensionContext,
    ExtensionError,
    ToolCallEventResult,
    ToolDefinition,
    ToolError,
    ToolResult,
    ToolResultDict,
    ToolResultEventResult,
    as_mapping,
    is_awaitable,
    maybe_await,
)

__all__ = [
    "EVENT_NAMES",
    "AgentEndEvent",
    "AgentErrorEvent",
    "AgentStartEvent",
    "BeforeAgentStartEvent",
    "Command",
    "CommandContext",
    "ContextEvent",
    "ContextWithSystemEvent",
    "Extension",
    "ExtensionAPI",
    "ExtensionContext",
    "ExtensionError",
    "ExtensionEvents",
    "ExtensionHost",
    "ExtensionLoadDiagnostic",
    "ExtensionRunner",
    "InputEvent",
    "InputOutcome",
    "LoadedExtensions",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "ResourcesDiscoverEvent",
    "SessionInfoChangedEvent",
    "SessionShutdownEvent",
    "SessionStartEvent",
    "ToolCallEvent",
    "ToolCallEventResult",
    "ToolDefinition",
    "ToolError",
    "ToolExecutionEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "ToolResult",
    "ToolResultDict",
    "ToolResultEvent",
    "ToolResultEventResult",
    "TurnEndEvent",
    "TurnStartEvent",
    "UserBashEvent",
    "UserBashOutcome",
    "as_mapping",
    "bash_result_from",
    "discover_extension_entries",
    "extension_search_paths",
    "is_awaitable",
    "load_extensions",
    "load_extensions_async",
    "make_agent_tool",
    "maybe_await",
    "tool_result_from",
]

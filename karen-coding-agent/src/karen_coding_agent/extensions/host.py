"""The session-bound side of the extension API (pi's ``ExtensionActions``).

The loader hands every extension an :class:`~karen_coding_agent.extensions.loader.ExtensionAPI`
that is safe to build before any session exists; the *actions* behind it — write
an entry, send a message, switch models — are supplied by the host, which the
CLI creates with the loaded settings and binds to the ``AgentSession`` once the
session opens. Everything in here resolves the session at call time, so a
``fork`` or ``switch_session`` that swaps it is followed without re-binding.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from .types import (
    ExtensionContext,
    ToolDefinition,
    ToolError,
    ToolResultDict,
)

__all__ = ["ExtensionHost", "bash_result_from", "make_agent_tool", "tool_result_from"]


def _now_ms() -> int:
    return int(time.time() * 1000)


def bash_result_from(value: Any) -> Any:
    """Coerce what a ``user_bash`` handler returned into a ``BashResult``.

    Accepts the executor's ``BashResult`` itself or a mapping with its fields
    (``output``, ``exit_code``, ``cancelled``, ``truncated``, ``full_output_path``);
    anything else is a contract violation and raises, which the caller surfaces
    as a failed command rather than a half-recorded one.
    """
    from ..bash_executor import BashResult

    if isinstance(value, BashResult):
        return value
    if isinstance(value, dict):
        return BashResult(
            str(value.get("output", "")),
            value.get("exit_code", value.get("exitCode")),
            bool(value.get("cancelled", False)),
            bool(value.get("truncated", False)),
            value.get("full_output_path", value.get("fullOutputPath")),
        )
    raise TypeError(
        "a user_bash handler returns a BashResult or a mapping with its fields, "
        f"not {type(value).__name__}"
    )


def tool_result_from(value: Any) -> Any:
    """Normalise what an extension tool returned into an ``AgentToolResult``.

    Accepted shapes: an ``AgentToolResult`` already, a
    :class:`~karen_coding_agent.extensions.types.ToolResultDict`, a mapping with
    its fields, a plain string (model-facing content, nothing else), or
    nothing at all (side-effect tool; the model still gets a stub back). Any
    other shape is wrapped as a text block of its ``str()`` — the loop's
    validator would otherwise surface a raw type error as a tool result, and
    an extension author who returned, say, a bare number would be left to
    guess why their tool is "failing".
    Raising — including a returned :class:`ToolError`, which is raisable — is
    how failure is reported, so it is also the one case *not* converted here.
    """
    from karen_ai import TextContent
    from karen_agent.types import AgentToolResult

    if isinstance(value, AgentToolResult):
        return value
    if isinstance(value, ToolResultDict):
        return AgentToolResult(
            content=value.content, details=value.details, usage=value.usage, terminate=value.terminate
        )
    if isinstance(value, dict):
        return AgentToolResult(
            content=value.get("content") or [],
            details=value.get("details"),
            usage=value.get("usage"),
            terminate=value.get("terminate"),
        )
    if value is None:
        # An extension tool that returns nothing made its point through
        # side effects; the model still needs *something* back.
        return AgentToolResult(content=[TextContent(text="(no output)")])
    if not isinstance(value, str):
        return AgentToolResult(content=[TextContent(text=str(value))])
    return AgentToolResult(content=[TextContent(text=value)])


def make_agent_tool(definition: ToolDefinition) -> Any:
    """Wrap a registered tool definition in karen-agent's ``AgentTool``.

    The wrapper is where the four karen-native arguments
    ``(tool_call_id, args, signal, on_update)`` meet the extension's
    ``execute()``, and where its return value is normalised. A failure becomes
    an error result through karen's normal path (``execute`` raising).
    """
    from karen_agent.types import AgentTool

    async def execute(tool_call_id: str, args: Dict[str, Any], signal: Any, on_update: Any) -> Any:
        result = definition.execute(tool_call_id, args, signal, on_update)
        if hasattr(result, "__await__"):
            result = await result
        if isinstance(result, ToolError):
            raise RuntimeError(result.message)
        return tool_result_from(result)

    return AgentTool(
        name=definition.name,
        description=definition.description,
        parameters=definition.parameters,
        label=definition.label or definition.name,
        execute=execute,
    )


class ExtensionHost:
    """Session-bound actions behind the API, and the context builders.

    *mode* is the CLI's face (``"tui"``, ``"repl"``, ``"print"`` or ``"rpc"``);
    *notify* is how an extension talks to the user without entering the
    transcript — the CLI decides where it goes for each face.
    """

    def __init__(
        self,
        *,
        settings: Any = None,
        mode: str = "repl",
        has_ui: bool = False,
        notify: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.session: Any = None
        self.settings = settings
        self.mode = mode
        self.has_ui = has_ui
        self._notify = notify or (lambda message, level="info": None)

    def bind(self, session: Any) -> None:
        self.session = session

    def set_notify(self, notify: Callable[[str, str], None]) -> None:
        """Rebind the notification sink (the CLI's face changed, e.g. REPL → TUI)."""
        self._notify = notify

    def notify(self, message: str, level: str = "info") -> None:
        self._notify(message, level)

    # -- context ----------------------------------------------------------------

    def make_context(self, signal: Optional[Any] = None) -> ExtensionContext:
        """The context a handler or command sees.

        The callbacks read the agent's live state rather than a snapshot, so
        they stay right across runs; there is no session yet only while
        factories run, and factories get no handlers.
        """
        session = self.session
        cwd = getattr(session, "cwd", ".")

        def is_idle() -> bool:
            if session is None:
                return True
            state = getattr(getattr(session, "agent", None), "state", None)
            return not getattr(state, "is_streaming", False)

        def has_pending_messages() -> bool:
            """Whether a steer/follow-up is queued — pi's `hasPendingMessages`."""
            if session is None:
                return False
            agent = getattr(session, "agent", None)
            if agent is None or not hasattr(agent, "peek_queued_messages"):
                return False
            return bool(agent.peek_queued_messages())

        return ExtensionContext(
            session=session,
            mode=self.mode,
            has_ui=self.has_ui,
            cwd=cwd,
            signal=signal,
            is_idle=is_idle,
            has_pending_messages=has_pending_messages,
        )

    # -- session actions (the ExtensionAPI delegates here) ----------------------

    def _require_session(self) -> Any:
        if self.session is None:
            raise RuntimeError("no session is bound to this extension runtime")
        return self.session

    async def append_entry(self, custom_type: str, data: Any = None) -> str:
        return await self._require_session()._append_custom_entry(custom_type, data)

    async def send_message(
        self, custom_type: str, content: Any, display: bool, details: Any
    ) -> None:
        """Store a custom message the model will see on its next request.

        While a run is streaming the message is persisted now but *queued* for
        the live transcript — appending it in place would splice it between an
        in-flight `tool_use` and its `tool_result`, which is exactly the hazard
        `record_bash_result` guards against. The settle flush re-appends it in
        the persisted order.
        """
        from karen_agent.messages import create_custom_message

        session = self._require_session()
        message = create_custom_message(
            custom_type=custom_type,
            content=content,
            display=display,
            details=details,
            timestamp=_now_ms(),
        )
        await session._persist_message(message)
        if getattr(session.agent.state, "is_streaming", False):
            session._pending_custom_messages.append(message)
        else:
            session.agent.state.messages.append(message)

    async def send_user_message(self, text: str, images: Any, deliver_as: str) -> None:
        # An extension-sent message is still user input: `AgentSession` routes it
        # through the `input` handlers (source "extension") and queues it.
        await self._require_session().send_user_message(text, images, deliver_as=deliver_as)

    async def exec_bash(self, command: str, **kwargs: Any) -> Any:
        return await self._require_session().execute_bash(command, **kwargs)

    async def set_session_name(self, name: str) -> None:
        await self._require_session().set_session_name(name)

    async def set_label(self, entry_id: str, label: Optional[str]) -> None:
        await self._require_session().set_label(entry_id, label)

    def get_active_tools(self) -> List[str]:
        session = self._require_session()
        return [getattr(tool, "name", "") for tool in session.agent.state.tools]

    def set_active_tools(self, names: Sequence[str]) -> None:
        """Replace the active tool set; unknown names are ignored, like pi.

        The pool is every tool the session knows — built-ins, MCP tools, and the
        extension tools — so re-activating a name after narrowing works. The
        declared set lives on the agent's *state* (`agent.state.tools`), which is
        what a run reads; assigning a set there narrows what the model may call.
        """
        session = self._require_session()
        pool = list(session.tools)
        wanted = {name for name in names if any(getattr(tool, "name", None) == name for tool in pool)}
        session.agent.state.tools = [
            tool for tool in pool if getattr(tool, "name", None) in wanted
        ]

    def get_thinking_level(self) -> Optional[str]:
        session = self._require_session()
        state = getattr(getattr(session, "agent", None), "state", None)
        return getattr(state, "thinking_level", None)

    async def set_thinking_level(self, level: str) -> None:
        await self._require_session().set_thinking_level(level)

    async def set_model(self, spec: str) -> bool:
        """Resolve ``"provider/id"`` (or a bare id under the current provider)."""
        session = self._require_session()
        provider, _, model_id = spec.partition("/")
        if not model_id:
            provider, model_id = getattr(session.model, "provider", ""), provider
        model = session.models.get_model(provider, model_id)
        if model is None:
            return False
        session.set_model(model)
        return True

"""Dispatch of events to extension handlers (pi's ``extensions/runner.ts``).

The runner owns three things pi's does: the handler list per event (with the
per-event merge semantics), the tools and commands extensions contributed, and
the error isolation that keeps a broken handler from taking the session down.

Two shape rules come straight from pi and are worth knowing before reading on:

- **Handlers run in load order**, and an extension's own handlers in the order
  it registered them. A handler may be sync or ``async``; both are awaited the
  same way.
- **A failing handler is reported, not propagated.** The exception goes to the
  error sink (the CLI prints it), that handler is skipped, and the next one
  still runs. The fail-safe exceptions are ``tool_call`` and ``user_bash``:
  there a failure blocks the operation rather than letting something unvetted
  through — again pi's behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .events import (
    BeforeAgentStartEvent,
    ContextEvent,
    ContextWithSystemEvent,
    InputEvent,
    MessageEndEvent,
    ResourcesDiscoverEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserBashEvent,
)
from .types import (
    Command,
    Extension,
    ExtensionContext,
    ExtensionError,
    ToolCallEventResult,
    ToolDefinition,
    ToolResultEventResult,
    as_mapping,
    maybe_await,
)

__all__ = ["ExtensionRunner", "InputOutcome", "UserBashOutcome"]

#: Receives every handler/factory failure. The runner never raises out of a
#: dispatch, so this is the one place errors surface.
ErrorSink = Callable[[ExtensionError], None]


def _role_of(message: Any) -> Optional[str]:
    """A message's role, for the karen-native and LLM-projected shapes alike."""
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


@dataclass(frozen=True)
class _HandlerOutcome:
    """One handler's turn: its result, or the exception it failed with.

    Failures travel alongside the results rather than through runner state so
    two dispatches running at once (parallel tool calls do) cannot see each
    other's errors.
    """

    extension: Extension
    result: Any = None
    error: Optional[BaseException] = None


@dataclass
class InputOutcome:
    """What the ``input`` handlers decided about a submitted prompt."""

    #: True when a handler consumed the prompt: the session runs nothing.
    handled: bool = False
    text: str = ""
    images: Any = None


@dataclass
class UserBashOutcome:
    """What the ``user_bash`` handlers decided about a ``!command``."""

    #: Set when a handler answered the command itself; the shell is not touched.
    result: Any = None
    #: True when a handler failed. pi blocks the command in that case rather
    #: than falling through to local execution, and so does karen.
    blocked: bool = False
    #: Why it was blocked, for the user-facing error.
    reason: str = ""


class ExtensionRunner:
    """Dispatches events and holds what extensions contributed.

    The runner is created before any session exists (extensions load at CLI
    startup), so everything session-bound is read through *context_factory* at
    dispatch time instead of being captured.
    """

    def __init__(
        self,
        extensions: Optional[Sequence[Extension]] = None,
        *,
        context_factory: Optional[Callable[[Optional[Any]], ExtensionContext]] = None,
        on_error: Optional[ErrorSink] = None,
    ) -> None:
        self.extensions: List[Extension] = list(extensions or [])
        self._context_factory = context_factory or (lambda signal=None: ExtensionContext())
        self._on_error = on_error
        #: The host behind the context factory, when one exists (the loader sets
        #: it). Kept here so whoever adopts the runner — a session opening, an
        #: RPC rebind — can re-bind the session-bound actions without knowing
        #: where the host came from.
        self.host: Any = None
        #: Extension-to-extension bus (`pi.events`).
        self._bus: Dict[str, List[Callable[..., Any]]] = {}
        #: Every reported failure, in order — the CLI prints them at startup and
        #: the tests read them.
        self.errors: List[ExtensionError] = []

    # -- registration ----------------------------------------------------------

    def set_context_factory(
        self, factory: Callable[[Optional[Any]], ExtensionContext]
    ) -> None:
        """Rebind how a handler's context is built (the session changed)."""
        self._context_factory = factory

    def context(self, signal: Optional[Any] = None) -> ExtensionContext:
        return self._context_factory(signal)

    def has_handlers(self, event: str) -> bool:
        """Whether anything at all listens for *event*."""
        return any(extension.handlers.get(event) for extension in self.extensions)

    def handlers_for(self, event: str) -> List[Any]:
        """Every handler for *event*, in load order."""
        return [
            handler
            for extension in self.extensions
            for handler in extension.handlers.get(event, ())
        ]

    def tools(self) -> List[ToolDefinition]:
        """Every registered tool, the first registration of a name winning."""
        seen: Dict[str, ToolDefinition] = {}
        for extension in self.extensions:
            for name, tool in extension.tools.items():
                seen.setdefault(name, tool)
        return list(seen.values())

    def commands(self) -> Dict[str, Command]:
        """Every registered extension command by name, the first one winning."""
        seen: Dict[str, Command] = {}
        for extension in self.extensions:
            for name, command in extension.commands.items():
                seen.setdefault(name, command)
        return seen

    # -- the extension-to-extension bus ---------------------------------------

    def on_bus(self, name: str, handler: Callable[..., Any]) -> Callable[[], None]:
        """Subscribe to a bus event; returns the unsubscribe function."""
        self._bus.setdefault(name, []).append(handler)

        def unsubscribe() -> None:
            listeners = self._bus.get(name)
            if listeners and handler in listeners:
                listeners.remove(handler)

        return unsubscribe

    async def emit_bus(self, name: str, payload: Any = None) -> None:
        """Fire a bus event. Handlers are isolated exactly like event handlers."""
        for handler in list(self._bus.get(name, ())):
            try:
                await maybe_await(handler(payload))
            except Exception as error:  # noqa: BLE001 - isolated on purpose
                self._report(f"<bus:{name}>", name, error)

    # -- error plumbing --------------------------------------------------------

    def _report(self, extension_path: str, event: str, error: BaseException) -> None:
        self.report_error(extension_path, event, f"{type(error).__name__}: {error}")

    def report_error(self, extension_path: str, event: str, message: str) -> ExtensionError:
        """Report a contract problem noticed outside a dispatch — the same sink.

        The session uses this when an extension tool cannot join the active set
        (its name is already taken, by a built-in or another extension): the
        registration is dropped either way, but silently is the one thing it
        must not be.
        """
        record = ExtensionError(extension_path=extension_path, event=event, message=message)
        self.errors.append(record)
        if self._on_error is not None:
            self._on_error(record)
        return record

    def extension_for_tool(self, name: str) -> Optional[Extension]:
        """The extension that registered tool *name*, first registration winning."""
        for extension in self.extensions:
            if name in extension.tools:
                return extension
        return None

    async def _dispatch(self, event: str, payload: Any, signal: Optional[Any] = None):
        """Run every handler for *event*, yielding a :class:`_HandlerOutcome` each.

        The context is built on first use: an event with no handlers must not
        reach into a session that may not exist yet.
        """
        context: Optional[ExtensionContext] = None
        for extension in self.extensions:
            for handler in extension.handlers.get(event, ()):
                if context is None:
                    context = self.context(signal)
                try:
                    result = await maybe_await(handler(payload, context))
                except Exception as error:  # noqa: BLE001 - isolated on purpose
                    self._report(extension.module_path, event, error)
                    yield _HandlerOutcome(extension, error=error)
                    continue
                yield _HandlerOutcome(extension, result=result)

    # -- notification events ---------------------------------------------------

    async def emit(self, event: Any, signal: Optional[Any] = None) -> List[Any]:
        """Fire a notification event; returns the handlers' non-``None`` results.

        Nothing is merged or applied here — events that *change* something have
        a dedicated ``emit_*`` below, where the merge rule lives with the event.
        """
        name = getattr(type(event), "type", "")
        results: List[Any] = []
        async for outcome in self._dispatch(name, event, signal):
            if outcome.result is not None:
                results.append(outcome.result)
        return results

    # -- transforming events ---------------------------------------------------

    async def emit_tool_call(
        self, event: ToolCallEvent, signal: Optional[Any] = None
    ) -> ToolCallEventResult:
        """Gate a tool call: the first ``block`` wins and stops the chain.

        A handler that raises blocks too — pi's fail-safe for the one event that
        guards execution.
        """
        failed = False
        async for outcome in self._dispatch("tool_call", event, signal):
            if outcome.error is not None:
                failed = True
                continue
            mapping = as_mapping(outcome.result)
            if mapping and mapping.get("block"):
                return ToolCallEventResult(
                    block=True,
                    reason=str(mapping.get("reason") or ""),
                    terminate=mapping.get("terminate"),
                )
        if failed:
            return ToolCallEventResult(
                block=True, reason="a tool_call extension failed, so the call was not run"
            )
        return ToolCallEventResult()

    async def emit_tool_result(
        self, event: ToolResultEvent, signal: Optional[Any] = None
    ) -> ToolResultEventResult:
        """Rewrite a tool result field-wise, each handler seeing the last edit.

        A handler returns ``{"content": …}`` (and the other fields) or mutates the
        event in place; either way only what actually differs from the executed
        result comes back, so the caller can hand it to karen's field-merging
        ``AfterToolCallResult`` untouched.
        """
        original = {
            "content": event.content,
            "details": event.details,
            "usage": event.usage,
            "is_error": event.is_error,
        }
        async for outcome in self._dispatch("tool_result", event, signal):
            mapping = as_mapping(outcome.result)
            if not mapping:
                continue
            # Apply as we go: the next handler must observe this one's edits.
            for key in ("content", "details", "usage", "is_error"):
                if key in mapping and mapping[key] is not None:
                    setattr(event, key, mapping[key])
        changes = {
            key: getattr(event, key)
            for key, value in original.items()
            if getattr(event, key) != value
        }
        if not changes:
            return ToolResultEventResult()
        return ToolResultEventResult(**changes)

    async def emit_user_bash(
        self, event: UserBashEvent, signal: Optional[Any] = None
    ) -> UserBashOutcome:
        """Let a handler answer a ``!command`` instead of the shell.

        A handler that answers returns ``{"result": …}``; one that declines
        returns nothing at all. Anything else — pi's ``{"operations": …}``
        shape, which karen does not implement, or a typo — is treated as a
        failure and blocks the command rather than falling through to local
        execution: a handler that thought it intercepted the command must never
        see it run anyway.
        """
        failed = False
        async for outcome in self._dispatch("user_bash", event, signal):
            if outcome.error is not None:
                failed = True
                continue
            mapping = as_mapping(outcome.result)
            if not mapping:
                continue
            if mapping.get("result") is not None:
                return UserBashOutcome(result=mapping["result"])
            if "operations" in mapping:
                self._report(
                    outcome.extension.module_path,
                    "user_bash",
                    ValueError("`operations` is not supported by karen's user_bash"),
                )
                failed = True
                continue
            self._report(
                outcome.extension.module_path,
                "user_bash",
                ValueError("a user_bash handler returns {'result': …} or nothing"),
            )
            failed = True
        if failed:
            # pi blocks the command when a handler fails rather than letting it
            # fall through to local execution unvetted.
            return UserBashOutcome(
                blocked=True,
                reason="a user_bash extension failed, so the command was not run",
            )
        return UserBashOutcome()

    async def emit_input(
        self, event: InputEvent, signal: Optional[Any] = None
    ) -> InputOutcome:
        """Chain ``input`` transforms; ``handled`` consumes the prompt outright."""
        outcome = InputOutcome(text=event.text, images=event.images)
        async for handler_outcome in self._dispatch("input", event, signal):
            mapping = as_mapping(handler_outcome.result)
            if not mapping:
                continue
            action = mapping.get("action")
            if action == "handled":
                outcome.handled = True
                return outcome
            if action == "transform":
                if mapping.get("text") is not None:
                    outcome.text = mapping["text"]
                if "images" in mapping:
                    outcome.images = mapping["images"]
                # Later handlers see the rewrite, exactly like pi.
                event.text = outcome.text
                event.images = outcome.images
                continue
            # "continue", or an action we do not know: accept what we have.
            return outcome
        return outcome

    async def emit_context(
        self, event: ContextEvent, signal: Optional[Any] = None
    ) -> List[Any]:
        """Chain ``context`` replacements; a failed handler leaves messages alone."""
        messages = event.messages
        async for outcome in self._dispatch("context", event, signal):
            mapping = as_mapping(outcome.result)
            if mapping and mapping.get("messages") is not None:
                messages = mapping["messages"]
                event.messages = messages
        return messages

    async def emit_context_with_system(
        self, event: ContextWithSystemEvent, signal: Optional[Any] = None
    ) -> List[Any]:
        """Chain ``context_with_system`` replacements — the result is sent as-is.

        The event's contract asks handlers to keep a system message at index
        zero; a replacement that drops it is honoured (the result is what the
        provider gets) but reported, because the request would otherwise go out
        with no prompt and no tool declarations, which reads as a provider bug
        rather than an extension bug.
        """
        messages = event.messages
        had_system = bool(messages) and _role_of(messages[0]) == "system"
        async for outcome in self._dispatch("context_with_system", event, signal):
            mapping = as_mapping(outcome.result)
            if not mapping or mapping.get("messages") is None:
                continue
            messages = mapping["messages"]
            event.messages = messages
            if had_system and (not messages or _role_of(messages[0]) != "system"):
                self._report(
                    outcome.extension.module_path,
                    "context_with_system",
                    ValueError("the replacement dropped the leading system message"),
                )
        return messages

    async def emit_message_end(
        self, event: MessageEndEvent, signal: Optional[Any] = None
    ) -> Any:
        """Let handlers replace a finalized message; the same role is required.

        A replacement whose role differs from the original's is refused and
        reported: the transcript would no longer line up with the tool
        call/result pairing the providers require.
        """
        message = event.message
        role = getattr(message, "role", None)
        async for outcome in self._dispatch("message_end", event, signal):
            mapping = as_mapping(outcome.result)
            if not mapping or mapping.get("message") is None:
                continue
            replacement = mapping["message"]
            if getattr(replacement, "role", None) != role:
                self._report(
                    outcome.extension.module_path,
                    "message_end",
                    ValueError("a message_end replacement keeps the original role"),
                )
                continue
            message = replacement
            event.message = replacement
        return message

    async def emit_before_agent_start(
        self, event: BeforeAgentStartEvent, signal: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Collect the run-level injections: every custom message, and the system prompt.

        Each handler's message is kept (pi pushes them all, in handler order);
        the system prompt is last-writer-wins, and a handler that returns one —
        even the empty string — sets it, so ``{"system_prompt": ""}`` clears the
        prompt for the run rather than being ignored.
        """
        collected: Dict[str, Any] = {"messages": []}
        async for outcome in self._dispatch("before_agent_start", event, signal):
            mapping = as_mapping(outcome.result)
            if not mapping:
                continue
            if mapping.get("message"):
                collected["messages"].append(mapping["message"])
            if "system_prompt" in mapping and mapping["system_prompt"] is not None:
                collected["system_prompt"] = mapping["system_prompt"]
                # Later handlers observe an earlier replacement, like pi.
                event.system_prompt = mapping["system_prompt"]
        return collected

    async def emit_resources_discover(
        self, event: ResourcesDiscoverEvent, signal: Optional[Any] = None
    ) -> Dict[str, List[str]]:
        """Merge resource paths contributed by handlers, keeping first-seen order."""
        collected: Dict[str, List[str]] = {
            "skill_paths": [],
            "prompt_paths": [],
            "theme_paths": [],
        }
        async for outcome in self._dispatch("resources_discover", event, signal):
            mapping = as_mapping(outcome.result)
            if not mapping:
                continue
            for key in collected:
                for value in mapping.get(key) or ():
                    if value not in collected[key]:
                        collected[key].append(value)
        return collected

    # -- commands --------------------------------------------------------------

    async def run_command(self, name: str, args: Optional[str], context: Any) -> bool:
        """Run the extension command *name*; False when nothing registered it.

        A failing command is reported and still counts as handled — the user
        asked for it, so quietly turning it into a prompt would be worse.
        """
        command = self.commands().get(name)
        if command is None:
            return False
        try:
            await maybe_await(command.handler(args, context))
        except Exception as error:  # noqa: BLE001 - isolated on purpose
            self._report(f"<command:{name}>", "command", error)
        return True

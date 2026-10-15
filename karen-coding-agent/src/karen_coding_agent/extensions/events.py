"""Event payloads dispatched to extension handlers (pi's ``extensions/types.ts``).

Every event carries pi's name in ``type`` as a class attribute, so handlers can
branch on it and the runner can dispatch generically. Payload field names are
pi's, snake_cased where the wire name was camelCase (``tool_call_id``,
``exclude_from_context``); karen's own ``AgentMessage`` objects appear as-is, so a
handler that returns messages works with anything the session already stores.

Which events can *change* something is the runner's business (see
``runner.py``): ``tool_call`` (block), ``tool_result`` (field overrides),
``user_bash`` (replace the run), ``input`` (transform or consume the prompt),
``context`` / ``context_with_system`` (replace the messages),
``resources_discover`` (add resource paths), ``before_agent_start`` (inject a
message and/or replace the system prompt) and ``message_end`` (replace the
finalized message). The rest are notifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional

__all__ = [
    "EVENT_NAMES",
    "AgentEndEvent",
    "AgentErrorEvent",
    "AgentStartEvent",
    "BeforeAgentStartEvent",
    "ContextEvent",
    "ContextWithSystemEvent",
    "InputEvent",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "ResourcesDiscoverEvent",
    "SessionInfoChangedEvent",
    "SessionShutdownEvent",
    "SessionStartEvent",
    "ToolCallEvent",
    "ToolExecutionEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "ToolResultEvent",
    "TurnEndEvent",
    "TurnStartEvent",
    "UserBashEvent",
]


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


@dataclass
class SessionStartEvent:
    """The session is open and its context is loaded.

    *reason* is ``"startup"`` (a fresh runtime), ``"new"`` (the caller replaced
    a live session), ``"resume"`` (an existing on-disk session was reopened, a
    switch included) or ``"fork"`` (a fork or clone). pi also has ``"reload"``;
    karen has no reload path, so it is never emitted.
    *previous_session_file* names the session left behind when the runtime moved
    to another one.
    """

    type: ClassVar[str] = "session_start"
    reason: str = "startup"
    previous_session_file: Optional[str] = None


@dataclass
class SessionInfoChangedEvent:
    """The session's display name changed (``/name``, ``pi.set_session_name``)."""

    type: ClassVar[str] = "session_info_changed"
    name: Optional[str] = None


@dataclass
class SessionShutdownEvent:
    """The runtime is going away — quit, or session replacement.

    *reason* uses pi's vocabulary: ``"quit"`` when the runtime stops, ``"new"``
    when a live session is replaced (``/new``, RPC ``new_session``), ``"fork"``
    for a fork or clone, ``"resume"`` for a switch to an existing session. A
    replacement's shutdown and the `session_start` that follows carry the same
    reason, so a handler can pair them. (pi's ``"reload"`` is never emitted —
    karen has no reload path.)

    Extension authors release session-scoped resources here, and do it
    idempotently: cancellation, reload and process exit can all land on this
    path.
    """

    type: ClassVar[str] = "session_shutdown"
    reason: str = "quit"
    target_session_file: Optional[str] = None


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


@dataclass
class AgentStartEvent:
    """A run began."""

    type: ClassVar[str] = "agent_start"


@dataclass
class AgentEndEvent:
    """The run's message sequence settled (retries and compaction aside)."""

    type: ClassVar[str] = "agent_end"
    messages: List[Any] = field(default_factory=list)


@dataclass
class AgentErrorEvent:
    """The run ended in a failure the session could not retry away."""

    type: ClassVar[str] = "agent_error"
    error: str = ""


@dataclass
class TurnStartEvent:
    """One model request + tool batch began."""

    type: ClassVar[str] = "turn_start"
    turn_index: int = 0
    timestamp: int = 0


@dataclass
class TurnEndEvent:
    """One model request + tool batch finished."""

    type: ClassVar[str] = "turn_end"
    turn_index: int = 0
    timestamp: int = 0
    message: Any = None
    tool_results: List[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass
class MessageStartEvent:
    """A message began streaming."""

    type: ClassVar[str] = "message_start"
    message: Any = None


@dataclass
class MessageUpdateEvent:
    """A streamed delta arrived for the message in flight."""

    type: ClassVar[str] = "message_update"
    message: Any = None
    assistant_message_event: Any = None


@dataclass
class MessageEndEvent:
    """A message finalized; returning ``{"message": m}`` replaces it (same role only)."""

    type: ClassVar[str] = "message_end"
    message: Any = None


# ---------------------------------------------------------------------------
# Tool execution (notification)
# ---------------------------------------------------------------------------


@dataclass
class ToolExecutionStartEvent:
    """A tool call started executing."""

    type: ClassVar[str] = "tool_execution_start"
    tool_call_id: str = ""
    tool_name: str = ""
    args: Any = None


@dataclass
class ToolExecutionUpdateEvent:
    """A tool reported partial progress."""

    type: ClassVar[str] = "tool_execution_update"
    tool_call_id: str = ""
    tool_name: str = ""
    args: Any = None
    partial_result: Any = None


@dataclass
class ToolExecutionEndEvent:
    """A tool call finished, successfully or not."""

    type: ClassVar[str] = "tool_execution_end"
    tool_call_id: str = ""
    tool_name: str = ""
    args: Any = None
    result: Any = None
    is_error: bool = False


# ---------------------------------------------------------------------------
# Tool calls and results (transforming)
# ---------------------------------------------------------------------------


@dataclass
class ToolCallEvent:
    """A tool call is about to execute.

    Return ``{"block": True, "reason": …}`` to stop it (the model sees the reason
    as an error result); the first block wins and later handlers are skipped.
    Mutating *input* in place patches the arguments the tool runs with, and
    later handlers observe the mutation — no re-validation happens after it, so
    patch deliberately.
    """

    type: ClassVar[str] = "tool_call"
    tool_call_id: str = ""
    tool_name: str = ""
    input: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultEvent:
    """A tool finished; handlers may rewrite what the model sees.

    Return ``{"content": […]}/{"details": …}/{"is_error": …}`` — only the fields
    you name change, and each later handler sees the earlier handlers' edits.
    This fires before karen normalizes tool-result images, so injected images are
    normalized too.
    """

    type: ClassVar[str] = "tool_result"
    tool_call_id: str = ""
    tool_name: str = ""
    input: Dict[str, Any] = field(default_factory=dict)
    content: List[Any] = field(default_factory=list)
    details: Any = None
    is_error: bool = False
    usage: Any = None


# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------


@dataclass
class UserBashEvent:
    """A ``!command`` line is about to run in the shell.

    Return ``{"result": BashResult}`` to answer it yourself instead — the first
    handler that does wins and the shell is never touched. Returning ``None``
    passes the command along; a handler that raises blocks the command rather
    than letting it fall through, exactly like pi. So does a return value that
    is neither of the two shapes — pi's ``{"operations": …}`` alternative is
    not ported, and a handler that returns it blocks the command rather than
    seeing it run locally.
    """

    type: ClassVar[str] = "user_bash"
    command: str = ""
    exclude_from_context: bool = False
    cwd: str = ""


@dataclass
class InputEvent:
    """A user prompt arrived, before it reaches the agent.

    Return ``{"action": "transform", "text": …, "images": …}`` to rewrite it
    (later handlers see the rewrite), ``{"action": "handled"}`` to consume it —
    no run happens — or ``{"action": "continue"}`` to stop the chain and accept
    the text as it stands.
    """

    type: ClassVar[str] = "input"
    text: str = ""
    images: Optional[List[Any]] = None
    #: "interactive" | "rpc" | "extension" — where the prompt came from.
    source: str = "interactive"
    #: "steer" | "followUp" | None — set when the prompt is queued mid-run.
    streaming_behavior: Optional[str] = None


# ---------------------------------------------------------------------------
# Model context
# ---------------------------------------------------------------------------


@dataclass
class ContextEvent:
    """The conversation is about to be converted for the model.

    Return ``{"messages": [...]}`` to replace them; handlers chain, so each sees
    the previous replacement. A handler that fails is reported and skipped, and
    the messages it would have replaced stay in place.

    pi fires this with the conversation *without* the prompt and tool system
    messages, and restores that state afterwards. karen has no separate prompt
    state to restore, so handlers see the whole transcript here — including the
    leading system message — and the returned list is converted as usual.
    """

    type: ClassVar[str] = "context"
    messages: List[Any] = field(default_factory=list)


@dataclass
class ContextWithSystemEvent:
    """The converted messages are about to be sent; the result is sent as returned.

    Use it only when a request-local transformation must own the complete
    transcript, and keep a system message at index zero.
    """

    type: ClassVar[str] = "context_with_system"
    messages: List[Any] = field(default_factory=list)


@dataclass
class BeforeAgentStartEvent:
    """A run is about to start, with the prompt already resolved.

    Return ``{"message": {"content": …, "custom_type": …, "display": …}}`` to
    append a custom message the model will see on this run, ``{"system_prompt":
    "…"}`` to replace the system prompt for the duration of the run, or both.
    Later handlers observe an earlier handler's replacement.
    """

    type: ClassVar[str] = "before_agent_start"
    prompt: str = ""
    images: Optional[List[Any]] = None
    system_prompt: str = ""


@dataclass
class ResourcesDiscoverEvent:
    """Skills and prompt templates are being collected.

    Return ``{"skill_paths": [...], "prompt_paths": [...], "theme_paths": [...]}``
    — the lists are merged across handlers, and each path is a directory or a
    ``SKILL.md``-style file the loader already understands. Themes are accepted
    for pi parity; karen has no theme loader yet, so those paths are ignored.
    """

    type: ClassVar[str] = "resources_discover"
    cwd: str = ""
    #: "startup" | "reload"
    reason: str = "startup"


#: Every event name a handler can subscribe to — the ``type`` of each payload
#: class above. ``ExtensionAPI.on`` validates subscriptions against it.
EVENT_NAMES = frozenset(
    cls.type
    for cls in (
        SessionStartEvent,
        SessionInfoChangedEvent,
        SessionShutdownEvent,
        AgentStartEvent,
        AgentEndEvent,
        AgentErrorEvent,
        TurnStartEvent,
        TurnEndEvent,
        MessageStartEvent,
        MessageUpdateEvent,
        MessageEndEvent,
        ToolExecutionStartEvent,
        ToolExecutionUpdateEvent,
        ToolExecutionEndEvent,
        ToolCallEvent,
        ToolResultEvent,
        UserBashEvent,
        InputEvent,
        ContextEvent,
        ContextWithSystemEvent,
        BeforeAgentStartEvent,
        ResourcesDiscoverEvent,
    )
)

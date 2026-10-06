"""Types for the agent loop, mirroring pi's `packages/agent/src/types.ts`.

The loop works with `AgentMessage` throughout and transforms to `Message[]` only
at the LLM call boundary. `AgentMessage` is `Any` on purpose: apps may add custom
message shapes (anything with a `role` attribute) and decide in `convert_to_llm`
how each one reaches the model — or that it never does.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union

from karen_ai import (
    AbortSignal,
    AssistantMessage,
    AssistantMessageEvent,
    AssistantMessageEventStream,
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    TextContent,
    Tool,
    ToolResultMessage,
    TranscriptContext,
    Usage,
)
from karen_ai.types import KarenBase
from pydantic import ConfigDict, Field
from typing_extensions import Annotated

# A loop message: one of karen_ai's LLM messages or an app-defined custom message.
AgentMessage = Any

#: Stream function used by the agent loop. `Models.stream_simple` satisfies this
#: shape via `models_stream_fn()`. Contract (mirrors pi): never raise — encode
#: failures in the returned stream as an error event and a final message whose
#: stop_reason is "error"/"aborted".
StreamFn = Callable[
    [Model, TranscriptContext, Optional[SimpleStreamOptions]],
    Union[AssistantMessageEventStream, Awaitable[AssistantMessageEventStream]],
]

ToolExecutionMode = Literal["sequential", "parallel"]
QueueMode = Literal["all", "one-at-a-time"]

# pi's AgentToolCall: the toolCall content block of an assistant message.
AgentToolCall = Any  # karen_ai.ToolCall in practice


# ---------------------------------------------------------------------------
# Tool definitions and results
# ---------------------------------------------------------------------------


class AgentToolResult(KarenBase):
    """Final or partial result produced by a tool."""

    content: List[Union[TextContent, ImageContent]] = Field(default_factory=list)
    details: Any = None
    usage: Optional[Usage] = None
    #: Hint that the agent should stop after the current tool batch. Early
    #: termination only happens when every finalized result in the batch sets it.
    terminate: Optional[bool] = None


#: Callback used by tools to stream partial execution updates. Calls made after
#: the tool's execute() settles are ignored.
AgentToolUpdateCallback = Callable[[AgentToolResult], None]


class AgentTool(Tool):
    """Tool definition used by the agent runtime (pi's AgentTool)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    #: Optional shim applied to raw tool-call arguments before schema validation.
    prepare_arguments: Optional[Callable[[Any], Dict[str, Any]]] = None
    #: Execute the tool call. Raise on failure instead of encoding errors in content.
    execute: Optional[
        Callable[[str, Any, Optional[AbortSignal], Optional[AgentToolUpdateCallback]], Awaitable[AgentToolResult]]
    ] = None
    replay: Optional[Literal["never", "safe"]] = None
    #: Per-tool execution mode override; None follows the loop default.
    execution_mode: Optional[ToolExecutionMode] = None


# ---------------------------------------------------------------------------
# Loop configuration and hook payloads
# ---------------------------------------------------------------------------


class AgentContext(KarenBase):
    """Context snapshot passed into the low-level agent loop."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: List[AgentMessage] = Field(default_factory=list)
    tools: Optional[List[AgentTool]] = None


class BeforeToolCallResult(KarenBase):
    """Returning block=True prevents the tool from executing (error result instead)."""

    block: Optional[bool] = None
    reason: Optional[str] = None
    terminate: Optional[bool] = None


class AfterToolCallResult(KarenBase):
    """Field-by-field overrides; omitted fields keep the executed result's values."""

    content: Optional[List[Union[TextContent, ImageContent]]] = None
    details: Any = None
    is_error: Optional[bool] = None
    usage: Optional[Usage] = None
    terminate: Optional[bool] = None


class BeforeToolCallContext(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    context: AgentContext


class AfterToolCallContext(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    result: AgentToolResult
    is_error: bool
    context: AgentContext


class AgentTurnContext(KarenBase):
    """Context passed to completed-turn callbacks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: AssistantMessage
    tool_results: List[ToolResultMessage]
    context: AgentContext
    new_messages: List[AgentMessage]


class AgentTurnDecision(KarenBase):
    action: Literal["continue", "end"]


class AgentLoopTurnUpdate(KarenBase):
    """Replacement runtime state before another provider request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    context: Optional[AgentContext] = None
    messages: Optional[List[AgentMessage]] = None
    model: Optional[Model] = None
    thinking_level: Optional[str] = None


class PrepareRequestContext(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    context: AgentContext
    model: Model
    thinking_level: str


class AgentRequestUpdate(KarenBase):
    """Replacement runtime state for the request being prepared (no messages)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    context: Optional[AgentContext] = None
    model: Optional[Model] = None
    thinking_level: Optional[str] = None


class PrepareNextTurnContext(AgentTurnContext):
    pass


class AgentLoopConfig(SimpleStreamOptions):
    """Configuration for one agent loop run (pi's AgentLoopConfig extends SimpleStreamOptions)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    #: AgentMessage[] -> Message[] at the LLM call boundary. Must not raise.
    convert_to_llm: Callable[[List[AgentMessage]], Union[List[Message], Awaitable[List[Message]]]]
    #: Optional AgentMessage-level transform applied before convert_to_llm (e.g. pruning).
    transform_context: Optional[
        Callable[[List[AgentMessage], Optional[AbortSignal]], Awaitable[List[AgentMessage]]]
    ] = None
    #: Resolve an API key dynamically per LLM call (expiring OAuth tokens). Must not raise.
    get_api_key: Optional[Callable[[str], Union[Optional[str], Awaitable[Optional[str]]]]] = None
    #: Called after the turn's messages, immediately before turn_end. None preserves scheduling.
    finish_turn: Optional[Callable[[AgentTurnContext, Optional[AbortSignal]], Any]] = None
    #: Called immediately before every provider request, including the first.
    prepare_request: Optional[Callable[[PrepareRequestContext, Optional[AbortSignal]], Any]] = None
    #: Called after turn_end when the loop will continue, before the next turn starts.
    prepare_next_turn: Optional[Callable[[PrepareNextTurnContext], Any]] = None
    #: Steering messages to inject mid-run (polled after each turn). Must not raise; return [].
    get_steering_messages: Optional[Callable[[], Awaitable[List[AgentMessage]]]] = None
    #: Follow-up messages to process when the agent would otherwise stop.
    get_follow_up_messages: Optional[Callable[[], Awaitable[List[AgentMessage]]]] = None
    #: Tool execution mode; default "parallel".
    tool_execution: Optional[ToolExecutionMode] = None
    #: Called after argument validation, before execution. Return block=True to prevent.
    before_tool_call: Optional[Callable[[BeforeToolCallContext, Optional[AbortSignal]], Any]] = None
    #: Called after execution, before tool_execution_end / result message events.
    after_tool_call: Optional[Callable[[AfterToolCallContext, Optional[AbortSignal]], Any]] = None


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class AgentStartEvent(KarenBase):
    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["agent_end"] = "agent_end"
    messages: List[AgentMessage]
    #: Set by the application layer when the failed run it is ending will be
    #: retried (pi's `AgentSession` adds `willRetry` to the forwarded event).
    will_retry: Optional[bool] = None


class TurnStartEvent(KarenBase):
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage
    tool_results: List[ToolResultMessage]


class MessageStartEvent(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["message_start"] = "message_start"
    message: AgentMessage


class MessageUpdateEvent(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent


class MessageEndEvent(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class ToolExecutionStartEvent(KarenBase):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: Any = None


class ToolExecutionUpdateEvent(KarenBase):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: Any = None
    partial_result: Any = None


class ToolExecutionEndEvent(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: Any = None
    is_error: bool


AgentEvent = Annotated[
    Union[
        AgentStartEvent,
        AgentEndEvent,
        TurnStartEvent,
        TurnEndEvent,
        MessageStartEvent,
        MessageUpdateEvent,
        MessageEndEvent,
        ToolExecutionStartEvent,
        ToolExecutionUpdateEvent,
        ToolExecutionEndEvent,
    ],
    Field(discriminator="type"),
]

#: Sink for agent events; may be sync or async.
AgentEventSink = Callable[[Any], Union[None, Awaitable[None]]]

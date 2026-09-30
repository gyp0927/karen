"""Stateful wrapper around the low-level agent loop (pi's `agent.ts`).

`Agent` owns the current transcript, emits lifecycle events, executes tools,
and exposes queueing APIs for steering and follow-up messages.

The app-facing surface maps to pi as: ``steering_mode``/``follow_up_mode``,
``steer()``/``follow_up()``, ``clear_steering_queue()``/``clear_follow_up_queue()``/
``clear_all_queues()``, ``has_queued_messages()``, ``peek_queued_messages()``,
``wait_for_idle()`` and ``continue_()`` (renamed — `continue` is a Python
keyword). Options are constructor keyword arguments instead of pi's
`AgentOptions` object.

Deviations: `thinking_level` is karen-ai's `ModelThinkingLevel` (it includes
``"off"``; the streamed `reasoning` field drops it). Failed runs follow pi
exactly — the error is converted into an assistant message with stop reason
``"error"``/``"aborted"`` and emitted as message/turn/agent events instead of
propagating out of `prompt()`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional, Set, Union

from karen_ai import (
    AbortController,
    AbortSignal,
    AssistantMessage,
    ImageContent,
    Model,
    ModelThinkingLevel,
    TextContent,
    ThinkingBudgets,
    Transport,
    Usage,
    UserMessage,
    create_initial_system_message,
    get_current_system_message,
    get_current_system_prompt,
    to_tool_declaration,
)
from karen_ai.types import ModelCost

from .agent_loop import run_agent_loop, run_agent_loop_continue
from .messages import message_field
from .stream_fn import get_default_stream_fn
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    PrepareNextTurnContext,
    QueueMode,
    StreamFn,
    ToolExecutionMode,
    TurnEndEvent,
)

__all__ = ["Agent", "AgentState", "AgentInitialState", "default_convert_to_llm", "DEFAULT_MODEL"]


def default_convert_to_llm(messages: List[AgentMessage]) -> List[Any]:
    """Keep the roles every provider understands; drop harness-only messages."""
    return [message for message in messages if message_field(message, "role") in ("system", "user", "assistant", "toolResult")]


DEFAULT_MODEL = Model(
    id="unknown",
    name="unknown",
    api="unknown",
    provider="unknown",
    base_url="",
    reasoning=False,
    input=[],
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=0,
    max_tokens=0,
)


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


class AgentState:
    """Current agent state (pi's `AgentState`).

    ``system_prompt`` is read-only: it is replayed from the transcript's system
    messages — to change the prompt, append a system message. Assigning
    ``tools`` or ``messages`` copies the provided top-level list.
    """

    def __init__(
        self,
        model: Model,
        thinking_level: ModelThinkingLevel,
        tools: List[AgentTool],
        messages: List[AgentMessage],
    ) -> None:
        self.model = model
        self.thinking_level = thinking_level
        self._tools = list(tools)
        self._messages = list(messages)
        #: True while the agent is processing a prompt or continuation. Stays
        #: true until awaited `agent_end` listeners settle.
        self.is_streaming = False
        #: Partial assistant message for the current streamed response, if any.
        self.streaming_message: Optional[AgentMessage] = None
        #: Tool call ids currently executing.
        self.pending_tool_calls: Set[str] = set()
        #: Error message from the most recent failed or aborted assistant turn.
        self.error_message: Optional[str] = None

    @property
    def system_prompt(self) -> str:
        """Current system prompt, replayed from the transcript's system messages."""
        return get_current_system_prompt(self._messages)

    @property
    def tools(self) -> List[AgentTool]:
        return self._tools

    @tools.setter
    def tools(self, next_tools: List[AgentTool]) -> None:
        self._tools = list(next_tools)

    @property
    def messages(self) -> List[AgentMessage]:
        return self._messages

    @messages.setter
    def messages(self, next_messages: List[AgentMessage]) -> None:
        self._messages = list(next_messages)


@dataclass
class AgentInitialState:
    """Initial state for :class:`Agent`.

    ``system_prompt`` and ``tools`` become the leading system message unless
    ``messages`` already starts with one.
    """

    system_prompt: Optional[str] = None
    model: Optional[Model] = None
    thinking_level: Optional[ModelThinkingLevel] = None
    tools: Optional[List[AgentTool]] = None
    messages: Optional[List[AgentMessage]] = None


def _create_agent_state(initial_state: Optional[AgentInitialState]) -> AgentState:
    tools = list(initial_state.tools) if initial_state and initial_state.tools else []
    messages = list(initial_state.messages) if initial_state and initial_state.messages else []
    initial_message = create_initial_system_message(
        initial_state.system_prompt if initial_state else None,
        [to_tool_declaration(tool) for tool in tools],
    )
    first_role = message_field(messages[0], "role") if messages else None
    if first_role != "system" and initial_message is not None:
        messages.insert(0, initial_message)

    return AgentState(
        model=(initial_state.model if initial_state else None) or DEFAULT_MODEL,
        thinking_level=(initial_state.thinking_level if initial_state else None) or "off",
        tools=tools,
        messages=messages,
    )


class _PendingMessageQueue:
    def __init__(self, mode: QueueMode) -> None:
        self._messages: List[AgentMessage] = []
        self.mode = mode

    def enqueue(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return len(self._messages) > 0

    def peek(self) -> List[AgentMessage]:
        if self.mode == "all":
            return list(self._messages)
        return self._messages[:1]

    def drain(self) -> List[AgentMessage]:
        drained = self.peek()
        del self._messages[: len(drained)]
        return drained

    def clear(self) -> None:
        self._messages.clear()


@dataclass
class _ActiveRun:
    future: "asyncio.Future[None]"
    abort_controller: AbortController = field(default_factory=AbortController)


class Agent:
    """Stateful wrapper around the low-level agent loop.

    `Agent` owns the current transcript, emits lifecycle events, executes tools,
    and exposes queueing APIs for steering and follow-up messages.
    """

    def __init__(
        self,
        *,
        initial_state: Optional[AgentInitialState] = None,
        convert_to_llm: Optional[Callable[[List[AgentMessage]], Any]] = None,
        transform_context: Optional[Callable[..., Awaitable[List[AgentMessage]]]] = None,
        stream_fn: Optional[StreamFn] = None,
        get_api_key: Optional[Callable[[str], Any]] = None,
        on_payload: Optional[Callable[..., Any]] = None,
        on_response: Optional[Callable[..., Any]] = None,
        on_provider_stream_event: Optional[Callable[..., Any]] = None,
        before_tool_call: Optional[
            Callable[[BeforeToolCallContext, Optional[AbortSignal]], Any]
        ] = None,
        after_tool_call: Optional[Callable[[AfterToolCallContext, Optional[AbortSignal]], Any]] = None,
        finish_turn: Optional[Callable[..., Any]] = None,
        prepare_request: Optional[Callable[..., Any]] = None,
        prepare_next_turn: Optional[Callable[[Optional[AbortSignal]], Any]] = None,
        prepare_next_turn_with_context: Optional[Callable[[PrepareNextTurnContext, Optional[AbortSignal]], Any]] = None,
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
        session_id: Optional[str] = None,
        thinking_budgets: Optional[ThinkingBudgets] = None,
        transport: Transport = "auto",
        max_retry_delay_ms: Optional[int] = None,
        tool_execution: ToolExecutionMode = "parallel",
    ) -> None:
        self._state = _create_agent_state(initial_state)
        self._listeners: List[Callable[[AgentEvent, AbortSignal], Any]] = []
        self._steering_queue = _PendingMessageQueue(steering_mode)
        self._follow_up_queue = _PendingMessageQueue(follow_up_mode)

        self.convert_to_llm = convert_to_llm or default_convert_to_llm
        self.transform_context = transform_context
        self.stream_function: StreamFn = stream_fn or get_default_stream_fn()
        self.get_api_key = get_api_key
        self.on_payload = on_payload
        self.on_response = on_response
        self.on_provider_stream_event = on_provider_stream_event
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.finish_turn = finish_turn
        self.prepare_request = prepare_request
        self.prepare_next_turn = prepare_next_turn
        self.prepare_next_turn_with_context = prepare_next_turn_with_context
        self._active_run: Optional[_ActiveRun] = None
        #: Session identifier forwarded to providers for cache-aware backends.
        self.session_id = session_id
        #: Optional per-level thinking token budgets forwarded to the stream function.
        self.thinking_budgets = thinking_budgets
        #: Preferred transport forwarded to the stream function.
        self.transport = transport
        #: Optional cap for provider-requested retry delays.
        self.max_retry_delay_ms = max_retry_delay_ms
        #: Tool execution strategy for assistant messages that contain multiple tool calls.
        self.tool_execution = tool_execution

    # -- state ----------------------------------------------------------------

    @property
    def state(self) -> AgentState:
        """Current agent state.

        Assigning ``state.tools`` or ``state.messages`` copies the provided
        top-level list.
        """
        return self._state

    # -- event subscription ----------------------------------------------------

    def subscribe(self, listener: Callable[[AgentEvent, AbortSignal], Any]) -> Callable[[], None]:
        """Subscribe to agent lifecycle events; returns an unsubscribe function.

        Listener results are awaited in subscription order and are included in
        the current run's settlement. Listeners also receive the active abort
        signal for the current run.

        ``agent_end`` is the final emitted event for a run, but the agent does
        not become idle until all awaited listeners for that event have settled.
        """
        if listener not in self._listeners:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    # -- queues ----------------------------------------------------------------

    @property
    def steering_mode(self) -> QueueMode:
        """Controls how queued steering messages are drained."""
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        """Controls how queued follow-up messages are drained."""
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_queue.mode = mode

    def steer(self, message: AgentMessage) -> None:
        """Queue a message to be injected after the current assistant turn finishes."""
        self._steering_queue.enqueue(message)

    def follow_up(self, message: AgentMessage) -> None:
        """Queue a message to run only after the agent would otherwise stop."""
        self._follow_up_queue.enqueue(message)

    def clear_steering_queue(self) -> None:
        """Remove all queued steering messages."""
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        """Remove all queued follow-up messages."""
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        """Remove all queued steering and follow-up messages."""
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        """True when either queue still contains pending messages."""
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    def peek_queued_messages(self) -> List[AgentMessage]:
        """Preview the messages selected for the next turn without consuming them."""
        steering = self._steering_queue.peek()
        return steering if len(steering) > 0 else self._follow_up_queue.peek()

    # -- run control ------------------------------------------------------------

    @property
    def signal(self) -> Optional[AbortSignal]:
        """Active abort signal for the current run, if any."""
        if self._active_run is None:
            return None
        return self._active_run.abort_controller.signal

    def abort(self) -> None:
        """Abort the current run, if one is active."""
        if self._active_run is not None:
            self._active_run.abort_controller.abort()

    async def wait_for_idle(self) -> None:
        """Resolve when the current run and all awaited event listeners have finished.

        Resolves after ``agent_end`` listeners settle.
        """
        run = self._active_run
        if run is None:
            return
        await run.future

    def reset(self) -> None:
        """Clear conversation state and queues while retaining the replayed prompt/tool baseline."""
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before resetting.")

        baseline = get_current_system_message(self._state.messages)
        self._state.messages = [baseline] if baseline is not None else []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._state.error_message = None
        self.clear_follow_up_queue()
        self.clear_steering_queue()

    async def prompt(
        self,
        input: Union[str, AgentMessage, List[AgentMessage]],
        images: Optional[List[ImageContent]] = None,
    ) -> None:
        """Start a new prompt from text, a single message, or a batch of messages."""
        if self._active_run is not None:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or follow_up() to queue messages, "
                "or wait for completion."
            )
        messages = self._normalize_prompt_input(input, images)
        await self._run_prompt_messages(messages)

    async def continue_(self) -> None:
        """Continue from the current transcript. The last message must be a user or tool-result message."""
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")

        messages = self._state.messages
        last_message = messages[-1] if messages else None
        if last_message is None or all(message_field(message, "role") == "system" for message in messages):
            raise RuntimeError("No messages to continue from")

        if message_field(last_message, "role") == "assistant":
            queued_steering = self._steering_queue.drain()
            if queued_steering:
                await self._run_prompt_messages(queued_steering, skip_initial_steering_poll=True)
                return

            queued_follow_ups = self._follow_up_queue.drain()
            if queued_follow_ups:
                await self._run_prompt_messages(queued_follow_ups)
                return

            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_continuation()

    # -- internals ---------------------------------------------------------------

    def _normalize_prompt_input(
        self,
        input: Union[str, AgentMessage, List[AgentMessage]],
        images: Optional[List[ImageContent]],
    ) -> List[AgentMessage]:
        if isinstance(input, list):
            return input
        if not isinstance(input, str):
            return [input]
        content: List[Any] = [TextContent(text=input)]
        if images:
            content.extend(images)
        return [UserMessage(content=content, timestamp=_now_ms())]

    async def _run_prompt_messages(
        self,
        messages: List[AgentMessage],
        skip_initial_steering_poll: bool = False,
    ) -> None:
        async def executor(signal: AbortSignal) -> None:
            await run_agent_loop(
                messages,
                self._create_context_snapshot(),
                self._create_loop_config(skip_initial_steering_poll=skip_initial_steering_poll),
                self._process_events,
                signal,
                self.stream_function,
            )

        await self._run_with_lifecycle(executor)

    async def _run_continuation(self) -> None:
        async def executor(signal: AbortSignal) -> None:
            await run_agent_loop_continue(
                self._create_context_snapshot(),
                self._create_loop_config(),
                self._process_events,
                signal,
                self.stream_function,
            )

        await self._run_with_lifecycle(executor)

    def _create_context_snapshot(self) -> AgentContext:
        return AgentContext(
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

    def _create_loop_config(self, skip_initial_steering_poll: bool = False) -> AgentLoopConfig:
        skip_poll = skip_initial_steering_poll

        async def get_steering_messages() -> List[AgentMessage]:
            nonlocal skip_poll
            if skip_poll:
                skip_poll = False
                return []
            return self._steering_queue.drain()

        async def get_follow_up_messages() -> List[AgentMessage]:
            return self._follow_up_queue.drain()

        prepare_next_turn = None
        if self.prepare_next_turn_with_context is not None or self.prepare_next_turn is not None:

            async def prepare_next_turn(context: PrepareNextTurnContext) -> Optional[AgentLoopTurnUpdate]:
                if self.prepare_next_turn_with_context is not None:
                    return await _maybe_await(self.prepare_next_turn_with_context(context, self.signal))
                return await _maybe_await(self.prepare_next_turn(self.signal))

        return AgentLoopConfig(
            model=self._state.model,
            reasoning=None if self._state.thinking_level == "off" else self._state.thinking_level,
            session_id=self.session_id,
            on_payload=self.on_payload,
            on_response=self.on_response,
            on_provider_stream_event=self.on_provider_stream_event,
            transport=self.transport,
            thinking_budgets=self.thinking_budgets,
            max_retry_delay_ms=self.max_retry_delay_ms,
            tool_execution=self.tool_execution,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            finish_turn=self.finish_turn,
            prepare_request=self.prepare_request,
            prepare_next_turn=prepare_next_turn,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
        )

    async def _run_with_lifecycle(self, executor: Callable[[AbortSignal], Awaitable[None]]) -> None:
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing.")

        abort_controller = AbortController()
        future: "asyncio.Future[None]" = asyncio.get_running_loop().create_future()
        self._active_run = _ActiveRun(future=future, abort_controller=abort_controller)

        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        try:
            await executor(abort_controller.signal)
        except Exception as error:  # noqa: BLE001 - pi converts any failure into an aborted/error assistant message
            await self._handle_run_failure(error, abort_controller.signal.aborted)
        finally:
            self._finish_run()

    async def _handle_run_failure(self, error: BaseException, aborted: bool) -> None:
        failure_message = AssistantMessage(
            content=[TextContent(text="")],
            api=self._state.model.api,
            provider=self._state.model.provider,
            model=self._state.model.id,
            usage=Usage(),
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
            timestamp=_now_ms(),
        )
        await self._process_events(MessageStartEvent(message=failure_message))
        await self._process_events(MessageEndEvent(message=failure_message))
        await self._process_events(TurnEndEvent(message=failure_message, tool_results=[]))
        await self._process_events(AgentEndEvent(messages=[failure_message]))

    def _finish_run(self) -> None:
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        run = self._active_run
        if run is not None:
            if not run.future.done():
                run.future.set_result(None)
            self._active_run = None

    async def _process_events(self, event: AgentEvent) -> None:
        """Reduce internal state for a loop event, then await listeners.

        ``agent_end`` only means no further loop events will be emitted. The run
        is considered idle later, after all awaited listeners for ``agent_end``
        finish and `_finish_run` clears runtime-owned state.
        """
        event_type = getattr(event, "type", None)
        if event_type == "message_start":
            self._state.streaming_message = event.message
        elif event_type == "message_update":
            self._state.streaming_message = event.message
        elif event_type == "message_end":
            self._state.streaming_message = None
            self._state.messages.append(event.message)
        elif event_type == "tool_execution_start":
            pending_tool_calls = set(self._state.pending_tool_calls)
            pending_tool_calls.add(event.tool_call_id)
            self._state.pending_tool_calls = pending_tool_calls
        elif event_type == "tool_execution_end":
            pending_tool_calls = set(self._state.pending_tool_calls)
            pending_tool_calls.discard(event.tool_call_id)
            self._state.pending_tool_calls = pending_tool_calls
        elif event_type == "turn_end":
            if message_field(event.message, "role") == "assistant":
                error_message = message_field(event.message, "error_message", "errorMessage")
                if error_message:
                    self._state.error_message = error_message
        elif event_type == "agent_end":
            self._state.streaming_message = None

        run = self._active_run
        if run is None:
            raise RuntimeError("Agent listener invoked outside active run")
        signal = run.abort_controller.signal
        for listener in list(self._listeners):
            await _maybe_await(listener(event, signal))

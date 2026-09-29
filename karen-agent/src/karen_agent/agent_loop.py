"""Agent loop that works with AgentMessage throughout, mirroring pi's `agent-loop.ts`.

Transforms to Message[] only at the LLM call boundary. Events are emitted through
an async sink; `agent_loop()` / `agent_loop_continue()` wrap a run in an
EventStream for consumers, exactly like karen-ai's own stream entry points (call
them inside a running event loop).
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
from typing import Any, Callable, List, Optional, Tuple

from karen_ai import (
    AbortSignal,
    AssistantMessage,
    Context,
    EventStream,
    Model,
    SimpleStreamOptions,
    SystemMessage,
    TextContent,
    ToolResultMessage,
    get_current_tools,
    get_tool_state_changes,
    normalize_context,
    to_tool_declaration,
    validate_tool_arguments,
)

from .stream_fn import get_default_stream_fn
from .types import (
    AfterToolCallContext,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentEventSink,
    AgentLoopConfig,
    AgentMessage,
    AgentRequestUpdate,
    AgentStartEvent,
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    AgentTurnContext,
    AgentTurnDecision,
    BeforeToolCallContext,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    PrepareNextTurnContext,
    PrepareRequestContext,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)

async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value




def _get(obj: Any, name: str) -> Any:
    """Field access on hook results: pydantic model, plain object, or dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)

def _now_ms() -> int:
    return int(time.time() * 1000)


def agent_loop(
    prompts: List[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal] = None,
    stream_fn: Optional[StreamFn] = None,
) -> EventStream:
    """Start an agent loop with new prompt messages (added to the context)."""
    stream = _create_agent_stream()

    async def emit(event: AgentEvent) -> None:
        stream.push(event)

    async def runner() -> None:
        messages = await run_agent_loop(prompts, context, config, emit, signal, stream_fn)
        stream.end(messages)

    asyncio.get_running_loop().create_task(runner())
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal] = None,
    stream_fn: Optional[StreamFn] = None,
) -> EventStream:
    """Continue from the current context without adding a new message (retries).

    The last context message must convert to a user or toolResult message via
    `convert_to_llm`; that can only be validated by the provider, not here.
    """
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if getattr(context.messages[-1], "role", None) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream = _create_agent_stream()

    async def emit(event: AgentEvent) -> None:
        stream.push(event)

    async def runner() -> None:
        messages = await run_agent_loop_continue(context, config, emit, signal, stream_fn)
        stream.end(messages)

    asyncio.get_running_loop().create_task(runner())
    return stream


async def run_agent_loop(
    prompts: List[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Optional[AbortSignal],
    stream_fn: Optional[StreamFn],
) -> List[AgentMessage]:
    initial_messages = _declare_tool_changes(context, prompts)
    new_messages: List[AgentMessage] = list(initial_messages)
    current_context = AgentContext(
        messages=[*context.messages, *initial_messages],
        tools=context.tools,
    )

    await _maybe_await(emit(AgentStartEvent()))
    await _maybe_await(emit(TurnStartEvent()))
    for message in initial_messages:
        await _maybe_await(emit(MessageStartEvent(message=message)))
        await _maybe_await(emit(MessageEndEvent(message=message)))

    await _run_loop(current_context, new_messages, config, signal, emit, stream_fn or get_default_stream_fn())
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Optional[AbortSignal],
    stream_fn: Optional[StreamFn],
) -> List[AgentMessage]:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if getattr(context.messages[-1], "role", None) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: List[AgentMessage] = []

    await _maybe_await(emit(AgentStartEvent()))
    await _maybe_await(emit(TurnStartEvent()))

    await _run_loop(context, new_messages, config, signal, emit, stream_fn or get_default_stream_fn())
    return new_messages


def _create_agent_stream() -> EventStream:
    return EventStream(
        lambda event: getattr(event, "type", None) == "agent_end",
        lambda event: event.messages if getattr(event, "type", None) == "agent_end" else [],
    )


async def _run_loop(
    initial_context: AgentContext,
    new_messages: List[AgentMessage],
    initial_config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
    stream_function: StreamFn,
) -> None:
    current_context = initial_context
    config = initial_config
    last_completed_turn: Optional[PrepareNextTurnContext] = None
    explicit_continuation = False
    # Check for steering messages at start (user may have typed while waiting).
    pending_messages: List[AgentMessage] = await _steering(config)

    # Outer loop: continues when queued follow-up messages arrive after the agent would stop.
    while True:
        has_more_tool_calls = True

        # Inner loop: process tool calls and steering messages.
        while has_more_tool_calls or pending_messages:
            prepared_messages: List[AgentMessage] = []
            if last_completed_turn:
                if config.prepare_next_turn:
                    next_turn_snapshot = await _maybe_await(config.prepare_next_turn(last_completed_turn))
                    if next_turn_snapshot:
                        current_context = _get(next_turn_snapshot, "context") or current_context
                        prepared_messages = _get(next_turn_snapshot, "messages") or []
                        config = _apply_thinking_update(
                            config, _get(next_turn_snapshot, "model"), _get(next_turn_snapshot, "thinking_level")
                        )
                # Preparation can be long-running (e.g. compaction); pick up steering queued
                # while it ran. Only poll again if the earlier poll returned nothing.
                if not pending_messages:
                    pending_messages = await _steering(config)
                await _maybe_await(emit(TurnStartEvent()))

            # Process prepared and queued messages before the next assistant response.
            for message in _declare_tool_changes(current_context, [*prepared_messages, *pending_messages]):
                await _maybe_await(emit(MessageStartEvent(message=message)))
                await _maybe_await(emit(MessageEndEvent(message=message)))
                current_context.messages.append(message)
                new_messages.append(message)
            pending_messages = []

            if config.prepare_request:
                request_update = await _maybe_await(
                    config.prepare_request(
                        PrepareRequestContext(
                            context=current_context,
                            model=config.model,
                            thinking_level=config.reasoning or "off",
                        ),
                        signal,
                    )
                )
                if request_update:
                    current_context = _get(request_update, "context") or current_context
                    config = _apply_thinking_update(
                        config, _get(request_update, "model"), _get(request_update, "thinking_level")
                    )

            # Stream the assistant response.
            message = await _stream_assistant_response(current_context, config, signal, emit, stream_function)
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                last_completed_turn = PrepareNextTurnContext(
                    message=message, tool_results=[], context=current_context, new_messages=new_messages
                )
                if config.finish_turn:
                    await _maybe_await(config.finish_turn(last_completed_turn, signal))
                await _maybe_await(emit(TurnEndEvent(message=message, tool_results=[])))
                await _maybe_await(emit(AgentEndEvent(messages=new_messages)))
                return

            # Check for tool calls.
            tool_calls = [c for c in message.content if getattr(c, "type", None) == "toolCall"]

            tool_results: List[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                # A "length" stop means the output was cut off by the token limit, so every
                # tool call may carry truncated arguments. Fail them all instead of
                # executing potentially borked calls.
                if message.stop_reason == "length":
                    executed_batch = await _fail_tool_calls_from_truncated_message(tool_calls, emit)
                else:
                    executed_batch = await _execute_tool_calls(current_context, message, config, signal, emit)
                tool_results.extend(executed_batch[0])
                has_more_tool_calls = not executed_batch[1]

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            last_completed_turn = PrepareNextTurnContext(
                message=message, tool_results=tool_results, context=current_context, new_messages=new_messages
            )
            decision = await _maybe_await(config.finish_turn(last_completed_turn, signal)) if config.finish_turn else None
            await _maybe_await(emit(TurnEndEvent(message=message, tool_results=tool_results)))

            if _get(decision, "action") == "end":
                await _maybe_await(emit(AgentEndEvent(messages=new_messages)))
                return

            explicit_continuation = _get(decision, "action") == "continue"
            pending_messages = await _steering(config)
            if has_more_tool_calls or pending_messages:
                explicit_continuation = False

        # The agent would stop here. Check for follow-up messages.
        follow_up_messages = await _follow_up(config)
        if follow_up_messages:
            explicit_continuation = False
            pending_messages = follow_up_messages
            continue

        # No natural request was selected, so fulfill the continuation decision with
        # one context-only turn.
        if explicit_continuation:
            explicit_continuation = False
            continue

        break

    await _maybe_await(emit(AgentEndEvent(messages=new_messages)))


async def _steering(config: AgentLoopConfig) -> List[AgentMessage]:
    if not config.get_steering_messages:
        return []
    return await _maybe_await(config.get_steering_messages()) or []


async def _follow_up(config: AgentLoopConfig) -> List[AgentMessage]:
    if not config.get_follow_up_messages:
        return []
    return await _maybe_await(config.get_follow_up_messages()) or []


def _apply_thinking_update(
    config: AgentLoopConfig, model: Optional[Model], thinking_level: Optional[str]
) -> AgentLoopConfig:
    """pi's `{...config, model, reasoning}`: undefined thinkingLevel keeps the current
    reasoning, "off" clears it, anything else replaces it."""
    update = {}
    if model is not None:
        update["model"] = model
    if thinking_level is not None:
        update["reasoning"] = None if thinking_level == "off" else thinking_level
    return config.model_copy(update=update)


def _declare_tool_changes(context: AgentContext, pending_messages: List[AgentMessage]) -> List[AgentMessage]:
    """Declare tool loadout changes to the model.

    `context.tools` is what the runtime can execute; the transcript's system messages
    declare what the model may call. Before each request the difference becomes
    tools_added/tools_removed on a system message. When a pending system message exists,
    its tool fields are treated as intent and replaced with the delta between the
    committed transcript and the executable set, so replay always yields exactly
    `context.tools`. Otherwise a new system message is inserted before the first
    non-system pending message.
    """
    system_index = -1
    for i in range(len(pending_messages) - 1, -1, -1):
        if getattr(pending_messages[i], "role", None) == "system":
            system_index = i
            break
    pending: Optional[SystemMessage] = pending_messages[system_index] if system_index >= 0 else None
    baseline = (
        [
            _with_tool_changes(pending, None, None) if index == system_index else message
            for index, message in enumerate(pending_messages)
        ]
        if pending is not None
        else pending_messages
    )
    changes = get_tool_state_changes(
        get_current_tools([*context.messages, *baseline]),
        [to_tool_declaration(t) for t in (context.tools or [])],
    )
    unchanged = not changes.tools_added and not changes.tools_removed

    if pending is not None:
        # Keep the caller's message object when it already declares no tool changes.
        if unchanged and not pending.tools_added and not pending.tools_removed:
            return pending_messages
        return [
            _with_tool_changes(pending, changes.tools_added, changes.tools_removed) if index == system_index else message
            for index, message in enumerate(baseline)
        ]
    if unchanged:
        return pending_messages
    update = _with_tool_changes(
        SystemMessage(content="", timestamp=_now_ms()), changes.tools_added, changes.tools_removed
    )
    insert_index = next((i for i, m in enumerate(pending_messages) if getattr(m, "role", None) != "system"), -1)
    index = insert_index if insert_index >= 0 else len(pending_messages)
    return [*pending_messages[:index], update, *pending_messages[index:]]


def _with_tool_changes(
    message: SystemMessage, tools_added: Optional[List], tools_removed: Optional[List]
) -> SystemMessage:
    """Copy a system message with its tool fields replaced; empty lists omit the field."""
    return message.model_copy(
        update={"tools_added": tools_added or None, "tools_removed": tools_removed or None}
    )


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
    stream_function: StreamFn,
) -> AssistantMessage:
    """Stream an assistant response from the LLM.

    This is where AgentMessage[] gets transformed to Message[] for the LLM.
    """
    # Apply context transform if configured (AgentMessage[] -> AgentMessage[]).
    messages = context.messages
    if config.transform_context:
        messages = await _maybe_await(config.transform_context(messages, signal))

    # Convert to LLM-compatible messages (AgentMessage[] -> Message[]).
    llm_messages = await _maybe_await(config.convert_to_llm(messages))

    llm_context = normalize_context(Context(messages=llm_messages))

    # Resolve the API key (important for expiring tokens).
    resolved_api_key = None
    if config.get_api_key:
        resolved_api_key = await _maybe_await(config.get_api_key(config.model.provider))
    resolved_api_key = resolved_api_key or config.api_key

    response = await _maybe_await(
        stream_function(
            config.model,
            llm_context,
            config.model_copy(update={"api_key": resolved_api_key, "signal": signal}),
        )
    )

    partial_message: Optional[AssistantMessage] = None
    added_partial = False

    async for event in response:
        if event.type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            await _maybe_await(emit(MessageStartEvent(message=copy.copy(partial_message))))
        elif event.type in (
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ):
            if partial_message is not None:
                partial_message = event.partial
                context.messages[-1] = partial_message
                await _maybe_await(
                    emit(MessageUpdateEvent(assistant_message_event=event, message=copy.copy(partial_message)))
                )
        elif event.type in ("done", "error"):
            final_message = await response.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                await _maybe_await(emit(MessageStartEvent(message=copy.copy(final_message))))
            await _maybe_await(emit(MessageEndEvent(message=final_message)))
            return final_message

    final_message = await response.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await _maybe_await(emit(MessageStartEvent(message=copy.copy(final_message))))
    await _maybe_await(emit(MessageEndEvent(message=final_message)))
    return final_message


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

# (messages, terminate)
ExecutedToolCallBatch = Tuple[List[ToolResultMessage], bool]


class _ImmediateToolCallOutcome:
    kind = "immediate"

    def __init__(self, result: AgentToolResult, is_error: bool) -> None:
        self.result = result
        self.is_error = is_error


class _ExecutedToolCallOutcome:
    def __init__(self, result: AgentToolResult, is_error: bool) -> None:
        self.result = result
        self.is_error = is_error


class _PreparedToolCall:
    kind = "prepared"

    def __init__(self, tool_call: AgentToolCall, tool: AgentTool, args: Any) -> None:
        self.tool_call = tool_call
        self.tool = tool
        self.args = args


class _FinalizedToolCallOutcome:
    def __init__(self, tool_call: AgentToolCall, result: AgentToolResult, is_error: bool) -> None:
        self.tool_call = tool_call
        self.result = result
        self.is_error = is_error


async def _fail_tool_calls_from_truncated_message(
    tool_calls: List[AgentToolCall],
    emit: AgentEventSink,
) -> ExecutedToolCallBatch:
    """Fail all tool calls from a message truncated by the output token limit.

    Streamed tool-call arguments are finalized with a best-effort JSON salvage parser,
    so a truncated message can yield tool calls whose arguments parse and validate but
    are silently incomplete. None are safe to execute; report each as an error so the
    model can re-issue them.
    """
    messages: List[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _maybe_await(
            emit(
                ToolExecutionStartEvent(
                    tool_call_id=tool_call.id, tool_name=tool_call.name, args=tool_call.arguments
                )
            )
        )
        finalized = _FinalizedToolCallOutcome(
            tool_call,
            _create_error_tool_result(
                f'Tool call "{tool_call.name}" was not executed: the response hit the output token limit, '
                "so its arguments may be truncated. Re-issue the tool call with complete arguments."
            ),
            True,
        )
        await _emit_tool_execution_end(finalized, emit)
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        messages.append(tool_result_message)
    return messages, False


async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> ExecutedToolCallBatch:
    tool_calls = [c for c in assistant_message.content if getattr(c, "type", None) == "toolCall"]
    has_sequential_tool_call = any(
        _find_tool(current_context, tc.name) is not None
        and _find_tool(current_context, tc.name).execution_mode == "sequential"
        for tc in tool_calls
    )
    if config.tool_execution == "sequential" or has_sequential_tool_call:
        return await _execute_tool_calls_sequential(current_context, assistant_message, tool_calls, config, signal, emit)
    return await _execute_tool_calls_parallel(current_context, assistant_message, tool_calls, config, signal, emit)


def _find_tool(context: AgentContext, name: str) -> Optional[AgentTool]:
    return next((t for t in (context.tools or []) if t.name == name), None)


async def _execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: List[AgentToolCall],
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> ExecutedToolCallBatch:
    finalized_calls: List[_FinalizedToolCallOutcome] = []
    messages: List[ToolResultMessage] = []

    for tool_call in tool_calls:
        await _maybe_await(
            emit(
                ToolExecutionStartEvent(
                    tool_call_id=tool_call.id, tool_name=tool_call.name, args=tool_call.arguments
                )
            )
        )

        preparation = await _prepare_tool_call(current_context, assistant_message, tool_call, config, signal)
        if isinstance(preparation, _ImmediateToolCallOutcome):
            finalized = _FinalizedToolCallOutcome(tool_call, preparation.result, preparation.is_error)
        else:
            executed = await _execute_prepared_tool_call(preparation, signal, emit)
            finalized = await _finalize_executed_tool_call(
                current_context, assistant_message, preparation, executed, config, signal
            )

        await _emit_tool_execution_end(finalized, emit)
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        finalized_calls.append(finalized)
        messages.append(tool_result_message)

        if signal is not None and signal.aborted:
            break

    return messages, _should_terminate_tool_batch(finalized_calls)


async def _execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: List[AgentToolCall],
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> ExecutedToolCallBatch:
    # Entries are either finished outcomes or factories producing them, mirroring
    # pi's `FinalizedToolCallEntry[]` resolved with Promise.all.
    finalized_entries: List[Any] = []

    for tool_call in tool_calls:
        await _maybe_await(
            emit(
                ToolExecutionStartEvent(
                    tool_call_id=tool_call.id, tool_name=tool_call.name, args=tool_call.arguments
                )
            )
        )

        preparation = await _prepare_tool_call(current_context, assistant_message, tool_call, config, signal)
        if isinstance(preparation, _ImmediateToolCallOutcome):
            finalized = _FinalizedToolCallOutcome(tool_call, preparation.result, preparation.is_error)
            await _emit_tool_execution_end(finalized, emit)
            finalized_entries.append(finalized)
            if signal is not None and signal.aborted:
                break
            continue

        def make_entry(prepared: _PreparedToolCall, tc: AgentToolCall) -> Callable:
            async def entry() -> _FinalizedToolCallOutcome:
                if signal is not None and signal.aborted:
                    aborted = _FinalizedToolCallOutcome(tc, _create_error_tool_result("Operation aborted"), True)
                    await _emit_tool_execution_end(aborted, emit)
                    return aborted
                executed = await _execute_prepared_tool_call(prepared, signal, emit)
                finalized = await _finalize_executed_tool_call(
                    current_context, assistant_message, prepared, executed, config, signal
                )
                await _emit_tool_execution_end(finalized, emit)
                return finalized

            return entry

        finalized_entries.append(make_entry(preparation, tool_call))
        if signal is not None and signal.aborted:
            break

    async def resolve(entry: Any) -> _FinalizedToolCallOutcome:
        if callable(entry):
            return await entry()
        return entry

    ordered_finalized_calls = await asyncio.gather(*(resolve(entry) for entry in finalized_entries))
    messages: List[ToolResultMessage] = []
    for finalized in ordered_finalized_calls:
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        messages.append(tool_result_message)

    return messages, _should_terminate_tool_batch(ordered_finalized_calls)


def _should_terminate_tool_batch(finalized_calls: List[_FinalizedToolCallOutcome]) -> bool:
    return bool(finalized_calls) and all(finalized.result.terminate is True for finalized in finalized_calls)


def _prepare_tool_call_arguments(tool: AgentTool, tool_call: AgentToolCall) -> AgentToolCall:
    if not tool.prepare_arguments:
        return tool_call
    prepared_arguments = tool.prepare_arguments(tool_call.arguments)
    if prepared_arguments is tool_call.arguments:
        return tool_call
    return tool_call.model_copy(update={"arguments": prepared_arguments})


async def _prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: AgentToolCall,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
) -> Any:
    tool = _find_tool(current_context, tool_call.name)
    if tool is None:
        return _ImmediateToolCallOutcome(_create_error_tool_result(f"Tool {tool_call.name} not found"), True)

    try:
        prepared_tool_call = _prepare_tool_call_arguments(tool, tool_call)
        validated_args = validate_tool_arguments(tool, prepared_tool_call)
        if config.before_tool_call:
            before_result = await _maybe_await(
                config.before_tool_call(
                    BeforeToolCallContext(
                        assistant_message=assistant_message,
                        tool_call=tool_call,
                        args=validated_args,
                        context=current_context,
                    ),
                    signal,
                )
            )
            if signal is not None and signal.aborted:
                return _ImmediateToolCallOutcome(_create_error_tool_result("Operation aborted"), True)
            if _get(before_result, "block"):
                result = _create_error_tool_result(_get(before_result, "reason") or "Tool execution was blocked")
                if _get(before_result, "terminate") is True:
                    result.terminate = True
                return _ImmediateToolCallOutcome(result, True)
        if signal is not None and signal.aborted:
            return _ImmediateToolCallOutcome(_create_error_tool_result("Operation aborted"), True)
        return _PreparedToolCall(tool_call, tool, validated_args)
    except Exception as error:
        return _ImmediateToolCallOutcome(_create_error_tool_result(str(error)), True)


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    signal: Optional[AbortSignal],
    emit: AgentEventSink,
) -> _ExecutedToolCallOutcome:
    update_events: List[Any] = []
    accepting_updates = True

    def on_update(partial_result: AgentToolResult) -> None:
        if not accepting_updates:
            return
        update_events.append(
            _maybe_await(
                emit(
                    ToolExecutionUpdateEvent(
                        tool_call_id=prepared.tool_call.id,
                        tool_name=prepared.tool_call.name,
                        args=prepared.tool_call.arguments,
                        partial_result=partial_result,
                    )
                )
            )
        )

    try:
        if prepared.tool.execute is None:
            raise RuntimeError(f"Tool {prepared.tool_call.name} has no execute()")
        result = await _maybe_await(prepared.tool.execute(prepared.tool_call.id, prepared.args, signal, on_update))
        accepting_updates = False
        for pending in update_events:
            await pending
        return _ExecutedToolCallOutcome(result, False)
    except Exception as error:
        accepting_updates = False
        for pending in update_events:
            await pending
        return _ExecutedToolCallOutcome(_create_error_tool_result(str(error)), True)
    finally:
        accepting_updates = False


async def _finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    executed: _ExecutedToolCallOutcome,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
) -> _FinalizedToolCallOutcome:
    result = executed.result
    is_error = executed.is_error

    if config.after_tool_call:
        try:
            after_result = await _maybe_await(
                config.after_tool_call(
                    AfterToolCallContext(
                        assistant_message=assistant_message,
                        tool_call=prepared.tool_call,
                        args=prepared.args,
                        result=result,
                        is_error=is_error,
                        context=current_context,
                    ),
                    signal,
                )
            )
            if after_result:
                result = AgentToolResult(
                    content=_get(after_result, "content") if _get(after_result, "content") is not None else result.content,
                    details=_get(after_result, "details") if _get(after_result, "details") is not None else result.details,
                    usage=_get(after_result, "usage") if _get(after_result, "usage") is not None else result.usage,
                    terminate=_get(after_result, "terminate")
                    if _get(after_result, "terminate") is not None
                    else result.terminate,
                )
                is_error = (
                    _get(after_result, "is_error") if _get(after_result, "is_error") is not None else is_error
                )
        except Exception as error:
            result = _create_error_tool_result(str(error))
            is_error = True

    return _FinalizedToolCallOutcome(prepared.tool_call, result, is_error)


def _create_error_tool_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)], details={})


async def _emit_tool_execution_end(finalized: _FinalizedToolCallOutcome, emit: AgentEventSink) -> None:
    await _maybe_await(
        emit(
            ToolExecutionEndEvent(
                tool_call_id=finalized.tool_call.id,
                tool_name=finalized.tool_call.name,
                result=finalized.result,
                is_error=finalized.is_error,
            )
        )
    )


def _create_tool_result_message(finalized: _FinalizedToolCallOutcome) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        # Untyped tools can return results without content; normalize so None never
        # enters session history or provider payloads.
        content=finalized.result.content or [],
        details=finalized.result.details,
        usage=finalized.result.usage,
        is_error=finalized.is_error,
        timestamp=_now_ms(),
    )


async def _emit_tool_result_message(tool_result_message: ToolResultMessage, emit: AgentEventSink) -> None:
    await _maybe_await(emit(MessageStartEvent(message=tool_result_message)))
    await _maybe_await(emit(MessageEndEvent(message=tool_result_message)))


__all__ = [
    "agent_loop",
    "agent_loop_continue",
    "run_agent_loop",
    "run_agent_loop_continue",
]

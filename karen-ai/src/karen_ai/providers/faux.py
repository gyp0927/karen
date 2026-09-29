"""Scripted faux provider for tests, mirroring pi-ai's providers/faux.ts.

Streams scripted AssistantMessages through the real event protocol: content
blocks are chopped into small chunks and emitted as *_start/*_delta/*_end
events, so tests exercise the same code paths as real adapters.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from ..auth.types import ApiKeyAuth, ApiKeyResolveInput, AuthResult, ModelAuth, ProviderAuth
from ..event_stream import AssistantMessageEventStream
from ..lazy import ProviderStreams
from ..models import CreateProviderOptions, Provider, create_provider
from ..types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TranscriptContext,
    Usage,
)

DEFAULT_API = "faux"
DEFAULT_PROVIDER = "faux"
DEFAULT_MODEL_ID = "faux-1"
DEFAULT_MODEL_NAME = "Faux Model"
DEFAULT_BASE_URL = "http://localhost:0"


def faux_text(text: str) -> TextContent:
    return TextContent(text=text)


def faux_thinking(thinking: str, signature: Optional[str] = None) -> ThinkingContent:
    return ThinkingContent(thinking=thinking, thinking_signature=signature)


def faux_tool_call(name: str, arguments: Dict[str, Any], id: Optional[str] = None) -> ToolCall:
    return ToolCall(id=id or f"tool_{uuid.uuid4().hex[:12]}", name=name, arguments=arguments)


def faux_assistant_message(
    content: Union[str, List],
    *,
    stop_reason: str = "stop",
    error_message: Optional[str] = None,
    response_id: Optional[str] = None,
    timestamp: Optional[int] = None,
    api: str = DEFAULT_API,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL_ID,
) -> AssistantMessage:
    blocks = [faux_text(content)] if isinstance(content, str) else content
    return AssistantMessage(
        role="assistant",
        content=blocks,
        api=api,
        provider=provider,
        model=model,
        usage=Usage(),
        stop_reason=stop_reason,  # type: ignore[arg-type]
        error_message=error_message,
        response_id=response_id,
        timestamp=timestamp if timestamp is not None else int(time.time() * 1000),
    )


def faux_model(
    id: str = DEFAULT_MODEL_ID,
    *,
    name: str = DEFAULT_MODEL_NAME,
    reasoning: bool = False,
    provider: str = DEFAULT_PROVIDER,
    api: str = DEFAULT_API,
    context_window: int = 128_000,
    max_tokens: int = 8192,
    input: Optional[List[str]] = None,
) -> Model:
    return Model(
        id=id,
        name=name,
        api=api,
        provider=provider,
        base_url=DEFAULT_BASE_URL,
        input=input or ["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        reasoning=reasoning,
        context_window=context_window,
        max_tokens=max_tokens,
    )


class FauxProviderState:
    def __init__(self) -> None:
        self.call_count = 0
        self.received_options: List[Any] = []


#: A scripted response: a ready message, or a factory producing one.
FauxResponseStep = Union[AssistantMessage, Callable[..., AssistantMessage]]

_CHUNK_MIN = 3
_CHUNK_MAX = 5


def _chunks(text: str) -> List[str]:
    if not text:
        return []
    size = max(_CHUNK_MIN, min(_CHUNK_MAX, len(text) // 3 or _CHUNK_MIN))
    return [text[i : i + size] for i in range(0, len(text), size)]


class FauxProviderRegistration:
    def __init__(
        self,
        provider: Provider,
        models: List[Model],
        state: FauxProviderState,
        set_responses,
        append_responses,
        pending_count,
        api: str = DEFAULT_API,
        unregister: Optional[Callable[[], None]] = None,
    ) -> None:
        self.provider = provider
        self.api = api
        self.models = models
        self.state = state
        self.set_responses = set_responses
        self.append_responses = append_responses
        self.get_pending_response_count = pending_count
        self.unregister = unregister or (lambda: None)

    def get_model(self, model_id: Optional[str] = None) -> Optional[Model]:
        if model_id is None:
            return self.models[0] if self.models else None
        for model in self.models:
            if model.id == model_id:
                return model
        return None


def register_faux_provider(
    *,
    api: str = DEFAULT_API,
    provider_id: str = DEFAULT_PROVIDER,
    models: Optional[List[Model]] = None,
    responses: Optional[Sequence[FauxResponseStep]] = None,
    token_delay: float = 0.0,
) -> FauxProviderRegistration:
    """Build a scripted faux provider. Queue responses with `registration.set_responses`."""
    state = FauxProviderState()
    response_queue: List[FauxResponseStep] = list(responses or [])

    model_list = models or [faux_model(provider=provider_id, api=api)]

    def set_responses(steps: Sequence[FauxResponseStep]) -> None:
        response_queue.clear()
        response_queue.extend(steps)

    def append_responses(steps: Sequence[FauxResponseStep]) -> None:
        response_queue.extend(steps)

    def pending_count() -> int:
        return len(response_queue)

    def next_response(context: TranscriptContext, options, model: Model) -> AssistantMessage:
        if not response_queue:
            raise RuntimeError("faux provider: no scripted responses left")
        step = response_queue.pop(0)
        if callable(step):
            return step(context, options, state, model)
        return step

    def make_stream(simple: bool):
        def do_stream(model: Model, context: TranscriptContext, options=None) -> AssistantMessageEventStream:
            event_stream = AssistantMessageEventStream()
            state.call_count += 1
            state.received_options.append(options)

            async def run() -> None:
                try:
                    scripted = next_response(context, options, model)
                    output = scripted.model_copy(deep=True)
                    output.api = model.api
                    output.provider = model.provider
                    output.model = model.id

                    event_stream.push(StartEvent(partial=output))
                    content = output.content
                    # Stream blocks one by one through the event protocol.
                    final_blocks: List = [None] * len(content)
                    for index, block in enumerate(content):
                        if token_delay:
                            await asyncio.sleep(token_delay)
                        if isinstance(block, TextContent):
                            text = block.text
                            block.text = ""
                            event_stream.push(TextStartEvent(content_index=index, partial=output))
                            for chunk in _chunks(text):
                                block.text += chunk
                                event_stream.push(TextDeltaEvent(content_index=index, delta=chunk, partial=output))
                            event_stream.push(TextEndEvent(content_index=index, content=block.text, partial=output))
                        elif isinstance(block, ThinkingContent):
                            thinking = block.thinking
                            block.thinking = ""
                            event_stream.push(ThinkingStartEvent(content_index=index, partial=output))
                            for chunk in _chunks(thinking):
                                block.thinking += chunk
                                event_stream.push(ThinkingDeltaEvent(content_index=index, delta=chunk, partial=output))
                            event_stream.push(ThinkingEndEvent(content_index=index, content=block.thinking, partial=output))
                        elif isinstance(block, ToolCall):
                            import json as _json

                            args_json = _json.dumps(block.arguments)
                            block.arguments = {}
                            event_stream.push(ToolCallStartEvent(content_index=index, partial=output))
                            for chunk in _chunks(args_json):
                                event_stream.push(ToolCallDeltaEvent(content_index=index, delta=chunk, partial=output))
                            block.arguments = _json.loads(args_json)
                            event_stream.push(ToolCallEndEvent(content_index=index, tool_call=block, partial=output))
                        final_blocks[index] = block

                    output.content = [b for b in final_blocks if b is not None]
                    if output.stop_reason in ("error", "aborted"):
                        event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))  # type: ignore[arg-type]
                    else:
                        event_stream.push(DoneEvent(reason=output.stop_reason, message=output))  # type: ignore[arg-type]
                    event_stream.end()
                except Exception as error:
                    failure = faux_assistant_message(
                        [], stop_reason="error", error_message=str(error), api=model.api, provider=model.provider, model=model.id
                    )
                    event_stream.push(ErrorEvent(reason="error", error=failure))
                    event_stream.end(failure)

            asyncio.get_running_loop().create_task(run())
            return event_stream

        return do_stream

    async def resolve(input: ApiKeyResolveInput) -> Optional[AuthResult]:
        return AuthResult(auth=ModelAuth(api_key="faux"), source="faux")

    provider = create_provider(
        CreateProviderOptions(
            id=provider_id,
            name="Faux",
            base_url=DEFAULT_BASE_URL,
            auth=ProviderAuth(api_key=ApiKeyAuth(name="Faux", resolve=resolve)),
            models=model_list,
            api=ProviderStreams(stream=make_stream(simple=False), stream_simple=make_stream(simple=True)),
        )
    )

    registration = FauxProviderRegistration(
        provider, model_list, state, set_responses, append_responses, pending_count, api=api
    )
    return registration

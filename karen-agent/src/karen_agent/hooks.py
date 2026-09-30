"""Ordered harness hook registry and aggregate runner (pi's `harness/hooks.ts`).

Eleven hooks let an application observe and steer a run; multiple handlers can
be registered per hook and each hook has its own aggregation semantics
(chaining, field-wise merge, first-structural-wins, fail-closed, ...).

Deviations from pi, all consequences of machinery karen does not have:

- pi runs handlers through effect ``Gate``s and one telemetry span per tool-hook
  handler, and every event carries ``lane``/``runId``; karen drops lanes, gates
  and harness telemetry, so events are exactly the ``HookMap`` payloads.
- pi handlers receive the chord ``Context`` as a second argument; karen
  handlers take just the event (async or sync return).
- Handler results are typed models instead of ad-hoc object literals. "Field
  absent" maps to "field not explicitly set" (pydantic ``model_fields_set``),
  and in ``StreamOptionsPatch`` an explicitly-set ``None`` deletes the key
  (JS ``undefined`` semantics); inside ``headers``/``metadata`` patch dicts a
  ``None`` value deletes that entry.
"""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union

from karen_ai import AssistantMessage, Model, Usage
from karen_ai.types import CacheRetention, KarenBase, Transport
from pydantic import ConfigDict

from .compaction import BranchPreparation, BranchSummaryResult, CompactResult, CompactionPreparation
from .resources import Resources
from .result import to_error
from .types import AgentMessage

__all__ = [
    "HookName",
    "HOOK_NAMES",
    "HookHandler",
    "HookErrorReporter",
    "HookRegistry",
    "HarnessStreamOptions",
    "StreamOptionsPatch",
    "apply_stream_options_patch",
    "create_stream_options_patch",
    "BeforeRunEvent",
    "BeforeRunResult",
    "BeforeDriveEvent",
    "BeforeRunEndEvent",
    "BeforeRunEndResult",
    "TransformContextEvent",
    "TransformContextResult",
    "BeforeRequestEvent",
    "BeforeRequestResult",
    "BeforePayloadEvent",
    "BeforePayloadResult",
    "AfterResponseEvent",
    "AfterResponseResult",
    "BeforeToolEvent",
    "ToolBlock",
    "BeforeToolResult",
    "AfterToolEvent",
    "AfterToolResult",
    "BeforeCompactionEvent",
    "BeforeCompactionResult",
    "BeforeNavigationEvent",
    "BeforeNavigationResult",
]

HookName = Literal[
    "before_run",
    "before_drive",
    "before_run_end",
    "transform_context",
    "before_request",
    "before_payload",
    "after_response",
    "before_tool",
    "after_tool",
    "before_compaction",
    "before_navigation",
]

HOOK_NAMES = (
    "before_run",
    "before_drive",
    "before_run_end",
    "transform_context",
    "before_request",
    "before_payload",
    "after_response",
    "before_tool",
    "after_tool",
    "before_compaction",
    "before_navigation",
)

#: Handlers receive the hook event and may return a result model or None,
#: synchronously or as an awaitable.
HookHandler = Callable[[Any], Union[Any, Awaitable[Any]]]

#: Receives (error, hook_name) for every handler failure.
HookErrorReporter = Callable[[Exception, str], Union[None, Awaitable[None]]]

_DeferredOption = Union[bool, Dict[str, Literal["15m", "1h", "24h"]]]


class HarnessStreamOptions(KarenBase):
    """Curated stream options exposed to ``before_request`` hooks (pi's ``AgentHarnessStreamOptions``)."""

    #: Preferred transport forwarded to the stream function.
    transport: Optional[Transport] = None
    #: Provider request timeout in milliseconds.
    timeout_ms: Optional[int] = None
    #: Maximum provider retry attempts.
    max_retries: Optional[int] = None
    #: Optional cap for provider-requested retry delays.
    max_retry_delay_ms: Optional[int] = None
    #: Additional request headers merged with auth and lifecycle headers.
    headers: Optional[Dict[str, str]] = None
    #: Provider metadata forwarded with requests.
    metadata: Optional[Dict[str, Any]] = None
    #: Provider cache retention hint.
    cache_retention: Optional[CacheRetention] = None
    #: Ask a capable provider to continue generation asynchronously.
    deferred: Optional[_DeferredOption] = None


class StreamOptionsPatch(KarenBase):
    """Partial :class:`HarnessStreamOptions` update.

    Only explicitly-set fields apply; an explicitly-set ``None`` deletes the
    key on the target (JS ``undefined``). In ``headers``/``metadata`` dicts a
    ``None`` value deletes that entry; setting the whole field to ``None``
    clears it.
    """

    transport: Optional[Transport] = None
    timeout_ms: Optional[int] = None
    max_retries: Optional[int] = None
    max_retry_delay_ms: Optional[int] = None
    cache_retention: Optional[CacheRetention] = None
    deferred: Optional[_DeferredOption] = None
    headers: Optional[Dict[str, Optional[str]]] = None
    metadata: Optional[Dict[str, Any]] = None


_SCALAR_OPTION_KEYS = ("transport", "timeout_ms", "max_retries", "max_retry_delay_ms", "cache_retention", "deferred")


def apply_stream_options_patch(base: HarnessStreamOptions, patch: StreamOptionsPatch) -> HarnessStreamOptions:
    """Apply a patch, returning a new options object (the base is not mutated)."""
    update: Dict[str, Any] = {}
    set_fields = patch.model_fields_set
    for key in _SCALAR_OPTION_KEYS:
        if key in set_fields:
            update[key] = getattr(patch, key)
    if "headers" in set_fields:
        if patch.headers is None:
            update["headers"] = None
        else:
            headers = dict(base.headers or {})
            for key, value in patch.headers.items():
                if value is None:
                    headers.pop(key, None)
                else:
                    headers[key] = value
            update["headers"] = headers
    if "metadata" in set_fields:
        if patch.metadata is None:
            update["metadata"] = None
        else:
            metadata = dict(base.metadata or {})
            for key, value in patch.metadata.items():
                if value is None:
                    metadata.pop(key, None)
                else:
                    metadata[key] = value
            update["metadata"] = metadata
    return base.model_copy(update=update)


def create_stream_options_patch(base: HarnessStreamOptions, value: HarnessStreamOptions) -> StreamOptionsPatch:
    """Compute the patch that turns ``base`` into ``value``."""
    patch_fields: Dict[str, Any] = {}
    for key in _SCALAR_OPTION_KEYS:
        if getattr(base, key) != getattr(value, key):
            patch_fields[key] = getattr(value, key)
    for key in ("headers", "metadata"):
        base_dict = getattr(base, key)
        value_dict = getattr(value, key)
        if base_dict is value_dict:
            continue
        if value_dict is None:
            patch_fields[key] = None
            continue
        diff: Dict[str, Any] = {}
        for old_key in base_dict or {}:
            if old_key not in value_dict:
                diff[old_key] = None
        for new_key, new_value in value_dict.items():
            if (base_dict or {}).get(new_key) != new_value:
                diff[new_key] = new_value
        if base_dict is None and not diff:
            patch_fields[key] = {}
        elif diff:
            patch_fields[key] = diff
    return StreamOptionsPatch(**patch_fields)


# ---------------------------------------------------------------------------
# Hook events and results (pi's HookMap)
# ---------------------------------------------------------------------------


class BeforeRunEvent(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: List[AgentMessage]
    resources: Optional[Resources] = None


class BeforeRunResult(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: Optional[List[AgentMessage]] = None


class BeforeDriveEvent(KarenBase):
    operation: Literal["run", "compaction", "navigation"]


class BeforeRunEndEvent(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    messages: List[AgentMessage]


class BeforeRunEndResult(KarenBase):
    follow_up: Optional[str] = None


class TransformContextEvent(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: List[AgentMessage]
    system_prompt: str


class TransformContextResult(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: Optional[List[AgentMessage]] = None
    system_prompt: Optional[str] = None


class BeforeRequestEvent(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    step: Literal["assistant", "deferred", "compaction", "branch_summary"]
    attempt: int
    stream_options: HarnessStreamOptions


class BeforeRequestResult(KarenBase):
    stream_options: Optional[StreamOptionsPatch] = None


class BeforePayloadEvent(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    payload: Any = None


class BeforePayloadResult(KarenBase):
    payload: Any


class AfterResponseEvent(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: Optional[int] = None
    headers: Optional[Dict[str, str]] = None
    message: AssistantMessage


class AfterResponseResult(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: Optional[AssistantMessage] = None


class BeforeToolEvent(KarenBase):
    tool_call_id: str
    tool_name: str
    args: Dict[str, Any]


class ToolBlock(KarenBase):
    reason: str
    terminate: Optional[bool] = None


class BeforeToolResult(KarenBase):
    args: Optional[Dict[str, Any]] = None
    block: Optional[ToolBlock] = None


class AfterToolEvent(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_call_id: str
    tool_name: str
    args: Dict[str, Any]
    content: Any  # AgentToolResult content blocks
    details: Optional[Any] = None
    is_error: bool
    usage: Optional[Usage] = None


class AfterToolResult(KarenBase):
    """Field-by-field overrides; fields not explicitly set keep the current values."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: Optional[Any] = None
    details: Optional[Any] = None
    is_error: Optional[bool] = None
    usage: Optional[Usage] = None
    terminate: Optional[bool] = None


class BeforeCompactionEvent(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    reason: Literal["manual", "threshold", "overflow"]
    preparation: CompactionPreparation
    custom_instructions: Optional[str] = None


class BeforeCompactionResult(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    decline: Optional[bool] = None
    compaction: Optional[CompactResult] = None


class BeforeNavigationEvent(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_id: str
    preparation: BranchPreparation
    custom_instructions: Optional[str] = None


class BeforeNavigationResult(KarenBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    decline: Optional[bool] = None
    summary: Optional[BranchSummaryResult] = None


class _HookRegistration:
    __slots__ = ("id", "handler")

    def __init__(self, id: Optional[str], handler: HookHandler) -> None:
        self.id = id
        self.handler = handler


class HookRegistry:
    """Ordered harness hook registry and aggregate runner."""

    def __init__(self, report_error: HookErrorReporter) -> None:
        self._report_error = report_error
        self._registrations: Dict[str, List[_HookRegistration]] = {}
        self._closed_error: Optional[Exception] = None

    def on(self, name: HookName, handler: HookHandler, *, id: Optional[str] = None) -> Callable[[], None]:
        """Register a handler; returns an unsubscribe function."""
        if self._closed_error is not None:
            raise self._closed_error
        if name not in HOOK_NAMES:
            raise ValueError(f"Unknown hook: {name!r}")
        registrations = self._registrations.setdefault(name, [])
        registration = _HookRegistration(id=id, handler=handler)
        registrations.append(registration)

        def unsubscribe() -> None:
            if registration in registrations:
                registrations.remove(registration)

        return unsubscribe

    def has(self, name: HookName) -> bool:
        return bool(self._registrations.get(name))

    def close(self, error: Exception) -> None:
        """Close the registry; the first close wins. Later on()/run() raise it."""
        if self._closed_error is None:
            self._closed_error = error

    async def run(self, name: HookName, event: Any) -> Any:
        """Invoke the aggregate for one hook."""
        if self._closed_error is not None:
            raise self._closed_error
        return await self._aggregate(name, event)

    # -- dispatch -----------------------------------------------------------

    async def _aggregate(self, name: HookName, event: Any) -> Any:
        if name == "before_run":
            return await self._before_run(event)
        if name == "before_drive":
            await self._invoke_all_fail_closed(name, event)
            return None
        if name == "before_run_end":
            follow_up: Optional[str] = None
            follow_up_set = False
            for registration in self._registrations_for(name):
                try:
                    result = await self._invoke(registration, event)
                    if result is not None and result.follow_up is not None:
                        follow_up = result.follow_up
                        follow_up_set = True
                except Exception as error:
                    await self._report(error, name)
            return BeforeRunEndResult(follow_up=follow_up) if follow_up_set else None
        if name == "transform_context":
            return await self._transform_context(event)
        if name == "before_request":
            return await self._before_request(event)
        if name == "before_payload":
            return await self._chain_field(name, event, "payload", BeforePayloadResult)
        if name == "after_response":
            return await self._chain_field(name, event, "message", AfterResponseResult)
        if name == "before_tool":
            return await self._before_tool(event)
        if name == "after_tool":
            return await self._after_tool(event)
        if name == "before_compaction":
            return await self._first_structural(name, event, "compaction")
        if name == "before_navigation":
            return await self._first_structural(name, event, "summary")
        raise ValueError(f"Unknown hook: {name!r}")

    # -- per-hook aggregates -------------------------------------------------

    async def _before_run(self, event: BeforeRunEvent) -> Optional[BeforeRunResult]:
        prompt = event.prompt
        injected: List[AgentMessage] = []
        for registration in self._registrations_for("before_run"):
            try:
                result = await self._invoke(registration, event.model_copy(update={"prompt": prompt}))
                if result is not None and result.messages is not None:
                    injected = [*injected, *result.messages]
                    prompt = [*prompt, *result.messages]
            except Exception as error:
                await self._report(error, "before_run")
        return None if not injected else BeforeRunResult(messages=injected)

    async def _before_tool(self, event: BeforeToolEvent) -> BeforeToolResult:
        args = event.args
        block: Optional[ToolBlock] = None
        for registration in self._registrations_for("before_tool"):
            try:
                result = await self._invoke(registration, event.model_copy(update={"args": args}))
                if result is not None:
                    if result.args is not None:
                        args = result.args
                    if result.block is not None:
                        block = result.block
                        break
            except Exception as error:
                normalized = to_error(error)
                await self._report(normalized, "before_tool")
                block = ToolBlock(reason=str(normalized))
                break
        return BeforeToolResult(
            args=None if args is event.args else args,
            block=block,
        )

    async def _transform_context(self, event: TransformContextEvent) -> TransformContextResult:
        messages = event.messages
        system_prompt = event.system_prompt
        for registration in self._registrations_for("transform_context"):
            try:
                result = await self._invoke(
                    registration,
                    event.model_copy(update={"messages": messages, "system_prompt": system_prompt}),
                )
                if result is not None:
                    if result.messages is not None:
                        messages = result.messages
                    if result.system_prompt is not None:
                        system_prompt = result.system_prompt
            except Exception as error:
                await self._report(error, "transform_context")
        return TransformContextResult(messages=messages, system_prompt=system_prompt)

    async def _before_request(self, event: BeforeRequestEvent) -> Optional[BeforeRequestResult]:
        stream_options = event.stream_options
        changed = False
        for registration in self._registrations_for("before_request"):
            try:
                result = await self._invoke(
                    registration, event.model_copy(update={"stream_options": stream_options})
                )
                if result is not None and result.stream_options is not None:
                    stream_options = apply_stream_options_patch(stream_options, result.stream_options)
                    changed = True
            except Exception as error:
                await self._report(error, "before_request")
        if not changed:
            return None
        return BeforeRequestResult(
            stream_options=create_stream_options_patch(event.stream_options, stream_options)
        )

    async def _chain_field(self, name: str, event: Any, field: str, result_type: Any) -> Any:
        """before_payload / after_response: chain one field, always return it."""
        current = getattr(event, field)
        for registration in self._registrations_for(name):
            try:
                result = await self._invoke(registration, event.model_copy(update={field: current}))
                if result is not None:
                    value = getattr(result, field, None)
                    if value is not None:
                        current = value
            except Exception as error:
                await self._report(error, name)
        return result_type(**{field: current})

    async def _after_tool(self, event: AfterToolEvent) -> Optional[AfterToolResult]:
        current = {
            "content": event.content,
            "details": event.details,
            "is_error": event.is_error,
            "usage": event.usage,
        }
        aggregate: Dict[str, Any] = {}
        for registration in self._registrations_for("after_tool"):
            try:
                result = await self._invoke(registration, event.model_copy(update=current))
                if result is None:
                    continue
                set_fields = result.model_fields_set
                for field in ("content", "details", "is_error", "usage", "terminate"):
                    if field in set_fields:
                        aggregate[field] = getattr(result, field)
                current = {
                    key: getattr(result, key) if key in set_fields else current[key]
                    for key in ("content", "details", "is_error", "usage")
                }
            except Exception as error:
                await self._report(error, "after_tool")
        return AfterToolResult(**aggregate) if aggregate else None

    async def _first_structural(
        self,
        name: Literal["before_compaction", "before_navigation"],
        event: Any,
        result_field: Literal["compaction", "summary"],
    ) -> Any:
        """First handler returning decline=True xor a structural result wins."""
        for registration in self._registrations_for(name):
            try:
                result = await self._invoke(registration, event)
                if result is None:
                    continue
                decline = result.decline is True
                structural = getattr(result, result_field) is not None
                if decline and structural:
                    await self._report(
                        Exception(f"{name} hook cannot return both decline and {result_field}"), name
                    )
                    continue
                if decline or structural:
                    return result
            except Exception as error:
                await self._report(error, name)
        return None

    async def _invoke_all_fail_closed(self, name: HookName, event: Any) -> None:
        for registration in self._registrations_for(name):
            try:
                await self._invoke(registration, event)
            except Exception as error:
                normalized = to_error(error)
                await self._report(normalized, name)
                raise normalized

    # -- plumbing ------------------------------------------------------------

    def _registrations_for(self, name: HookName) -> List[_HookRegistration]:
        return list(self._registrations.get(name) or [])

    async def _invoke(self, registration: _HookRegistration, event: Any) -> Any:
        value = registration.handler(event)
        if inspect.isawaitable(value):
            value = await value
        return value

    async def _report(self, error: Any, name: HookName) -> None:
        normalized = to_error(error)
        result = self._report_error(normalized, name)
        if inspect.isawaitable(result):
            await result

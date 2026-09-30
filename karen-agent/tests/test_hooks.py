"""HookRegistry aggregation semantics (pi's hooks.ts)."""

import pytest
from karen_ai import UserMessage
from karen_ai.providers import faux_assistant_message, faux_model

from karen_agent.compaction import BranchPreparation, CompactionPreparation, CompactionSettings, create_file_ops
from karen_agent.hooks import (
    AfterResponseEvent,
    AfterResponseResult,
    AfterToolEvent,
    AfterToolResult,
    BeforeCompactionEvent,
    BeforeCompactionResult,
    BeforeDriveEvent,
    BeforeNavigationEvent,
    BeforeNavigationResult,
    BeforePayloadEvent,
    BeforePayloadResult,
    BeforeRequestEvent,
    BeforeRequestResult,
    BeforeRunEndEvent,
    BeforeRunEndResult,
    BeforeRunEvent,
    BeforeRunResult,
    BeforeToolEvent,
    BeforeToolResult,
    CompactResult,
    HarnessStreamOptions,
    HookRegistry,
    StreamOptionsPatch,
    ToolBlock,
    TransformContextEvent,
    TransformContextResult,
    apply_stream_options_patch,
    create_stream_options_patch,
)


class Recorder:
    def __init__(self):
        self.errors = []

    def __call__(self, error, hook):
        self.errors.append((str(error), hook))


def make_registry():
    recorder = Recorder()
    return recorder, HookRegistry(recorder)


def _compaction_preparation():
    return CompactionPreparation(
        messages_to_summarize=[],
        turn_prefix_messages=[],
        retained_tail=[],
        is_split_turn=False,
        tokens_before=0,
        file_ops=create_file_ops(),
        settings=CompactionSettings(),
    )


def _branch_preparation():
    return BranchPreparation(messages=[], file_ops=create_file_ops(), total_tokens=0)


def test_on_has_unsubscribe():
    _, registry = make_registry()
    assert not registry.has("before_run")
    unsub = registry.on("before_run", lambda event: None)
    assert registry.has("before_run")
    unsub()
    assert not registry.has("before_run")
    with pytest.raises(ValueError):
        registry.on("nope", lambda event: None)


async def test_close_raises_on_later_calls():
    _, registry = make_registry()
    registry.close(Exception("shutdown"))
    with pytest.raises(Exception, match="shutdown"):
        registry.on("before_run", lambda event: None)
    with pytest.raises(Exception, match="shutdown"):
        await registry.run("before_run", BeforeRunEvent(prompt=[]))


async def test_before_run_chains_prompt_and_collects_injected():
    recorder, registry = make_registry()
    seen_prompts = []

    def first(event):
        seen_prompts.append(len(event.prompt))
        return BeforeRunResult(messages=[UserMessage(content="injected-1", timestamp=0)])

    def second(event):
        seen_prompts.append(len(event.prompt))
        return BeforeRunResult(messages=[UserMessage(content="injected-2", timestamp=0)])

    registry.on("before_run", first)
    registry.on("before_run", second)
    result = await registry.run("before_run", BeforeRunEvent(prompt=[UserMessage(content="orig", timestamp=0)]))
    assert seen_prompts == [1, 2]  # second handler sees the first's injection
    assert [m.content for m in result.messages] == ["injected-1", "injected-2"]


async def test_before_run_no_messages_returns_none_and_errors_are_reported():
    recorder, registry = make_registry()

    def failing(event):
        raise RuntimeError("hook broke")

    registry.on("before_run", failing)
    registry.on("before_run", lambda event: None)
    result = await registry.run("before_run", BeforeRunEvent(prompt=[]))
    assert result is None
    assert recorder.errors == [("hook broke", "before_run")]


async def test_before_drive_fail_closed():
    recorder, registry = make_registry()
    calls = []

    def failing(event):
        calls.append("failing")
        raise RuntimeError("stop the run")

    registry.on("before_drive", lambda event: calls.append("first"))
    registry.on("before_drive", failing)
    registry.on("before_drive", lambda event: calls.append("never"))
    with pytest.raises(RuntimeError, match="stop the run"):
        await registry.run("before_drive", BeforeDriveEvent(operation="run"))
    assert calls == ["first", "failing"]
    assert recorder.errors == [("stop the run", "before_drive")]


async def test_before_run_end_last_follow_up_wins():
    _, registry = make_registry()
    registry.on("before_run_end", lambda event: BeforeRunEndResult(follow_up="first"))
    registry.on("before_run_end", lambda event: None)
    registry.on("before_run_end", lambda event: BeforeRunEndResult(follow_up="last"))
    result = await registry.run("before_run_end", BeforeRunEndEvent(run_id="r", messages=[]))
    assert result.follow_up == "last"

    _, empty = make_registry()
    assert await empty.run("before_run_end", BeforeRunEndEvent(run_id="r", messages=[])) is None


async def test_transform_context_chains_both_fields():
    _, registry = make_registry()
    registry.on(
        "transform_context",
        lambda event: TransformContextResult(messages=event.messages + [UserMessage(content="added", timestamp=0)]),
    )
    registry.on("transform_context", lambda event: TransformContextResult(system_prompt=event.system_prompt + " B"))
    registry.on("transform_context", lambda event: TransformContextResult(system_prompt=event.system_prompt + " C"))
    result = await registry.run(
        "transform_context", TransformContextEvent(messages=[], system_prompt="A")
    )
    assert [m.content for m in result.messages] == ["added"]
    assert result.system_prompt == "A B C"


async def test_before_request_chains_patches_and_returns_diff():
    _, registry = make_registry()
    base = HarnessStreamOptions(timeout_ms=1000, headers={"a": "1", "b": "2"})
    registry.on(
        "before_request",
        lambda event: BeforeRequestResult(
            stream_options=StreamOptionsPatch(transport="sse", headers={"b": None, "c": "3"})
        ),
    )
    registry.on(
        "before_request",
        lambda event: BeforeRequestResult(stream_options=StreamOptionsPatch(max_retries=5)),
    )
    event = BeforeRequestEvent(model=faux_model(), step="assistant", attempt=1, stream_options=base)
    result = await registry.run("before_request", event)
    patch = result.stream_options
    assert patch.transport == "sse"
    assert patch.max_retries == 5
    assert patch.headers == {"b": None, "c": "3"}
    assert "timeout_ms" not in patch.model_fields_set  # unchanged


async def test_before_request_no_change_returns_none():
    _, registry = make_registry()
    registry.on("before_request", lambda event: None)
    event = BeforeRequestEvent(model=faux_model(), step="assistant", attempt=1, stream_options=HarnessStreamOptions())
    assert await registry.run("before_request", event) is None


def test_stream_options_patch_scalar_and_deletes():
    base = HarnessStreamOptions(transport="sse", timeout_ms=5, headers={"a": "1"}, metadata={"m": 1})
    patched = apply_stream_options_patch(base, StreamOptionsPatch(transport=None, max_retries=3))
    assert patched.transport is None  # explicit None deletes
    assert patched.max_retries == 3
    assert patched.timeout_ms == 5
    assert base.transport == "sse"  # base not mutated

    # headers: merge with per-key delete; metadata: clear-all via None
    patched2 = apply_stream_options_patch(
        base, StreamOptionsPatch(headers={"a": None, "b": "2"}, metadata=None)
    )
    assert patched2.headers == {"b": "2"}
    assert patched2.metadata is None


def test_create_stream_options_patch_diff():
    base = HarnessStreamOptions(transport="sse", headers={"a": "1", "gone": "x"}, metadata={"k": 1})
    value = HarnessStreamOptions(transport="websocket", headers={"a": "1", "b": "2"}, metadata={"k": 1})
    patch = create_stream_options_patch(base, value)
    assert patch.model_dump(exclude_unset=True) == {
        "transport": "websocket",
        "headers": {"gone": None, "b": "2"},
    }
    # identical -> empty patch
    same = create_stream_options_patch(base, base.model_copy(update={}))
    assert same.model_dump(exclude_unset=True) == {}
    # headers cleared entirely
    cleared = create_stream_options_patch(base, HarnessStreamOptions(transport="sse", metadata={"k": 1}))
    assert cleared.model_dump(exclude_unset=True)["headers"] is None


async def test_before_payload_chains():
    _, registry = make_registry()
    registry.on("before_payload", lambda event: BeforePayloadResult(payload={**event.payload, "a": 1}))
    registry.on("before_payload", lambda event: BeforePayloadResult(payload={**event.payload, "b": 2}))
    result = await registry.run("before_payload", BeforePayloadEvent(model=faux_model(), payload={}))
    assert result.payload == {"a": 1, "b": 2}


async def test_after_response_chains_message():
    _, registry = make_registry()
    original = faux_assistant_message("v1")
    replacement = faux_assistant_message("v2")
    registry.on("after_response", lambda event: AfterResponseResult(message=replacement))
    result = await registry.run(
        "after_response", AfterResponseEvent(status=200, message=original)
    )
    assert result.message is replacement
    # a handler returning nothing leaves the chain unchanged
    _, registry2 = make_registry()
    registry2.on("after_response", lambda event: None)
    result2 = await registry2.run("after_response", AfterResponseEvent(message=original))
    assert result2.message is original


async def test_before_tool_args_chaining_and_block():
    _, registry = make_registry()
    registry.on("before_tool", lambda event: BeforeToolResult(args={**event.args, "extra": 1}))
    registry.on("before_tool", lambda event: BeforeToolResult(block=ToolBlock(reason="denied", terminate=True)))
    registry.on("before_tool", lambda event: BeforeToolResult(args={"never": True}))  # not reached
    result = await registry.run(
        "before_tool", BeforeToolEvent(tool_call_id="t", tool_name="bash", args={"command": "ls"})
    )
    assert result.args == {"command": "ls", "extra": 1}
    assert result.block.reason == "denied"
    assert result.block.terminate is True


async def test_before_tool_handler_error_blocks():
    recorder, registry = make_registry()

    def failing(event):
        raise RuntimeError("kaboom")

    registry.on("before_tool", failing)
    result = await registry.run("before_tool", BeforeToolEvent(tool_call_id="t", tool_name="bash", args={}))
    assert result.block.reason == "kaboom"
    assert recorder.errors == [("kaboom", "before_tool")]


async def test_before_tool_noop_returns_empty_result():
    _, registry = make_registry()
    registry.on("before_tool", lambda event: None)
    result = await registry.run("before_tool", BeforeToolEvent(tool_call_id="t", tool_name="bash", args={"a": 1}))
    assert result.args is None
    assert result.block is None


async def test_after_tool_field_wise_aggregation():
    from karen_ai import TextContent, Usage

    _, registry = make_registry()
    registry.on(
        "after_tool",
        lambda event: AfterToolResult(content=[TextContent(text="rewritten")], is_error=False),
    )
    registry.on("after_tool", lambda event: AfterToolResult(terminate=True))
    event = AfterToolEvent(
        tool_call_id="t", tool_name="read", args={}, content=[TextContent(text="orig")], is_error=True
    )
    result = await registry.run("after_tool", event)
    assert result.content[0].text == "rewritten"
    assert result.is_error is False
    assert result.terminate is True
    assert "usage" not in result.model_fields_set

    # no effective overrides -> None
    _, registry2 = make_registry()
    registry2.on("after_tool", lambda event: None)
    assert await registry2.run("after_tool", event) is None


async def test_after_tool_handlers_see_chained_state():
    _, registry = make_registry()
    seen = []
    registry.on("after_tool", lambda event: AfterToolResult(is_error=False))
    registry.on("after_tool", lambda event: seen.append(event.is_error) or None)
    event = AfterToolEvent(tool_call_id="t", tool_name="x", args={}, content=[], is_error=True)
    await registry.run("after_tool", event)
    assert seen == [False]


async def test_before_compaction_first_structural():
    _, registry = make_registry()
    compaction = CompactResult(summary="s", tokens_before=1)
    registry.on("before_compaction", lambda event: None)
    registry.on("before_compaction", lambda event: BeforeCompactionResult(compaction=compaction))
    registry.on("before_compaction", lambda event: BeforeCompactionResult(decline=True))
    result = await registry.run(
        "before_compaction", BeforeCompactionEvent(reason="threshold", preparation=_compaction_preparation())
    )
    assert result.compaction is compaction

    _, registry2 = make_registry()
    registry2.on("before_compaction", lambda event: BeforeCompactionResult(decline=True))
    result2 = await registry2.run("before_compaction", BeforeCompactionEvent(reason="manual", preparation=_compaction_preparation()))
    assert result2.decline is True

    _, registry3 = make_registry()
    registry3.on("before_compaction", lambda event: None)
    assert await registry3.run(
        "before_compaction", BeforeCompactionEvent(reason="manual", preparation=_compaction_preparation())
    ) is None


async def test_before_compaction_decline_plus_compaction_is_an_error():
    recorder, registry = make_registry()
    compaction = CompactResult(summary="s", tokens_before=1)
    registry.on(
        "before_compaction", lambda event: BeforeCompactionResult(decline=True, compaction=compaction)
    )
    registry.on("before_compaction", lambda event: BeforeCompactionResult(decline=True))
    result = await registry.run(
        "before_compaction", BeforeCompactionEvent(reason="overflow", preparation=_compaction_preparation())
    )
    assert recorder.errors == [("before_compaction hook cannot return both decline and compaction", "before_compaction")]
    assert result.decline is True  # fell through to the second handler


async def test_before_navigation_summary_field():
    from karen_agent.compaction import BranchSummaryResult

    _, registry = make_registry()
    summary = BranchSummaryResult(summary="s", read_files=[], modified_files=[])
    registry.on("before_navigation", lambda event: BeforeNavigationResult(summary=summary))
    result = await registry.run(
        "before_navigation", BeforeNavigationEvent(target_id="t", preparation=_branch_preparation())
    )
    assert result.summary is summary

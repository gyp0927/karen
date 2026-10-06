"""Compaction core (pi's compaction/compaction.ts)."""

import math

import pytest
from karen_ai import SimpleStreamOptions, TextContent, ToolCall, Usage, UserMessage
from karen_ai.providers import faux_assistant_message, faux_model

from karen_agent.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    CompactGenerationOptions,
    CompactionPreparation,
    CompactionSettings,
    SummaryGenerationOptions,
    compact_with_request,
    complete_summary,
    create_file_ops,
    create_summary_request_options,
    estimate_context_tokens,
    estimate_tokens,
    find_cut_point,
    find_turn_start_index,
    generate_summary_with_request,
    get_last_assistant_usage,
    prepare_compaction,
    should_compact,
)
from karen_agent.messages import (
    BashExecutionMessage,
    create_branch_summary_message,
    create_compaction_summary_message,
)
from karen_agent.result import CompactionError, Err, Ok
from karen_agent.session.types import BranchSummaryEntry, CompactionEntry, MessageEntry


def _user(text, timestamp=1):
    return UserMessage(content=text, timestamp=timestamp)


def _entry(id, message):
    return MessageEntry(id=id, message=message, timestamp=1)


def _assistant_with_usage(text, total_tokens, stop_reason="stop"):
    message = faux_assistant_message(text, stop_reason=stop_reason)
    message.usage = Usage(input=1, output=1, total_tokens=total_tokens)
    return message


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def test_estimate_tokens_roles():
    assert estimate_tokens(_user("a" * 8)) == 2  # ceil(8/4)
    image_message = {"role": "user", "content": [{"type": "text", "text": "abcd"}, {"type": "image"}]}
    assert estimate_tokens(image_message) == math.ceil((4 + 4800) / 4)
    assistant = faux_assistant_message(
        [TextContent(text="1234"), ToolCall(id="t", name="read", arguments={"p": "x"})]
    )
    # text 4 + name 4 + args len('{"p":"x"}')=9 -> 17 -> ceil(17/4)=5
    assert estimate_tokens(assistant) == 5
    assert estimate_tokens(BashExecutionMessage(command="ls", output="1234", exit_code=0, cancelled=False, truncated=False, timestamp=0)) == math.ceil((2 + 4) / 4)
    assert estimate_tokens(create_compaction_summary_message("12345678", 1, 0)) == 2
    assert estimate_tokens({"role": "mystery"}) == 0


def test_estimate_context_tokens_without_usage():
    estimate = estimate_context_tokens([_user("a" * 8), _user("b" * 8)])
    assert estimate.tokens == 4
    assert estimate.usage_tokens == 0
    assert estimate.trailing_tokens == 4
    assert estimate.last_usage_index is None


def test_estimate_context_tokens_with_usage():
    messages = [
        _user("q"),
        _assistant_with_usage("a", 1000),
        _user("b" * 40),  # 10 trailing tokens
    ]
    estimate = estimate_context_tokens(messages)
    assert estimate.tokens == 1010
    assert estimate.usage_tokens == 1000
    assert estimate.trailing_tokens == 10
    assert estimate.last_usage_index == 1


def test_usage_from_aborted_or_zero_assistant_ignored():
    aborted = _assistant_with_usage("a", 1000, stop_reason="aborted")
    zero = faux_assistant_message("b")  # usage all zeros
    estimate = estimate_context_tokens([aborted, zero])
    assert estimate.last_usage_index is None


def test_get_last_assistant_usage():
    good = _assistant_with_usage("a", 500)
    entries = [_entry("e1", _user("q")), _entry("e2", good), _entry("e3", _user("later"))]
    usage = get_last_assistant_usage(entries)
    assert usage is not None and usage.total_tokens == 500
    assert get_last_assistant_usage([_entry("e", _user("q"))]) is None


def test_should_compact():
    settings = CompactionSettings()
    assert should_compact(200000 - 16384 + 1, 200000, settings) is True
    assert should_compact(200000 - 16384, 200000, settings) is False
    assert should_compact(999999, 200000, CompactionSettings(enabled=False)) is False


# ---------------------------------------------------------------------------
# Cut points
# ---------------------------------------------------------------------------


def _assistant_entry(id, text):
    return _entry(id, faux_assistant_message(text))


def test_find_cut_point_no_cut_points():
    from karen_ai import ToolResultMessage

    tool_result = ToolResultMessage(tool_call_id="t", tool_name="x", content=[TextContent(text="r")], is_error=False, timestamp=0)
    entries = [_entry("a", tool_result)]
    result = find_cut_point(entries, 0, 1, 100)
    assert result.first_kept_entry_index == 0
    assert result.turn_start_index == -1
    assert result.is_split_turn is False


def test_find_cut_point_keeps_budget_from_the_end():
    entries = [
        _entry("u1", _user("a" * 400)),  # 100 tokens
        _assistant_entry("a1", "b" * 400),  # 100
        _entry("u2", _user("c" * 400)),  # 100
        _assistant_entry("a2", "d" * 400),  # 100
    ]
    result = find_cut_point(entries, 0, 4, 150)
    # walking from the end: a2 (100) then u2 (200 >= 150) -> cut at u2
    assert result.first_kept_entry_index == 2
    assert result.is_split_turn is False
    assert result.turn_start_index == -1


def test_find_cut_point_splits_turn_on_assistant_cut():
    entries = [
        _entry("u1", _user("a" * 400)),
        _assistant_entry("a1", "b" * 400),
        _assistant_entry("a2", "c" * 400),
    ]
    result = find_cut_point(entries, 0, 3, 150)
    # a2 (100), a1 (200 >= 150) -> cut at a1 (assistant) -> split turn starting at u1
    assert result.first_kept_entry_index == 1
    assert result.turn_start_index == 0
    assert result.is_split_turn is True


def test_find_cut_point_walks_back_over_branch_summary():
    entries = [
        _entry("u1", _user("a" * 400)),
        _assistant_entry("a1", "b" * 400),
        BranchSummaryEntry(id="bs", summary="s", from_id=None, from_hook=False, timestamp=1),
        _entry("u2", _user("c" * 400)),
    ]
    result = find_cut_point(entries, 0, 4, 1)
    # budget exhausted at u2 (index 3), but the cut walks back over the
    # branch_summary entry; the branch summary becomes the turn start
    assert result.first_kept_entry_index == 2
    assert result.turn_start_index == 2
    assert result.is_split_turn is True


def test_find_cut_point_branch_summary_is_a_cut_point():
    entries = [
        _entry("u1", _user("a" * 400)),
        BranchSummaryEntry(id="bs", summary="s", from_id=None, from_hook=False, timestamp=1),
        _assistant_entry("a1", "b" * 400),
    ]
    result = find_cut_point(entries, 0, 3, 150)
    # a1 (100), u1 not reached... accumulated after a1=100 <150, then bs (not message, skip),
    # then u1 (200 >= 150) -> first cut point >= 0 is 0 (u1) -> cut at 0
    assert result.first_kept_entry_index == 0


def test_find_turn_start_index():
    entries = [_entry("u1", _user("x")), _assistant_entry("a1", "y"), _assistant_entry("a2", "z")]
    assert find_turn_start_index(entries, 2, 0) == 0
    entries_with_branch = [
        _entry("u1", _user("x")),
        BranchSummaryEntry(id="bs", summary="s", from_id=None, from_hook=False, timestamp=1),
        _assistant_entry("a1", "y"),
    ]
    assert find_turn_start_index(entries_with_branch, 2, 0) == 1


# ---------------------------------------------------------------------------
# prepare_compaction
# ---------------------------------------------------------------------------


def test_prepare_compaction_empty_or_trailing_compaction():
    assert isinstance(prepare_compaction([], DEFAULT_COMPACTION_SETTINGS), Ok)
    assert prepare_compaction([], DEFAULT_COMPACTION_SETTINGS).value is None
    trailing = [CompactionEntry(id="c", summary="s", tokens_before=1, from_hook=False, timestamp=1)]
    assert prepare_compaction(trailing, DEFAULT_COMPACTION_SETTINGS).value is None


def test_prepare_compaction_simple():
    settings = CompactionSettings(keep_recent_tokens=100)
    entries = [
        _entry("u1", _user("a" * 400)),  # 100 tokens
        _assistant_entry("a1", "b" * 400),  # 100
        _entry("u2", _user("c" * 400)),  # 100
    ]
    result = prepare_compaction(entries, settings)
    assert isinstance(result, Ok)
    preparation = result.value
    assert preparation.is_split_turn is False
    # keep 150 -> cut at u2 (index 2): u1+a1 summarized, u2 retained
    assert [getattr(m, "content", None) for m in preparation.retained_tail] == ["c" * 400]
    assert len(preparation.messages_to_summarize) == 2
    assert preparation.previous_summary is None
    assert preparation.tokens_before > 0


def test_prepare_compaction_with_previous_compaction():
    settings = CompactionSettings(keep_recent_tokens=100)
    prev = CompactionEntry(
        id="c1",
        summary="previous summary",
        tokens_before=50,
        retained_tail=[_user("retained question")],
        details={"readFiles": ["old.py"], "modifiedFiles": ["mod.py"]},
        from_hook=False,
        timestamp=1,
    )
    edit_call = faux_assistant_message([ToolCall(id="t", name="edit", arguments={"path": "new.py"})])
    entries = [
        prev,
        _entry("u2", _user("a" * 400)),  # 100
        _entry("a2", edit_call),
        _entry("u3", _user("b" * 400)),  # 100
    ]
    result = prepare_compaction(entries, settings)
    preparation = result.value
    assert preparation.previous_summary == "previous summary"
    # virtual retained entries are part of the compactable range
    summarized_contents = [getattr(m, "content", None) for m in preparation.messages_to_summarize]
    assert "retained question" in summarized_contents
    # file ops seeded from previous details + new tool calls
    assert "old.py" in preparation.file_ops.read
    assert "mod.py" in preparation.file_ops.edited
    assert "new.py" in preparation.file_ops.edited


def test_prepare_compaction_split_turn():
    settings = CompactionSettings(keep_recent_tokens=150)
    entries = [
        _entry("u1", _user("a" * 400)),  # 100  <- turn start
        _assistant_entry("a1", "b" * 400),  # 100
        _assistant_entry("a2", "c" * 400),  # 100
    ]
    preparation = prepare_compaction(entries, settings).value
    assert preparation.is_split_turn is True
    # cut at a1; turn prefix = u1; retained = a1, a2; history (before turn) empty
    assert preparation.messages_to_summarize == []
    assert [getattr(m, "content", None) for m in preparation.turn_prefix_messages] == ["a" * 400]
    assert len(preparation.retained_tail) == 2


# ---------------------------------------------------------------------------
# Summary generation through a scripted request boundary
# ---------------------------------------------------------------------------


class ScriptedRequest:
    """Records (ai_context, options) and responds with a scripted assistant message."""

    def __init__(self, *responses):
        self.calls = []
        self._responses = list(responses)

    async def __call__(self, ai_context, options):
        self.calls.append((ai_context, options))
        if not self._responses:
            return faux_assistant_message("summary text")
        return self._responses.pop(0)


async def test_generate_summary_prompt_assembly():
    request = ScriptedRequest()
    model = faux_model(max_tokens=8192)
    result = await _generate(model, request, custom_instructions=None, previous_summary=None)
    assert isinstance(result, Ok)
    assert result.value.text == "summary text"
    ai_context, options = request.calls[0]
    assert ai_context.system_prompt.startswith("You are a context summarization assistant")
    prompt = ai_context.messages[0].content[0].text
    assert prompt.startswith("<conversation>\n[User]: hello\n</conversation>\n\n")
    assert "## Goal" in prompt
    assert "<previous-summary>" not in prompt
    # max_tokens = min(floor(0.8 * 16384), 8192) = 8192
    assert options.max_tokens == 8192
    # summary requests pin cache_retention=none and a fresh session id
    assert options.cache_retention == "none"
    assert options.session_id


async def _generate(model, request, custom_instructions, previous_summary):
    return await generate_summary_with_request(
        [_user("hello")],
        SummaryGenerationOptions(
            model=model,
            reserve_tokens=16384,
            custom_instructions=custom_instructions,
            previous_summary=previous_summary,
        ),
        request,
    )


async def test_generate_summary_update_prompt_and_custom_instructions():
    request = ScriptedRequest()
    model = faux_model()
    result = await _generate(model, request, "focus on tests", "old summary")
    assert isinstance(result, Ok)
    ai_context, _ = request.calls[0]
    prompt = ai_context.messages[0].content[0].text
    assert "<previous-summary>\nold summary\n</previous-summary>\n\n" in prompt
    assert "NEW conversation messages" in prompt  # update prompt, not fresh prompt
    assert prompt.endswith("Additional focus: focus on tests")


async def test_generate_summary_reasoning_and_max_tokens():
    request = ScriptedRequest()
    model = faux_model(reasoning=True, max_tokens=0)  # uncapped -> floor(0.8*reserve)
    await _generate_options(model, request, thinking_level="high", reserve=1000)
    _, options = request.calls[0]
    assert options.max_tokens == 800
    assert options.reasoning == "high"

    request2 = ScriptedRequest()
    await _generate_options(model, request2, thinking_level="off", reserve=1000)
    _, options2 = request2.calls[0]
    assert options2.reasoning is None


async def _generate_options(model, request, thinking_level, reserve):
    return await generate_summary_with_request(
        [_user("hello")],
        SummaryGenerationOptions(model=model, reserve_tokens=reserve, thinking_level=thinking_level),
        request,
    )


async def test_generate_summary_error_mapping():
    aborted = faux_assistant_message("", stop_reason="aborted")
    aborted.error_message = ""
    result = await _generate(faux_model(), ScriptedRequest(aborted), None, None)
    assert isinstance(result, Err)
    assert result.error.code == "aborted"
    assert str(result.error) == "Summarization aborted"

    failed = faux_assistant_message("", stop_reason="error")
    failed.error_message = "boom"
    result2 = await _generate(faux_model(), ScriptedRequest(failed), None, None)
    assert result2.error.code == "summarization_failed"
    assert str(result2.error) == "Summarization failed: boom"


async def test_compact_with_request_appends_file_tags():
    file_ops = create_file_ops()
    file_ops.read.add("a.py")
    file_ops.edited.add("b.py")
    preparation = CompactionPreparation(
        messages_to_summarize=[_user("hello")],
        turn_prefix_messages=[],
        retained_tail=[_user("kept")],
        is_split_turn=False,
        tokens_before=123,
        file_ops=file_ops,
        settings=CompactionSettings(),
    )
    request = ScriptedRequest()
    result = await compact_with_request(
        preparation, CompactGenerationOptions(model=faux_model()), request
    )
    assert isinstance(result, Ok)
    assert result.value.summary == (
        "summary text\n\n<read-files>\na.py\n</read-files>\n\n<modified-files>\nb.py\n</modified-files>"
    )
    assert result.value.tokens_before == 123
    assert result.value.retained_tail == preparation.retained_tail
    assert result.value.details == {"readFiles": ["a.py"], "modifiedFiles": ["b.py"]}
    assert result.value.usage is not None


async def test_compact_with_request_split_turn_double_summary():
    preparation = CompactionPreparation(
        messages_to_summarize=[_user("old stuff")],
        turn_prefix_messages=[_user("big request"), faux_assistant_message("early work")],
        retained_tail=[faux_assistant_message("recent work")],
        is_split_turn=True,
        tokens_before=999,
        previous_summary=None,
        file_ops=create_file_ops(),
        settings=CompactionSettings(),
    )
    request = ScriptedRequest(
        faux_assistant_message("history summary"),
        faux_assistant_message("prefix summary"),
    )
    result = await compact_with_request(
        preparation, CompactGenerationOptions(model=faux_model()), request
    )
    assert isinstance(result, Ok)
    assert result.value.summary == "history summary\n\n---\n\n**Turn Context (split turn):**\n\nprefix summary"
    assert len(request.calls) == 2
    # second call used the turn-prefix prompt
    second_prompt = request.calls[1][0].messages[0].content[0].text
    assert "PREFIX of a turn" in second_prompt
    # turn prefix max_tokens = floor(0.5 * reserve)
    assert request.calls[1][1].max_tokens == math.floor(0.5 * 16384)


async def test_compact_with_request_split_turn_without_history():
    preparation = CompactionPreparation(
        messages_to_summarize=[],
        turn_prefix_messages=[_user("big request")],
        retained_tail=[],
        is_split_turn=True,
        tokens_before=1,
        file_ops=create_file_ops(),
        settings=CompactionSettings(),
    )
    request = ScriptedRequest(faux_assistant_message("prefix summary"))
    result = await compact_with_request(
        preparation, CompactGenerationOptions(model=faux_model()), request
    )
    assert result.value.summary.startswith("No prior history.\n\n---\n\n**Turn Context (split turn):**")
    assert len(request.calls) == 1  # no history request when nothing to summarize


async def test_compact_with_request_propagates_errors():
    preparation = CompactionPreparation(
        messages_to_summarize=[_user("x")],
        turn_prefix_messages=[],
        retained_tail=[],
        is_split_turn=False,
        tokens_before=1,
        file_ops=create_file_ops(),
        settings=CompactionSettings(),
    )
    failed = faux_assistant_message("", stop_reason="error")
    failed.error_message = "nope"
    result = await compact_with_request(
        preparation, CompactGenerationOptions(model=faux_model()), ScriptedRequest(failed)
    )
    assert isinstance(result, Err)
    assert result.error.code == "summarization_failed"


def test_create_summary_request_options_keeps_explicit_session_id():
    options = SimpleStreamOptions(max_tokens=10, session_id="fixed")
    merged = create_summary_request_options(options)
    assert merged.session_id == "fixed"
    merged2 = create_summary_request_options(SimpleStreamOptions(max_tokens=10))
    assert merged2.session_id and merged2.session_id != "fixed"
    assert merged2.cache_retention == "none"


async def test_complete_summary_default_boundary():
    from karen_ai import Context

    class FakeModels:
        def __init__(self):
            self.calls = []

        async def complete_simple(self, model, context, options=None):
            self.calls.append((model, context, options))
            return faux_assistant_message("done")

    models = FakeModels()
    model = faux_model()
    message = await complete_summary(
        models, model, Context(messages=[_user("x")]), SimpleStreamOptions(max_tokens=5)
    )
    assert message.content[0].text == "done"
    _, _, options = models.calls[0]
    assert options.max_tokens == 5
    assert options.cache_retention == "none"
    assert options.session_id


# ---------------------------------------------------------------------------
# assistant-call retry (pi's completeSimpleWithRetries)
# ---------------------------------------------------------------------------


class FlakyModels:
    """Fails the first `failures` calls with a transient error, then succeeds."""

    def __init__(self, failures=1, error="Error 503 Service Unavailable", text="summary"):
        self.failures = failures
        self.error = error
        self.text = text
        self.calls = 0

    async def complete_simple(self, model, context, options=None):
        self.calls += 1
        if self.calls <= self.failures:
            failed = faux_assistant_message("", stop_reason="error")
            failed.error_message = self.error
            return failed
        return faux_assistant_message(self.text)


def _one_message_preparation():
    return CompactionPreparation(
        messages_to_summarize=[_user("x")],
        turn_prefix_messages=[],
        retained_tail=[],
        is_split_turn=False,
        tokens_before=1,
        file_ops=create_file_ops(),
        settings=CompactionSettings(),
    )


def _retry_policy(max_retries=3):
    from karen_ai import RetryPolicy

    return RetryPolicy(enabled=True, max_retries=max_retries, base_delay_ms=1, max_agent_delay_ms=1)


def retry_callbacks(events):
    """Collect retry callback invocations into `events` as tagged tuples."""
    from karen_ai import RetryCallbacks

    return RetryCallbacks(
        on_retry_scheduled=lambda attempt, max_attempts, delay_ms, error: events.append(
            ("scheduled", attempt, max_attempts, delay_ms, error)
        ),
        on_retry_attempt_start=lambda: events.append(("attempt_start",)),
        on_retry_finished=lambda success, attempt, final_error=None: events.append(
            ("finished", success, attempt, final_error)
        ),
    )


async def test_compact_retries_transient_summary_errors():
    from karen_agent.compaction import compact

    models = FlakyModels(failures=1)
    events = []
    result = await compact(
        _one_message_preparation(),
        models,
        faux_model(),
        retry=_retry_policy(),
        callbacks=retry_callbacks(events),
    )
    assert isinstance(result, Ok)
    assert "summary" in result.value.summary
    assert models.calls == 2
    assert [event[0] for event in events] == ["scheduled", "attempt_start", "finished"]
    assert events[0] == ("scheduled", 1, 3, 1, "Error 503 Service Unavailable")
    assert events[2] == ("finished", True, 1, None)


async def test_compact_without_a_policy_fails_on_the_first_error():
    from karen_agent.compaction import compact

    models = FlakyModels(failures=1)
    result = await compact(_one_message_preparation(), models, faux_model())
    assert isinstance(result, Err)
    assert result.error.code == "summarization_failed"
    assert models.calls == 1


async def test_compact_gives_up_when_the_retry_budget_is_exhausted():
    from karen_agent.compaction import compact

    models = FlakyModels(failures=99)
    events = []
    result = await compact(
        _one_message_preparation(),
        models,
        faux_model(),
        retry=_retry_policy(max_retries=2),
        callbacks=retry_callbacks(events),
    )
    assert isinstance(result, Err)
    assert models.calls == 3  # initial call + 2 retries
    assert events[-1] == ("finished", False, 2, "Error 503 Service Unavailable")


async def test_compact_does_not_retry_account_limit_errors():
    from karen_agent.compaction import compact

    models = FlakyModels(failures=99, error="insufficient_quota: billing hard limit reached")
    result = await compact(
        _one_message_preparation(), models, faux_model(), retry=_retry_policy()
    )
    assert isinstance(result, Err)
    assert models.calls == 1

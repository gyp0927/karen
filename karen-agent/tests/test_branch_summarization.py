"""Branch summarization (pi's compaction/branch-summarization.ts)."""

import pytest
from karen_ai import TextContent, ToolCall, Usage, UserMessage
from karen_ai.providers import faux_assistant_message, faux_model

from karen_agent.compaction import (
    BranchPreparation,
    create_file_ops,
    GenerateBranchSummaryOptions,
    PreparedBranchSummaryOptions,
    collect_entries_for_branch_summary,
    generate_branch_summary,
    generate_branch_summary_with_request,
    prepare_branch_entries,
)
from karen_agent.result import Err, Ok
from karen_agent.session.types import (
    BranchScan,
    BranchSummaryEntry,
    CompactionEntry,
    MessageEntry,
)


def _user(text, timestamp=1):
    return UserMessage(content=text, timestamp=timestamp)


def _entry(id, message, parent_id=None):
    return MessageEntry(id=id, parent_id=parent_id, message=message, timestamp=1)


# ---------------------------------------------------------------------------
# collect_entries_for_branch_summary (against a fake branch/session)
# ---------------------------------------------------------------------------


class FakeBranch:
    def __init__(self, entries_by_id):
        self._entries = entries_by_id

    async def find_entries(self, query=None):
        # walk parents from query.start, newest first
        result = []
        current = query.start if query else None
        while current is not None:
            entry = self._entries.get(current)
            if entry is None:
                break
            result.append(entry)
            current = entry.parent_id
        return result


class FakeSession:
    def __init__(self, entries_by_id):
        self._entries = entries_by_id

    async def get_entry(self, id):
        return self._entries.get(id)


def _chain(*ids):
    """Entries id[0] <- id[1] <- ... (parent chain)."""
    entries = {}
    for i, id in enumerate(ids):
        entries[id] = _entry(id, _user(f"msg {id}"), parent_id=ids[i - 1] if i else None)
    return entries


async def test_collect_entries_no_old_tip():
    result = await collect_entries_for_branch_summary(FakeBranch({}), FakeSession({}), None, "t")
    assert result.entries == []
    assert result.common_ancestor_id is None


async def test_collect_entries_walks_to_common_ancestor():
    # main: a -> b -> c (old tip); target branch: a -> d
    entries = _chain("a", "b", "c")
    entries["d"] = _entry("d", _user("msg d"), parent_id="a")
    result = await collect_entries_for_branch_summary(FakeBranch(entries), FakeSession(entries), "c", "d")
    assert result.common_ancestor_id == "a"
    assert [e.id for e in result.entries] == ["b", "c"]  # chronological order


async def test_collect_entries_corrupt_session_raises():
    entries = _chain("a", "b")
    entries["x"] = _entry("x", _user("msg x"))  # unrelated target, no common ancestor
    session_entries = {"b": entries["b"], "x": entries["x"]}  # "a" missing -> corrupt
    with pytest.raises(Exception, match="Corrupt session: entry a not found"):
        await collect_entries_for_branch_summary(FakeBranch(entries), FakeSession(session_entries), "b", "x")


# ---------------------------------------------------------------------------
# prepare_branch_entries
# ---------------------------------------------------------------------------


def test_prepare_branch_entries_skips_tool_results_and_collects_ops():
    from karen_ai import ToolResultMessage

    read_call = faux_assistant_message([ToolCall(id="t", name="read", arguments={"path": "x.py"})])
    tool_result = ToolResultMessage(tool_call_id="t", tool_name="read", content=[TextContent(text="data")], is_error=False, timestamp=0)
    entries = [_entry("1", _user("q")), _entry("2", read_call), _entry("3", tool_result)]
    preparation = prepare_branch_entries(entries)
    assert len(preparation.messages) == 2  # tool result excluded
    assert "x.py" in preparation.file_ops.read
    assert preparation.total_tokens > 0


def test_prepare_branch_entries_seeds_from_prior_branch_summary_details():
    prior = BranchSummaryEntry(
        id="bs",
        summary="s",
        from_id=None,
        details={"readFiles": ["old.py"], "modifiedFiles": ["mod.py"]},
        from_hook=False,
        timestamp=1,
    )
    preparation = prepare_branch_entries([prior, _entry("1", _user("q"))])
    assert "old.py" in preparation.file_ops.read
    assert "mod.py" in preparation.file_ops.edited


def test_prepare_branch_entries_token_budget():
    entries = [_entry(str(i), _user("x" * 400)) for i in range(10)]  # 100 tokens each
    preparation = prepare_branch_entries(entries, token_budget=250)
    # fits 2 full messages (200); the third would exceed -> stop
    assert len(preparation.messages) == 2
    assert preparation.total_tokens == 200
    # newest kept
    assert preparation.messages[0].content == entries[8].message.content


def test_prepare_branch_entries_budget_exception_for_summaries():
    compaction = CompactionEntry(id="c", summary="s" * 100, tokens_before=1, from_hook=False, timestamp=1)
    entries = [_entry("1", _user("x" * 400)), compaction]
    # budget 20: the compaction summary (25 tokens) exceeds the budget, but summary
    # entries are still included when the accumulated total is below 90% of budget
    preparation = prepare_branch_entries(entries, token_budget=20)
    assert len(preparation.messages) == 1  # the compaction summary
    assert preparation.messages[0].role == "compactionSummary"


# ---------------------------------------------------------------------------
# generate_branch_summary_with_request
# ---------------------------------------------------------------------------


class ScriptedRequest:
    def __init__(self, *responses):
        self.calls = []
        self._responses = list(responses)

    async def __call__(self, ai_context, options):
        self.calls.append((ai_context, options))
        if not self._responses:
            return faux_assistant_message("branch summary text")
        return self._responses.pop(0)


def _preparation(*messages):
    file_ops_messages = list(messages)
    from karen_agent.compaction import extract_file_ops_from_message

    ops = create_file_ops()
    for m in file_ops_messages:
        extract_file_ops_from_message(m, ops)
    tokens = sum(1 for _ in messages)
    return BranchPreparation(messages=list(messages), file_ops=ops, total_tokens=tokens)


async def test_generate_branch_summary_empty():
    result = await generate_branch_summary_with_request(
        BranchPreparation(messages=[], file_ops=create_file_ops(), total_tokens=0),
        PreparedBranchSummaryOptions(),
        ScriptedRequest(),
    )
    assert isinstance(result, Ok)
    assert result.value.summary == "No content to summarize"
    assert result.value.usage is None
    assert result.value.read_files == []


async def test_generate_branch_summary_prompt_and_result():
    edit_call = faux_assistant_message([ToolCall(id="t", name="edit", arguments={"path": "x.py"})])
    preparation = _preparation(_user("explore"), edit_call)
    request = ScriptedRequest()
    result = await generate_branch_summary_with_request(preparation, PreparedBranchSummaryOptions(), request)
    assert isinstance(result, Ok)
    assert result.value.summary.startswith(
        "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n"
    )
    assert "branch summary text" in result.value.summary
    assert result.value.summary.endswith("<modified-files>\nx.py\n</modified-files>")
    assert result.value.usage is not None
    ai_context, options = request.calls[0]
    assert options.max_tokens == 2048
    prompt = ai_context.messages[0].content[0].text
    assert prompt.startswith("<conversation>\n")
    assert "## Goal" in prompt


async def test_generate_branch_summary_custom_and_replace_instructions():
    preparation = _preparation(_user("x"))
    request = ScriptedRequest()
    await generate_branch_summary_with_request(
        preparation, PreparedBranchSummaryOptions(custom_instructions="focus Y"), request
    )
    prompt = request.calls[0][0].messages[0].content[0].text
    assert prompt.endswith("Additional focus: focus Y")

    request2 = ScriptedRequest()
    await generate_branch_summary_with_request(
        preparation,
        PreparedBranchSummaryOptions(custom_instructions="only this", replace_instructions=True),
        request2,
    )
    prompt2 = request2.calls[0][0].messages[0].content[0].text
    assert prompt2.endswith("only this")
    assert "## Goal" not in prompt2


async def test_generate_branch_summary_error_mapping():
    aborted = faux_assistant_message("", stop_reason="aborted")
    aborted.error_message = None
    result = await generate_branch_summary_with_request(
        _preparation(_user("x")), PreparedBranchSummaryOptions(), ScriptedRequest(aborted)
    )
    assert isinstance(result, Err)
    assert result.error.code == "aborted"
    assert str(result.error) == "Branch summary aborted"

    failed = faux_assistant_message("", stop_reason="error")
    failed.error_message = "bad"
    result2 = await generate_branch_summary_with_request(
        _preparation(_user("x")), PreparedBranchSummaryOptions(), ScriptedRequest(failed)
    )
    assert result2.error.code == "summarization_failed"
    assert str(result2.error) == "Branch summary failed: bad"


async def test_generate_branch_summary_end_to_end_with_fake_models():
    class FakeModels:
        def __init__(self):
            self.calls = []

        async def complete_simple(self, model, context, options=None):
            self.calls.append((model, context, options))
            return faux_assistant_message("E2E summary")

    models = FakeModels()
    model = faux_model(context_window=100000)
    entries = [_entry("1", _user("hello"))]
    result = await generate_branch_summary(entries, GenerateBranchSummaryOptions(models=models, model=model))
    assert isinstance(result, Ok)
    assert "E2E summary" in result.value.summary
    _, _, options = models.calls[0]
    assert options.cache_retention == "none"
    assert options.session_id


async def test_generate_branch_summary_retries_transient_errors():
    from karen_ai import RetryCallbacks, RetryPolicy

    class FlakyModels:
        def __init__(self):
            self.calls = 0

        async def complete_simple(self, model, context, options=None):
            self.calls += 1
            if self.calls == 1:
                failed = faux_assistant_message("", stop_reason="error")
                failed.error_message = "overloaded_error"
                return failed
            return faux_assistant_message("retried summary")

    events = []
    models = FlakyModels()
    result = await generate_branch_summary(
        [_entry("1", _user("hello"))],
        GenerateBranchSummaryOptions(
            models=models,
            model=faux_model(context_window=100000),
            retry=RetryPolicy(enabled=True, max_retries=2, base_delay_ms=1, max_agent_delay_ms=1),
            callbacks=RetryCallbacks(
                on_retry_scheduled=lambda *args: events.append(("scheduled",) + args),
                on_retry_finished=lambda *args: events.append(("finished",) + args),
            ),
        ),
    )
    assert isinstance(result, Ok)
    assert "retried summary" in result.value.summary
    assert models.calls == 2
    assert events[0] == ("scheduled", 1, 2, 1, "overloaded_error")
    assert events[-1] == ("finished", True, 1)


async def test_generate_branch_summary_without_a_policy_reports_the_first_failure():
    class FailingModels:
        def __init__(self):
            self.calls = 0

        async def complete_simple(self, model, context, options=None):
            self.calls += 1
            failed = faux_assistant_message("", stop_reason="error")
            failed.error_message = "overloaded_error"
            return failed

    models = FailingModels()
    result = await generate_branch_summary(
        [_entry("1", _user("hello"))],
        GenerateBranchSummaryOptions(models=models, model=faux_model(context_window=100000)),
    )
    assert isinstance(result, Err)
    assert models.calls == 1

"""Offline end-to-end tests for AgentSession, driven by the scripted faux
provider through the real event protocol — no network or credentials."""

import asyncio
import time

import pytest

from karen_ai import Usage, create_models
from karen_ai.providers import faux_assistant_message, faux_model, register_faux_provider
from karen_agent.compaction import CompactionSettings
from karen_agent.session import BranchScan
from karen_agent.session.jsonl import JSONL_FORMAT_VERSION
from karen_coding_agent import AgentSession


def _models_with_faux(responses, **model_kwargs):
    models = create_models()
    registration = register_faux_provider(
        models=[faux_model(**model_kwargs)] if model_kwargs else None,
        responses=responses,
    )
    models.set_provider(registration.provider)
    return models, registration


async def _open(tmp_path, models, registration, **kwargs):
    kwargs.setdefault("sessions_root", str(tmp_path / "sessions"))
    kwargs.setdefault("fresh", True)
    session = AgentSession(
        cwd=str(tmp_path), models=models, model=registration.get_model(), **kwargs
    )
    await session.open()
    return session


async def _entries(session):
    branch = await session.session.branch("main")
    return await branch.find_entries(BranchScan(order="oldestFirst"))


async def _messages(session):
    return [entry.message for entry in await _entries(session) if entry.type == "message"]


async def _wait_for_event(events, event_type, timeout=5.0):
    """Wait until a session event of `event_type` has been emitted."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(event.get("type") == event_type for event in events):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"{event_type} was never emitted")


# ---------------------------------------------------------------------------
# persistence + resume
# ---------------------------------------------------------------------------


async def test_session_header_matches_the_disk_line(tmp_path):
    models, registration = _models_with_faux([])
    session = await _open(tmp_path, models, registration)

    header = session.session_header()
    assert header["kind"] == "header"
    assert header["v"] == JSONL_FORMAT_VERSION
    assert header["id"] == session.session.metadata.id
    session_files = list((tmp_path / "sessions").rglob("*.jsonl"))
    assert len(session_files) == 1
    import json

    first_line = json.loads(session_files[0].read_text(encoding="utf-8").splitlines()[0])
    assert header == first_line
    await session.close()


async def test_prompt_persists_every_message(tmp_path):
    models, registration = _models_with_faux([faux_assistant_message("hi there")])
    session = await _open(tmp_path, models, registration)

    await session.prompt("hello")

    messages = await _messages(session)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].content[0].text == "hi there"
    assert registration.get_pending_response_count() == 0
    await session.close()


async def test_system_prompt_sections_ride_the_system_message(tmp_path):
    models, registration = _models_with_faux([faux_assistant_message("ok")])
    sections = {"preamble": "You are karen.", "cwd": "<cwd>\n/tmp\n</cwd>"}
    session = await _open(tmp_path, models, registration, system_prompt_sections=sections)

    system_message = session.agent.state.messages[0]
    assert system_message.role == "system"
    assert system_message.content == ""  # sections carry the prompt, like pi
    assert system_message.sections == sections
    assert system_message.tools_added  # tools still ride the system message
    # the rendered text matches what a request would send
    assert session.system_prompt_text == "You are karen.\n\n<cwd>\n/tmp\n</cwd>"

    await session.prompt("hello")
    await session.close()


async def test_resume_rebuilds_context(tmp_path):
    models, registration = _models_with_faux([faux_assistant_message("hi there")])
    session = await _open(tmp_path, models, registration)
    await session.prompt("hello")
    await session.close()

    reopened = await _open(tmp_path, models, registration, fresh=False)
    roles = [getattr(m, "role", None) for m in reopened.agent.state.messages]
    assert roles == ["system", "user", "assistant"]
    assert reopened.agent.state.messages[1].content[0].text == "hello"
    await reopened.close()


# ---------------------------------------------------------------------------
# threshold auto-compaction
# ---------------------------------------------------------------------------


def _state_text(session) -> str:
    parts = []
    for message in session.agent.state.messages:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(getattr(block, "text", "") for block in content)
        # compactionSummary/branchSummary harness messages carry their text in
        # a `summary` field rather than `content`
        summary = getattr(message, "summary", None)
        if isinstance(summary, str):
            parts.append(summary)
    return "\n".join(parts)


async def test_threshold_compaction_after_run(tmp_path):
    # window 200 - reserve 10 = 190 tokens: the first exchange stays under, the
    # second pushes the context over; keep_recent keeps the oversized second
    # turn whole, so the summary replaces exactly the first exchange.
    first_reply = "a" * 400  # ~100 tokens
    second_reply = "b" * 1000  # ~250 tokens
    responses = [
        faux_assistant_message(first_reply),
        faux_assistant_message(second_reply),
        faux_assistant_message("summary of the conversation"),
        faux_assistant_message("turn-prefix summary"),  # only if the cut splits a turn
    ]
    models, registration = _models_with_faux(responses, context_window=200)
    events = []
    session = await _open(
        tmp_path,
        models,
        registration,
        compaction_settings=CompactionSettings(keep_recent_tokens=50, reserve_tokens=10),
        listener=events.append,
    )

    await session.prompt("hello")
    assert not [e for e in events if e.get("type") == "compaction_start"]  # still under the threshold

    await session.prompt("again")

    entries = await _entries(session)
    assert any(entry.type == "compaction" for entry in entries)
    started = [e for e in events if e.get("type") == "compaction_start"]
    assert [e["reason"] for e in started] == ["threshold"]
    state_text = _state_text(session)
    assert "summary of the conversation" in state_text
    assert first_reply not in state_text  # compacted away
    assert second_reply in state_text  # retained tail
    await session.close()


# ---------------------------------------------------------------------------
# overflow recovery
# ---------------------------------------------------------------------------


async def test_overflow_omits_attempt_compacts_and_retries(tmp_path):
    responses = [
        faux_assistant_message("first answer"),  # seed history so compaction has a cut point
        faux_assistant_message([], stop_reason="error", error_message="400 prompt is too long: 999999 tokens"),
        faux_assistant_message("summary of prior work"),
        faux_assistant_message("recovered answer"),
    ]
    models, registration = _models_with_faux(responses)
    events = []
    session = await _open(
        tmp_path,
        models,
        registration,
        compaction_settings=CompactionSettings(keep_recent_tokens=10),
        listener=events.append,
    )
    await session.prompt("first")

    await session.prompt("second")

    event_types = [e.get("type") for e in events]
    assert "overflow_retry" in event_types
    assert [e["reason"] for e in events if e.get("type") == "compaction_start"] == ["overflow"]
    messages = await _messages(session)
    # the failed attempt is omitted from the reachable transcript; the retry is there
    assert all(getattr(m, "stop_reason", None) != "error" for m in messages)
    assert messages[-1].content[0].text == "recovered answer"
    assert registration.get_pending_response_count() == 0
    await session.close()


async def test_overflow_gives_up_after_one_retry(tmp_path):
    responses = [
        faux_assistant_message("first answer"),
        faux_assistant_message([], stop_reason="error", error_message="prompt is too long"),
        faux_assistant_message("summary of prior work"),
        faux_assistant_message([], stop_reason="error", error_message="prompt is too long"),
    ]
    models, registration = _models_with_faux(responses)
    events = []
    session = await _open(
        tmp_path,
        models,
        registration,
        compaction_settings=CompactionSettings(keep_recent_tokens=10),
        listener=events.append,
    )
    await session.prompt("first")

    await session.prompt("second")

    event_types = [e.get("type") for e in events]
    assert "overflow_give_up" in event_types
    # exactly one recovery compaction, and the second failure stays in the transcript
    assert [e["reason"] for e in events if e.get("type") == "compaction_start"] == ["overflow"]
    messages = await _messages(session)
    assert getattr(messages[-1], "stop_reason", None) == "error"
    assert registration.get_pending_response_count() == 0
    await session.close()


async def test_silent_overflow_compacts_without_retry(tmp_path):
    def silent_overflow(context, options, state, model):
        message = faux_assistant_message("done")
        message.usage = Usage(input=1500, output=5, total_tokens=1505)
        return message

    responses = [
        faux_assistant_message("first answer"),
        silent_overflow,
        faux_assistant_message("summary of prior work"),
    ]
    models, registration = _models_with_faux(responses, context_window=1000)
    events = []
    session = await _open(
        tmp_path,
        models,
        registration,
        compaction_settings=CompactionSettings(keep_recent_tokens=10, reserve_tokens=10),
        listener=events.append,
    )
    await session.prompt("first")

    await session.prompt("second")

    event_types = [e.get("type") for e in events]
    assert "overflow_retry" not in event_types  # a completed response is never retried
    assert [e["reason"] for e in events if e.get("type") == "compaction_start"] == ["overflow"]
    messages = await _messages(session)
    # the completed response is preserved
    assert any(getattr(c, "text", "") == "done" for m in messages for c in getattr(m, "content", []))
    assert registration.get_pending_response_count() == 0
    await session.close()


# ---------------------------------------------------------------------------
# _overflow_action decision table (no session needed)
# ---------------------------------------------------------------------------


@pytest.fixture
def session(tmp_path):
    models, registration = _models_with_faux([])
    return AgentSession(
        cwd=str(tmp_path),
        models=models,
        model=registration.get_model(),
        sessions_root=str(tmp_path / "sessions"),
        fresh=True,
    )


def test_overflow_action_ignores_aborted(session):
    message = faux_assistant_message([], stop_reason="aborted", error_message="prompt is too long")
    assert session._overflow_action(message) is None


def test_overflow_action_ignores_other_models(session):
    message = faux_assistant_message([], stop_reason="error", error_message="prompt is too long")
    message.model = "some-other-model"
    assert session._overflow_action(message) is None


def test_overflow_action_ignores_non_overflow_errors(session):
    message = faux_assistant_message([], stop_reason="error", error_message="500 internal server error")
    assert session._overflow_action(message) is None


def test_overflow_action_ignores_user_messages(session):
    from karen_ai import UserMessage

    assert session._overflow_action(UserMessage(content="hi", timestamp=1)) is None


# ---------------------------------------------------------------------------
# session navigation (M7)
# ---------------------------------------------------------------------------


async def _two_turn_session(tmp_path):
    models, registration = _models_with_faux(
        [faux_assistant_message("first reply"), faux_assistant_message("second reply")]
    )
    session = await _open(tmp_path, models, registration)
    await session.prompt("first question")
    await session.prompt("second question")
    return session, registration


def _texts(messages):
    return [m.content[0].text for m in messages if getattr(m, "content", None)]


async def test_navigate_tree_moves_the_tip_without_dropping_entries(tmp_path):
    session, _ = await _two_turn_session(tmp_path)
    entries = await session.entries()
    first_reply = entries[1]

    result = await session.navigate_tree(first_reply.id)

    assert result == {"cancelled": False, "editorText": None, "summaryEntryId": None}
    assert await session.branch_tip_id() == first_reply.id
    # context is rebuilt from the new ancestry; the file keeps every entry
    assert _texts(session.agent.state.messages[1:]) == ["first question", "first reply"]
    assert [entry.id for entry in await session.entries()] == [entry.id for entry in entries]
    await session.close()


async def test_navigate_tree_on_a_user_message_returns_its_text(tmp_path):
    session, _ = await _two_turn_session(tmp_path)
    second_question = (await session.entries())[2]

    result = await session.navigate_tree(second_question.id)

    assert result["editorText"] == "second question"
    assert await session.branch_tip_id() == second_question.parent_id
    assert _texts(session.agent.state.messages[1:]) == ["first question", "first reply"]
    await session.close()


async def test_navigate_tree_to_the_first_user_message_resets_the_tip(tmp_path):
    session, _ = await _two_turn_session(tmp_path)
    first_question = (await session.entries())[0]

    result = await session.navigate_tree(first_question.id)

    assert result["editorText"] == "first question"
    assert await session.branch_tip_id() is None
    assert len(session.agent.state.messages) == 1  # just the seeded system message
    await session.close()


async def test_navigate_tree_is_a_noop_at_the_current_tip(tmp_path):
    session, _ = await _two_turn_session(tmp_path)
    tip = await session.branch_tip_id()

    result = await session.navigate_tree(tip)

    assert result == {"cancelled": False, "editorText": None, "summaryEntryId": None}
    assert await session.branch_tip_id() == tip
    await session.close()


async def test_navigate_tree_rejects_unknown_ids(tmp_path):
    session, _ = await _two_turn_session(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        await session.navigate_tree("missing-entry")
    await session.close()


async def test_navigate_tree_summarizes_the_abandoned_branch(tmp_path):
    models, registration = _models_with_faux(
        [
            faux_assistant_message("first reply"),
            faux_assistant_message("second reply"),
            faux_assistant_message("branch summary text"),
        ]
    )
    events = []
    session = await _open(tmp_path, models, registration, listener=events.append)
    await session.prompt("first question")
    await session.prompt("second question")
    entries = await session.entries()

    result = await session.navigate_tree(entries[1].id, summarize=True, label="kept")

    summary_id = result["summaryEntryId"]
    assert summary_id
    assert await session.branch_tip_id() == summary_id
    summary_entry = await session.session.get_entry(summary_id)
    assert summary_entry.type == "branch_summary"
    assert "branch summary text" in summary_entry.summary  # karen wraps it in pi's branch-summary preamble
    assert summary_entry.parent_id == entries[1].id
    assert summary_entry.from_hook is False
    assert (await session.entry_labels())[summary_id] == "kept"
    tree_events = [e for e in events if e.get("type") == "session_tree"]
    assert tree_events[-1]["new_leaf_id"] == summary_id
    assert tree_events[-1]["old_leaf_id"] == entries[3].id
    await session.close()


async def test_session_tree_and_labels(tmp_path):
    session, _ = await _two_turn_session(tmp_path)
    entries = await session.entries()
    await session.set_label(entries[1].id, "checkpoint")

    roots = await session.session_tree()

    assert [root.entry.id for root in roots] == [entries[0].id]
    assert roots[0].label is None
    assert roots[0].children[0].label == "checkpoint"
    assert roots[0].children[0].children[0].entry.id == entries[2].id
    with pytest.raises(ValueError, match="not found"):
        await session.set_label("missing-entry", "nope")
    await session.close()


async def test_session_name_round_trip(tmp_path):
    events = []
    session, _ = await _two_turn_session(tmp_path)
    session._listener = events.append

    assert await session.session_name() is None
    await session.set_session_name("  my session  ")

    assert await session.session_name() == "my session"
    assert events[-1] == {"type": "session_info_changed", "name": "my session"}
    with pytest.raises(ValueError, match="empty"):
        await session.set_session_name("   ")
    await session.close()


async def test_session_stats_count_messages_tokens_and_cost(tmp_path):
    from karen_ai import Usage, UsageCost

    reply = faux_assistant_message("hi").model_copy(
        update={"usage": Usage(input=10, output=5, total_tokens=15, cost=UsageCost(total=0.25))}
    )
    models, registration = _models_with_faux([reply])
    session = await _open(tmp_path, models, registration)
    await session.prompt("hello")

    stats = await session.session_stats()

    assert stats["sessionId"] == session.session.metadata.id
    assert stats["sessionFile"].endswith(".jsonl")
    assert stats["userMessages"] == 1
    assert stats["assistantMessages"] == 1
    assert stats["totalMessages"] == 2
    assert stats["toolCalls"] == 0
    assert stats["tokens"]["input"] == 10
    assert stats["tokens"]["output"] == 5
    assert stats["tokens"]["total"] == 15
    assert stats["cost"] == pytest.approx(0.25)
    await session.close()


async def test_user_messages_for_forking_lists_the_prompts(tmp_path):
    session, _ = await _two_turn_session(tmp_path)

    messages = await session.user_messages_for_forking()

    assert [m["text"] for m in messages] == ["first question", "second question"]
    assert all(m["entryId"] for m in messages)
    await session.close()


async def test_fork_copies_the_branch_before_a_user_message(tmp_path):
    session, _ = await _two_turn_session(tmp_path)
    original_id = session.session.metadata.id
    entries = await session.entries()
    second_question = entries[2]

    result = await session.fork(second_question.id, position="before")

    assert result["selectedText"] == "second question"
    assert session.session.metadata.id != original_id
    assert session.session.metadata.parent_session_id == original_id
    assert [entry.id for entry in await session.entries()] == [entry.id for entry in entries[:2]]
    assert _texts(session.agent.state.messages[1:]) == ["first question", "first reply"]
    # both sessions are on disk, and the fork got its own file
    assert len(await session.list_sessions()) == 2
    await session.close()


async def test_fork_before_rejects_non_user_entries(tmp_path):
    session, _ = await _two_turn_session(tmp_path)
    original_id = session.session.metadata.id
    first_reply = (await session.entries())[1]

    with pytest.raises(ValueError, match="Invalid entry ID for forking"):
        await session.fork(first_reply.id, position="before")

    assert session.session.metadata.id == original_id  # nothing was forked
    await session.close()


async def test_fork_at_an_entry_keeps_it(tmp_path):
    session, _ = await _two_turn_session(tmp_path)
    entries = await session.entries()
    first_reply = entries[1]

    result = await session.fork(first_reply.id, position="at")

    assert result["selectedText"] is None
    assert [entry.id for entry in await session.entries()] == [entry.id for entry in entries[:2]]
    assert await session.branch_tip_id() == first_reply.id
    await session.close()


async def test_clone_copies_the_whole_branch(tmp_path):
    session, _ = await _two_turn_session(tmp_path)
    original_id = session.session.metadata.id
    entries = await session.entries()

    await session.clone()

    assert session.session.metadata.id != original_id
    assert session.session.metadata.parent_session_id == original_id
    assert [entry.id for entry in await session.entries()] == [entry.id for entry in entries]
    assert await session.branch_tip_id() == entries[-1].id
    await session.close()


async def test_switch_session_reopens_another_session(tmp_path):
    session, _ = await _two_turn_session(tmp_path)
    first_id = session.session.metadata.id

    await session.clone()
    second_id = session.session.metadata.id
    assert second_id != first_id
    target = next(m for m in await session.list_sessions() if m.id == first_id)

    await session.switch_session(target)

    assert session.session.metadata.id == first_id
    assert _texts(session.agent.state.messages[1:]) == [
        "first question",
        "first reply",
        "second question",
        "second reply",
    ]
    # the clone survives on disk
    assert second_id in [m.id for m in await session.list_sessions()]
    await session.close()


# ---------------------------------------------------------------------------
# auto-retry
# ---------------------------------------------------------------------------


def _retry_policy(**overrides):
    from karen_ai import RetryPolicy

    values = {"enabled": True, "max_retries": 3, "base_delay_ms": 1, "max_agent_delay_ms": 1}
    values.update(overrides)
    return RetryPolicy(**values)


def _transient_failure(text="Error 503 Service Unavailable"):
    return faux_assistant_message([], stop_reason="error", error_message=text)


async def test_auto_retry_recovers_from_a_transient_error(tmp_path):
    responses = [_transient_failure(), faux_assistant_message("recovered answer")]
    models, registration = _models_with_faux(responses)
    events = []
    session = await _open(
        tmp_path, models, registration, retry_policy=_retry_policy(), listener=events.append
    )

    await session.prompt("hello")

    started = [e for e in events if e["type"] == "auto_retry_start"]
    ended = [e for e in events if e["type"] == "auto_retry_end"]
    assert len(started) == 1
    assert started[0]["attempt"] == 1
    assert started[0]["maxAttempts"] == 3
    assert started[0]["errorMessage"] == "Error 503 Service Unavailable"
    assert ended == [{"type": "auto_retry_end", "success": True, "attempt": 1}]
    messages = await _messages(session)
    assert messages[-1].content[0].text == "recovered answer"
    # the failed attempt is durably omitted from the reachable transcript
    assert all(getattr(m, "stop_reason", None) != "error" for m in messages)
    assert registration.get_pending_response_count() == 0
    await session.close()


async def test_auto_retry_gives_up_after_the_budget(tmp_path):
    responses = [_transient_failure(f"Error 503 #{index}") for index in range(3)]
    models, registration = _models_with_faux(responses)
    events = []
    session = await _open(
        tmp_path,
        models,
        registration,
        retry_policy=_retry_policy(max_retries=2),
        listener=events.append,
    )

    await session.prompt("hello")

    assert [e["attempt"] for e in events if e["type"] == "auto_retry_start"] == [1, 2]
    assert [e for e in events if e["type"] == "auto_retry_end"] == [
        {"type": "auto_retry_end", "success": False, "attempt": 2, "finalError": "Error 503 #2"}
    ]
    # the final failure is kept: the budget was already spent
    messages = await _messages(session)
    assert getattr(messages[-1], "stop_reason", None) == "error"
    assert registration.get_pending_response_count() == 0
    await session.close()


async def test_auto_retry_skips_account_limit_errors(tmp_path):
    responses = [
        faux_assistant_message([], stop_reason="error", error_message="insufficient_quota: billing"),
        faux_assistant_message("never used"),
    ]
    models, registration = _models_with_faux(responses)
    events = []
    session = await _open(
        tmp_path, models, registration, retry_policy=_retry_policy(), listener=events.append
    )

    await session.prompt("hello")

    assert [e for e in events if e["type"] in ("auto_retry_start", "auto_retry_end")] == []
    messages = await _messages(session)
    assert getattr(messages[-1], "stop_reason", None) == "error"
    await session.close()


async def test_auto_retry_can_be_disabled(tmp_path):
    responses = [_transient_failure(), faux_assistant_message("never used")]
    models, registration = _models_with_faux(responses)
    events = []
    session = await _open(
        tmp_path,
        models,
        registration,
        retry_policy=_retry_policy(enabled=False),
        listener=events.append,
    )
    assert session.auto_retry_enabled is False

    await session.prompt("hello")

    assert [e for e in events if e["type"] == "auto_retry_start"] == []
    assert getattr((await _messages(session))[-1], "stop_reason", None) == "error"
    await session.close()


async def test_set_auto_retry_enabled_toggles_the_policy(tmp_path):
    models, registration = _models_with_faux([])
    session = await _open(tmp_path, models, registration)
    assert session.auto_retry_enabled is True
    session.set_auto_retry_enabled(False)
    assert session.auto_retry_enabled is False
    session.set_auto_retry_enabled(True)
    assert session.retry.enabled is True
    await session.close()


async def test_agent_end_carries_will_retry(tmp_path):
    responses = [_transient_failure(), faux_assistant_message("recovered answer")]
    models, registration = _models_with_faux(responses)
    session = await _open(tmp_path, models, registration, retry_policy=_retry_policy())
    ends = []
    session.subscribe(
        lambda event, signal: ends.append(event) if getattr(event, "type", None) == "agent_end" else None
    )

    await session.prompt("hello")

    assert [e.will_retry for e in ends] == [True, False]
    await session.close()


async def test_abort_retry_cancels_the_backoff(tmp_path):
    responses = [_transient_failure(), faux_assistant_message("never used")]
    models, registration = _models_with_faux(responses)
    events = []
    session = await _open(
        tmp_path,
        models,
        registration,
        retry_policy=_retry_policy(base_delay_ms=30_000, max_agent_delay_ms=30_000),
        listener=events.append,
    )

    task = asyncio.ensure_future(session.prompt("hello"))
    await _wait_for_event(events, "auto_retry_start")
    assert session.is_retrying is True
    session.abort_retry()
    await asyncio.wait_for(task, timeout=5)

    assert session.is_retrying is False
    assert [e for e in events if e["type"] == "auto_retry_end"] == [
        {"type": "auto_retry_end", "success": False, "attempt": 1, "finalError": "Retry cancelled"}
    ]
    # the cancelled turn left no assistant response behind
    assert getattr((await _messages(session))[-1], "stop_reason", None) != "stop"
    await session.close()


async def test_summarization_retries_are_reported(tmp_path):
    responses = [
        faux_assistant_message("first answer"),
        _transient_failure("overloaded_error"),  # first summary attempt fails
        faux_assistant_message("summary text"),  # the retried attempt succeeds
    ]
    models, registration = _models_with_faux(responses)
    events = []
    session = await _open(
        tmp_path,
        models,
        registration,
        retry_policy=_retry_policy(),
        compaction_settings=CompactionSettings(keep_recent_tokens=10),
        listener=events.append,
    )
    await session.prompt("first")
    events.clear()

    assert await session.run_compaction("manual") is True

    types = [e["type"] for e in events]
    assert types[:4] == [
        "compaction_start",
        "summarization_retry_scheduled",
        "summarization_retry_attempt_start",
        "summarization_retry_finished",
    ]
    scheduled = events[1]
    assert scheduled["attempt"] == 1
    assert scheduled["errorMessage"] == "overloaded_error"
    assert events[2]["source"] == "compaction"
    assert events[2]["reason"] == "manual"
    assert types[-1] == "compaction_end"
    assert events[-1]["compacted"] is True
    await session.close()


async def test_compaction_retry_respects_a_disabled_policy(tmp_path):
    responses = [
        faux_assistant_message("first answer"),
        _transient_failure("overloaded_error"),
    ]
    models, registration = _models_with_faux(responses)
    events = []
    session = await _open(
        tmp_path,
        models,
        registration,
        retry_policy=_retry_policy(enabled=False),
        compaction_settings=CompactionSettings(keep_recent_tokens=10),
        listener=events.append,
    )
    await session.prompt("first")
    events.clear()

    assert await session.run_compaction("manual") is False

    assert [e for e in events if e["type"].startswith("summarization_retry")] == []
    assert events[-1]["compacted"] is False
    await session.close()

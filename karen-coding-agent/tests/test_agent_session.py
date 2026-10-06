"""Offline end-to-end tests for AgentSession, driven by the scripted faux
provider through the real event protocol — no network or credentials."""

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

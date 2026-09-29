"""Telemetry adapter conformance tests (port of pi-telemetry's conformance suite).

The suite runs the callback-adapter contract against `InMemoryTelemetryContext`;
any adapter for the same contract can be substituted.
"""

import collections.abc

import pytest

from karen_ai.telemetry import (
    InMemoryTelemetryContext,
    NOOP_TELEMETRY_CONTEXT,
    SpanOptions,
    SpanStatus,
    create_typed_span_starter,
    define_telemetry_schema,
)
from karen_ai.telemetry.memory import RecordedTelemetrySpan


class Unreadable:
    """Stand-in for the JS Proxy used upstream: every attribute access raises."""

    def __getattr__(self, name):  # pragma: no cover - exercised through callers
        raise RuntimeError("read")


class UnreadableSequence(collections.abc.Sequence):
    """Claims to be a sequence, but raises when actually read."""

    def __getitem__(self, index):
        raise RuntimeError("read")

    def __len__(self):
        raise RuntimeError("inspect")


@pytest.fixture
def context() -> InMemoryTelemetryContext:
    return InMemoryTelemetryContext()


def find_span(spans, name) -> RecordedTelemetrySpan:
    for span in spans:
        if span.name == name:
            return span
    raise AssertionError(f"Expected recorded span {name}")


async def test_admits_once_and_preserves_the_result(context):
    calls = []
    expected = {"value": 42}

    result = await context.start_span(SpanOptions(name="success"), lambda span: calls.append(span) or expected)

    assert calls and len(calls) == 1
    assert result is expected
    span = find_span(context.get_spans(), "success")
    assert span.status == SpanStatus(status="ok")
    assert span.settled is True


async def test_preserves_rejection_values(context):
    sync_error = ValueError("sync")
    with pytest.raises(ValueError) as caught:
        await context.start_span(SpanOptions(name="sync-error"), lambda span: (_ for _ in ()).throw(sync_error))
    assert caught.value is sync_error

    class AsyncError(Exception):
        pass

    async_error = AsyncError("async")

    async def raise_async(span):
        raise async_error

    with pytest.raises(AsyncError) as caught:
        await context.start_span(SpanOptions(name="async-error"), raise_async)
    assert caught.value is async_error

    class UnreadableError(Exception):
        """Raised value whose inspection fails, like an unreadable JS throw payload."""

        def __str__(self):
            raise RuntimeError("read")

    unreadable_error = UnreadableError()

    def raise_unreadable(span):
        raise unreadable_error

    with pytest.raises(UnreadableError) as caught:
        await context.start_span(SpanOptions(name="unreadable-error"), raise_unreadable)
    assert caught.value is unreadable_error

    spans = context.get_spans()
    for name in ["sync-error", "async-error", "unreadable-error"]:
        assert find_span(spans, name).status.status == "error"


async def test_uses_last_explicit_status_without_automatic_overwrite(context):
    def two_statuses(span):
        span.set_status(SpanStatus.failed("Expected", "first"))
        span.set_status(SpanStatus.ok())

    await context.start_span(SpanOptions(name="last-status"), two_statuses)

    thrown = RuntimeError("after explicit status")

    def explicit_then_throw(span):
        span.set_status(SpanStatus.ok())
        raise thrown

    with pytest.raises(RuntimeError):
        await context.start_span(SpanOptions(name="explicit-before-throw"), explicit_then_throw)

    rejected = RuntimeError("after async explicit status")

    async def explicit_then_reject(span):
        span.set_status(SpanStatus.failed("Expected", "async failure"))
        raise rejected

    with pytest.raises(RuntimeError):
        await context.start_span(SpanOptions(name="explicit-before-rejection"), explicit_then_reject)

    def expected_failure(span):
        span.set_status(SpanStatus.failed("Expected", "returned failure"))
        return {"ok": False}

    await context.start_span(SpanOptions(name="expected-failure"), expected_failure)

    spans = context.get_spans()
    assert find_span(spans, "last-status").status == SpanStatus(status="ok")
    assert find_span(spans, "explicit-before-throw").status == SpanStatus(status="ok")
    assert find_span(spans, "explicit-before-rejection").status == SpanStatus.failed(
        "Expected", "async failure"
    )
    assert find_span(spans, "expected-failure").status == SpanStatus.failed("Expected", "returned failure")


async def test_merges_attributes_and_records_ordered_events(context):
    def record(span):
        span.set_attributes({"count": 1, "overwrite": "middle"})
        span.set_attributes({"count": None, "overwrite": "end"})
        span.add_event("first", {"index": 1, "ignored": None})
        span.add_event("second", {"index": 2})

    await context.start_span(
        SpanOptions(name="recording", attributes={"start": "value", "overwrite": "start", "ignored": None}),
        record,
    )

    span = find_span(context.get_spans(), "recording")
    assert span.attributes == {"start": "value", "overwrite": "end", "count": 1}
    assert [(event.name, event.attributes) for event in span.events] == [
        ("first", {"index": 1}),
        ("second", {"index": 2}),
    ]


async def test_ignores_failed_attribute_calls_atomically(context):
    def record(span):
        span.set_attributes({"partial": "must not survive", "unreadable": UnreadableSequence()})

    await context.start_span(SpanOptions(name="atomic-attributes", attributes={"retained": "value"}), record)

    assert find_span(context.get_spans(), "atomic-attributes").attributes == {"retained": "value"}


async def test_makes_calls_after_settlement_inert(context):
    captured = {}

    await context.start_span(
        SpanOptions(name="settled", attributes={"value": "initial"}),
        lambda span: captured.setdefault("span", span),
    )
    settled_span = captured["span"]
    settled_span.set_attributes({"value": "late"})
    settled_span.add_event("late", {"value": True})
    settled_span.set_status(SpanStatus.failed("Late"))

    child_admitted = []

    def child(span):
        child_admitted.append(span)
        return 7

    assert await settled_span.start_span(SpanOptions(name="late-child"), child) == 7
    assert len(child_admitted) == 1

    spans = context.get_spans()
    assert len(spans) == 1
    assert spans[0].attributes == {"value": "initial"}
    assert spans[0].events == []
    assert spans[0].status == SpanStatus(status="ok")


async def test_records_nested_and_concurrent_child_relationships(context):
    import asyncio

    release_first = asyncio.Event()

    async def parent_body(parent):
        async def first_child(span):
            await release_first.wait()

        async def second_child(span):
            return "done"

        first = asyncio.ensure_future(parent.start_span(SpanOptions(name="first-child"), first_child))
        await asyncio.sleep(0)
        second = parent.start_span(SpanOptions(name="second-child"), second_child)
        assert await second == "done"
        release_first.set()
        await first

    await context.start_span(SpanOptions(name="parent"), parent_body)

    spans = context.get_spans()
    parent = find_span(spans, "parent")
    first = find_span(spans, "first-child")
    second = find_span(spans, "second-child")
    assert parent.parent_id is None
    assert first.parent_id == parent.id
    assert second.parent_id == parent.id
    assert second.end_sequence < first.end_sequence
    assert first.end_sequence < parent.end_sequence


async def test_suppresses_unreadable_telemetry_payloads(context):
    calls = []

    def body(span):
        calls.append(span)
        return 9

    assert await context.start_span(Unreadable(), body) == 9
    assert len(calls) == 1
    assert context.get_spans() == []

    def record(span):
        span.set_attributes(Unreadable())
        span.add_event("unreadable-event", Unreadable())
        span.set_status(Unreadable())

    await context.start_span(SpanOptions(name="unreadable-recording"), record)

    recorded = context.get_spans()
    assert len(recorded) == 1
    assert recorded[0].attributes == {}
    assert recorded[0].events == []
    assert recorded[0].status == SpanStatus(status="ok")


async def test_ignores_failed_status_calls_atomically(context):
    rejection = RuntimeError("rejected after unreadable status")

    async def body(span):
        span.set_status(Unreadable())
        raise rejection

    with pytest.raises(RuntimeError):
        await context.start_span(SpanOptions(name="unreadable-status"), body)

    assert find_span(context.get_spans(), "unreadable-status").status.status == "error"


async def test_noop_context_runs_work_without_recording():
    calls = []

    async def body(span):
        span.set_attributes({"anything": 1})
        span.add_event("event")
        span.set_status(SpanStatus.failed("ignored"))
        calls.append(span)
        return "value"

    assert await NOOP_TELEMETRY_CONTEXT.start_span(SpanOptions(name="noop"), body) == "value"
    assert len(calls) == 1
    child = await calls[0].start_span(SpanOptions(name="child"), lambda span: 1)
    assert child == 1


async def test_typed_span_starter_nests_children_and_passes_schemas():
    schema = define_telemetry_schema({"version": 1, "spans": {"request": {"description": "one request"}}})
    context = InMemoryTelemetryContext()
    starter = create_typed_span_starter(context, [schema])

    async def body(child_starter, span):
        span.set_attributes({"provider": "openai"})

        async def child_body(grandchild_starter, child_span):
            child_span.set_attributes({"cache.hit": False})
            return "leaf"

        return await child_starter.start("request.cache", {"cache.hit": False}, child_body)

    result = await starter.start("request", {"provider": "openai"}, body)

    assert result == "leaf"
    assert starter.schemas == (schema,)
    spans = context.get_spans()
    assert [span.name for span in spans] == ["request", "request.cache"]
    assert spans[1].parent_id == spans[0].id
    assert spans[0].attributes == {"provider": "openai"}
    assert spans[1].attributes == {"cache.hit": False}
    assert all(span.settled for span in spans)

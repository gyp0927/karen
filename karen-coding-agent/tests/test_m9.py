"""Tests for CA-M9: thinking-level control, the bash side channel, session
export, settings writes, and the image pipeline."""

import base64
import io
import json
import os
import shutil

import pytest

from karen_ai import ImageContent, create_models
from karen_ai.providers import faux_assistant_message, faux_model, faux_tool_call, register_faux_provider
from karen_agent.session import BranchScan
from karen_coding_agent import AgentSession
from karen_coding_agent import cli as karen_cli
from karen_coding_agent.agent_session import BLOCKED_IMAGE_PLACEHOLDER, THINKING_LEVEL_ENTRY
from karen_coding_agent.settings import (
    image_auto_resize,
    image_block_images,
    load_settings,
    update_settings,
)
from karen_coding_agent.utils import (
    detect_supported_image_mime_type,
    detect_supported_image_mime_type_from_file,
    format_dimension_note,
    normalize_tool_result_images,
    process_image,
    resize_image,
)
from karen_coding_agent.utils.exif_orientation import get_exif_orientation
from karen_ai import TextContent


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
    session = AgentSession(cwd=str(tmp_path), models=models, model=registration.get_model(), **kwargs)
    await session.open()
    return session


async def _branch_entries(session):
    """The entries a resume or an export would see — the current branch, not the
    whole file (an omitted attempt stays in the file, but off-branch)."""
    branch = await session.session.branch("main")
    return await branch.find_entries(BranchScan(order="oldestFirst"))


def _retry_policy():
    from karen_ai import RetryPolicy

    return RetryPolicy(enabled=True, max_retries=3, base_delay_ms=1, max_agent_delay_ms=1)


def _png_bytes(width, height, color=(120, 60, 200)):
    from PIL import Image

    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# thinking level
# ---------------------------------------------------------------------------


async def test_available_thinking_levels_reasoning_model(tmp_path):
    models, registration = _models_with_faux([], reasoning=True)
    session = await _open(tmp_path, models, registration)
    levels = session.get_available_thinking_levels()
    assert "off" in levels and "high" in levels
    assert session.supports_thinking()


async def test_available_thinking_levels_non_reasoning_model(tmp_path):
    models, registration = _models_with_faux([], reasoning=False)
    session = await _open(tmp_path, models, registration)
    assert session.get_available_thinking_levels() == ["off"]
    assert not session.supports_thinking()


async def test_set_thinking_level_emits_event(tmp_path):
    models, registration = _models_with_faux([], reasoning=True)
    session = await _open(tmp_path, models, registration)
    events = []
    session._listener = lambda event: events.append(event)
    await session.set_thinking_level("high")
    assert session.agent.state.thinking_level == "high"
    assert any(e.get("type") == "thinking_level_changed" and e.get("level") == "high" for e in events)
    # setting the same level again does not re-emit
    events.clear()
    await session.set_thinking_level("high")
    assert not events


async def test_set_thinking_level_clamps(tmp_path):
    models, registration = _models_with_faux([], reasoning=False)
    session = await _open(tmp_path, models, registration)
    await session.set_thinking_level("high")
    assert session.agent.state.thinking_level == "off"


async def test_cycle_thinking_level(tmp_path):
    models, registration = _models_with_faux([], reasoning=True)
    session = await _open(tmp_path, models, registration)
    await session.set_thinking_level("off")
    nxt = await session.cycle_thinking_level()
    assert nxt is not None and nxt != "off"


async def test_cycle_thinking_level_unsupported(tmp_path):
    models, registration = _models_with_faux([], reasoning=False)
    session = await _open(tmp_path, models, registration)
    assert await session.cycle_thinking_level() is None


# ---------------------------------------------------------------------------
# bash side channel
# ---------------------------------------------------------------------------


async def test_execute_bash(tmp_path):
    models, registration = _models_with_faux([])
    session = await _open(tmp_path, models, registration)
    result = await session.execute_bash("echo hello && echo world")
    assert result.exit_code == 0
    assert "hello" in result.output and "world" in result.output
    assert not result.cancelled


async def test_execute_bash_streams_and_emits(tmp_path):
    models, registration = _models_with_faux([])
    session = await _open(tmp_path, models, registration)
    events = []
    session._listener = lambda event: events.append(event)
    chunks = []
    result = await session.execute_bash("echo streamed", on_chunk=chunks.append)
    assert "streamed" in result.output
    assert any(e.get("type") == "bash_execution_update" for e in events)


async def test_abort_bash(tmp_path):
    import asyncio

    models, registration = _models_with_faux([])
    session = await _open(tmp_path, models, registration)

    async def run_and_abort():
        task = asyncio.create_task(session.execute_bash("sleep 5"))
        await asyncio.sleep(0.2)
        session.abort_bash()
        return await task

    result = await run_and_abort()
    assert result.cancelled
    assert result.exit_code is None


async def test_cancelling_execute_bash_kills_the_running_command(tmp_path):
    """The other half of `abort_bash`: Ctrl+C in the TUI cancels the run's task
    (`LiveTui.run`'s teardown), and `_run_tui_bash` only re-raises. The child
    has to die in the layer below, or a `!command` keeps writing files behind a
    dead app — the cooperative abort above is too late by then, because the
    cancellation has already unwound through the controller registry.
    """
    import asyncio

    models, registration = _models_with_faux([])
    session = await _open(tmp_path, models, registration)
    started = tmp_path / "started.txt"
    marker = tmp_path / "alive.txt"
    command = (
        f'echo started > "{started.as_posix()}"; '
        f'sleep 1; '
        f'echo alive > "{marker.as_posix()}"'
    )

    task = asyncio.create_task(session.execute_bash(command))
    # The child's own `started` file is the clock: cancelling before the shell
    # runs anything would pass this test without testing anything.
    for _ in range(1500):
        if started.exists():
            break
        await asyncio.sleep(0.01)
    assert started.exists(), "the shell never ran the command"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(1.5)
    assert not marker.exists(), "the !command outlived the run that started it"
    await session.close()


async def test_execute_bash_records_the_result_in_the_transcript(tmp_path):
    """pi's `recordBashResult`: the model sees the command and its output next turn."""
    models, registration = _models_with_faux([])
    session = await _open(tmp_path, models, registration)
    await session.execute_bash("echo recorded")

    bash = next(m for m in session.agent.state.messages if getattr(m, "role", None) == "bashExecution")
    assert bash.command == "echo recorded"
    assert "recorded" in bash.output and bash.exit_code == 0
    # persisted to the session file too
    entries = await session.entries()
    assert any(
        getattr(getattr(entry, "message", None), "role", None) == "bashExecution" for entry in entries
    )
    # and it reaches the LLM conversion as a user message
    converted = session._convert_to_llm(session.agent.state.messages)
    assert any("recorded" in str(getattr(m, "content", "")) for m in converted)
    await session.close()


async def test_execute_bash_exclude_from_context_hides_it(tmp_path):
    models, registration = _models_with_faux([])
    session = await _open(tmp_path, models, registration)
    await session.execute_bash("echo hidden", exclude_from_context=True)
    assert any(getattr(m, "role", None) == "bashExecution" for m in session.agent.state.messages)
    converted = session._convert_to_llm(session.agent.state.messages)
    assert not any("hidden" in str(getattr(m, "content", "")) for m in converted)
    await session.close()


async def test_execute_bash_queues_while_streaming(tmp_path):
    """A run in flight defers the bash message (pi's `_pendingBashMessages`)."""
    models, registration = _models_with_faux([])
    session = await _open(tmp_path, models, registration)
    session.agent.state.is_streaming = True

    await session.execute_bash("echo deferred")

    assert not any(getattr(m, "role", None) == "bashExecution" for m in session.agent.state.messages)
    assert len(session._pending_bash_messages) == 1
    await session.close()


async def test_queued_bash_result_is_flushed_when_the_run_settles(tmp_path):
    """The settle flush itself, driven by a real run rather than by hand.

    pi flushes pending side-channel results in the `finally` of the run, "after
    the agent turn completes to maintain proper message ordering", so the
    message closes the turn — behind that turn's reply.
    """
    from karen_agent.types import AgentTool, AgentToolResult

    models, registration = _models_with_faux(
        [faux_assistant_message([faux_tool_call("side_bash", {})]), faux_assistant_message("done")]
    )
    holder = {}

    async def execute(tool_call_id, params, signal, on_update):
        await holder["session"].execute_bash("echo mid-run")
        return AgentToolResult(content=[TextContent(text="ran")])

    tool = AgentTool(
        name="side_bash",
        description="run a side-channel bash command",
        label="side_bash",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    session = await _open(tmp_path, models, registration, tools=[tool])
    holder["session"] = session

    await session.prompt("go")

    assert session._pending_bash_messages == []
    roles = [getattr(m, "role", None) for m in session.agent.state.messages]
    # pi flushes "after agent turn completes to maintain proper message
    # ordering", so the message closes the turn, behind that turn's reply
    assert roles[-1] == "bashExecution"
    await session.close()


async def test_queued_bash_result_is_flushed_before_the_next_prompt(tmp_path):
    """pi flushes pending bash twice: at the settle and before a new prompt.

    A result recorded in the window between the two (after the agent_end flush,
    while `is_streaming` is still set) has to reach the model in the next turn —
    flushing only at the next run's end would put it after that turn's reply.
    """
    models, registration = _models_with_faux(
        [faux_assistant_message("first"), faux_assistant_message("second")]
    )
    session = await _open(tmp_path, models, registration)
    session.agent.state.is_streaming = True
    await session.execute_bash("echo late")
    assert len(session._pending_bash_messages) == 1
    session.agent.state.is_streaming = False  # the run settles; no agent_end fired

    await session.prompt("next")

    roles = [getattr(m, "role", None) for m in session.agent.state.messages]
    assert session._pending_bash_messages == []
    # the model sees it during the turn for "next", not after the reply
    assert roles.index("bashExecution") < roles.index("assistant")
    await session.close()


async def test_queued_bash_result_survives_a_retried_turn(tmp_path):
    """A result recorded during a failed attempt must not go down with it.

    pi flushes in the run's `finally`, i.e. after the retries settle. Flushing
    at the failed attempt's `agent_end` would put the message *behind* the
    attempt that `_omit_final_attempt` casts away, and the tip rewind would take
    its entry off the branch with it — invisible to the retried turn's model
    call, to a resumed session and to `/export`.
    """
    from karen_agent.types import AgentTool, AgentToolResult

    models, registration = _models_with_faux(
        [
            faux_assistant_message([faux_tool_call("side_bash", {})]),
            faux_assistant_message([], stop_reason="error", error_message="Error 503"),
            faux_assistant_message("recovered"),
        ]
    )
    holder = {}

    async def execute(tool_call_id, params, signal, on_update):
        await holder["session"].execute_bash("echo mid-run")
        return AgentToolResult(content=[TextContent(text="ran")])

    tool = AgentTool(
        name="side_bash",
        description="run a side-channel bash command",
        label="side_bash",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    session = await _open(tmp_path, models, registration, tools=[tool], retry_policy=_retry_policy())
    holder["session"] = session

    await session.prompt("go")

    messages = [entry.message for entry in await _branch_entries(session) if entry.type == "message"]
    roles = [getattr(m, "role", None) for m in messages]
    # the failed attempt is gone, the result it produced is not
    assert all(getattr(m, "stop_reason", None) != "error" for m in messages)
    assert messages[-1].command == "echo mid-run"
    assert roles[-1] == "bashExecution"
    await session.close()


async def test_thinking_level_change_survives_a_retried_turn(tmp_path):
    """A level recorded mid-attempt stays on the branch when the attempt is omitted.

    The change lands between the attempt's own entries, so the rewind has to go
    back to the last entry that is *not* part of the attempt, not simply to the
    last message entry — otherwise the entry is orphaned in the file, and the
    next resume silently falls back to the previous level.
    """
    from karen_agent.types import AgentTool, AgentToolResult

    models, registration = _models_with_faux(
        [
            faux_assistant_message([faux_tool_call("set_level", {})]),
            faux_assistant_message([], stop_reason="error", error_message="Error 503"),
            faux_assistant_message("recovered"),
        ],
        reasoning=True,
    )
    holder = {}

    async def execute(tool_call_id, params, signal, on_update):
        await holder["session"].set_thinking_level("high")
        return AgentToolResult(content=[TextContent(text="level set")])

    tool = AgentTool(
        name="set_level",
        description="change the thinking level",
        label="set_level",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    session = await _open(tmp_path, models, registration, tools=[tool], retry_policy=_retry_policy())
    holder["session"] = session

    await session.prompt("go")

    assert session.agent.state.thinking_level == "high"
    await session.close()

    resumed = await _open(tmp_path, models, registration, fresh=False)
    assert resumed.agent.state.thinking_level == "high"
    await resumed.close()


async def test_omit_final_attempt_reparents_entries_that_are_not_part_of_the_attempt(tmp_path):
    """A rewind takes the failed attempt off the branch and nothing else.

    pi edits the failed message out of the context and unparents nothing, so an
    entry recorded while the failed attempt was still the tip (an RPC
    `set_thinking_level` arriving between the failure and the recovery) keeps its
    place; and one recorded just *before* the failure is where the rewind has to
    land, rather than behind it on the last message entry.
    """
    models, registration = _models_with_faux([faux_assistant_message("hi")], reasoning=True)
    session = await _open(tmp_path, models, registration)
    await session.prompt("hello")

    # a change recorded behind the failing message
    failed = faux_assistant_message([], stop_reason="error", error_message="Error 503")
    session.agent.state.messages.append(failed)
    await session._persist_message(failed)
    await session.set_thinking_level("high")

    await session._omit_final_attempt()

    assert all(getattr(m, "stop_reason", None) != "error" for m in session.agent.state.messages)
    assert len(session.agent.state.messages) == 3
    custom = [entry for entry in await _branch_entries(session) if entry.type == "custom"]
    assert [entry.data for entry in custom] == [{"thinkingLevel": "high"}]
    # still the branch tip, so the retried reply lands behind it
    assert await session.branch_tip_id() == custom[0].id

    # a change recorded in front of the failing message
    await session.set_thinking_level("low")
    failed_again = faux_assistant_message([], stop_reason="error", error_message="Error 500")
    session.agent.state.messages.append(failed_again)
    await session._persist_message(failed_again)

    await session._omit_final_attempt()

    levels = [
        entry
        for entry in await _branch_entries(session)
        if entry.type == "custom" and entry.data == {"thinkingLevel": "low"}
    ]
    assert levels, "the level recorded before the failure went off-branch"
    assert await session.branch_tip_id() == levels[0].id
    await session.close()


async def test_bash_execution_update_carries_the_command_id(tmp_path):
    models, registration = _models_with_faux([])
    session = await _open(tmp_path, models, registration)
    seen = []
    session._listener = lambda event: seen.append(event)
    await session.execute_bash("echo tagged", command_id="cmd-7")
    updates = [e for e in seen if e.get("type") == "bash_execution_update"]
    assert updates and all(e["id"] == "cmd-7" for e in updates)


# ---------------------------------------------------------------------------
# session export
# ---------------------------------------------------------------------------


async def test_export_to_jsonl_linear_chain(tmp_path):
    models, registration = _models_with_faux([faux_assistant_message("answer")])
    session = await _open(tmp_path, models, registration)
    await session.prompt("hello")
    out = str(tmp_path / "export.jsonl")
    path = await session.export_to_jsonl(out)
    assert path == out
    lines = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    assert lines[0]["type"] == "session" and lines[0]["version"] == 3
    assert lines[0]["id"] == session.session.metadata.id
    # entries are re-parented into one linear chain
    assert lines[1]["parentId"] is None
    for prev, cur in zip(lines[1:], lines[2:]):
        assert cur["parentId"] == prev["id"]


async def test_export_to_html_self_contained(tmp_path):
    models, registration = _models_with_faux([faux_assistant_message("hi there")])
    session = await _open(tmp_path, models, registration)
    await session.prompt("hello")
    out = str(tmp_path / "export.html")
    path = await session.export_to_html(out)
    html = open(path, encoding="utf-8").read()
    assert "__SESSION_DATA__" not in html  # template was filled
    assert "hello" in base64.b64decode(_extract_payload(html)).decode("utf-8")


def _extract_payload(html):
    import re

    match = re.search(r'atob\("([A-Za-z0-9+/=]+)"\)', html)
    assert match, "no embedded session payload"
    return match.group(1)


# ---------------------------------------------------------------------------
# settings writes
# ---------------------------------------------------------------------------


def test_update_settings_deep_merge(tmp_path, monkeypatch):
    settings_path = str(tmp_path / "settings.json")
    monkeypatch.setenv("KAREN_SETTINGS_PATH", settings_path)
    update_settings({"defaultModel": "m1", "retry": {"maxRetries": 5}})
    update_settings({"retry": {"baseDelayMs": 250}, "defaultTools": ["-powershell"]})
    data = json.loads(open(settings_path, encoding="utf-8").read())
    # nested dict merged, not replaced
    assert data["retry"] == {"maxRetries": 5, "baseDelayMs": 250}
    assert data["defaultModel"] == "m1"
    assert data["defaultTools"] == ["-powershell"]
    loaded = load_settings(cwd=str(tmp_path))
    assert loaded.settings.retry == {"maxRetries": 5, "baseDelayMs": 250}


def test_update_settings_project_scope(tmp_path):
    project = tmp_path / "proj"
    (project / ".karen").mkdir(parents=True)
    path = update_settings({"defaultModel": "proj-model"}, scope="project", cwd=str(project))
    assert str(path).endswith(os.path.join(".karen", "settings.json"))
    loaded = load_settings(cwd=str(project), global_path=str(tmp_path / "none.json"))
    assert loaded.settings.default_model == "proj-model"


def test_image_settings_helpers():
    assert image_auto_resize(None) is True
    assert image_auto_resize({}) is True
    assert image_auto_resize({"autoResize": False}) is False
    assert image_block_images(None) is False
    assert image_block_images({"blockImages": True}) is True


# ---------------------------------------------------------------------------
# image pipeline
# ---------------------------------------------------------------------------


def test_detect_supported_image_mime_type():
    assert detect_supported_image_mime_type(_png_bytes(4, 4)) == "image/png"
    assert detect_supported_image_mime_type(b"not an image") is None


async def test_detect_supported_image_mime_type_from_file(tmp_path):
    path = tmp_path / "x.png"
    path.write_bytes(_png_bytes(4, 4))
    assert await detect_supported_image_mime_type_from_file(str(path)) == "image/png"
    assert await detect_supported_image_mime_type_from_file(str(tmp_path / "missing.png")) is None


async def test_resize_image_downscales():
    resized = await resize_image(_png_bytes(4000, 3000), "image/png")
    assert resized.was_resized
    assert resized.width <= 2000 and resized.height <= 2000
    note = format_dimension_note(resized)
    assert "Multiply coordinates by" in note


async def test_resize_image_passthrough_when_within_limits():
    resized = await resize_image(_png_bytes(100, 100), "image/png")
    assert not resized.was_resized
    assert format_dimension_note(resized) is None


async def test_process_image_converts_bmp():
    from PIL import Image

    image = Image.new("RGB", (10, 10), (1, 2, 3))
    buffer = io.BytesIO()
    image.save(buffer, "BMP")
    processed = await process_image(buffer.getvalue(), "image/bmp")
    assert processed.ok
    assert processed.mime_type == "image/png"
    assert any("converted from image/bmp" in h for h in processed.hints)


async def test_process_image_omits_undecodable():
    processed = await process_image(b"\x00\x01\x02garbage", "image/x-weird")
    assert not processed.ok
    assert "could not be converted" in processed.message


def test_exif_orientation_detection():
    from PIL import Image

    image = Image.new("RGB", (100, 50), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif)
    assert get_exif_orientation(buffer.getvalue()) == 6
    assert get_exif_orientation(_png_bytes(10, 10)) == 1


async def test_normalize_tool_result_images_resizes_oversized():
    big = base64.b64encode(_png_bytes(4000, 3000)).decode()
    content = [TextContent(text="shot"), ImageContent(data=big, mime_type="image/png")]
    normalized = await normalize_tool_result_images(content)
    assert normalized is not content
    image_block = next(b for b in normalized if b.type == "image")
    assert image_block.data != big  # was resized/re-encoded
    assert any(b.type == "text" and "Multiply coordinates" in b.text for b in normalized)


async def test_normalize_tool_result_images_identity_when_clean():
    small = base64.b64encode(_png_bytes(50, 50)).decode()
    content = [TextContent(text="ok"), ImageContent(data=small, mime_type="image/png")]
    normalized = await normalize_tool_result_images(content)
    assert [getattr(b, "data", None) for b in normalized] == [None, small]


async def test_normalize_tool_result_images_keeps_original_on_failure():
    bad = base64.b64encode(b"\x00garbage-not-image").decode()
    content = [ImageContent(data=bad, mime_type="image/x-weird")]
    normalized = await normalize_tool_result_images(content)
    # failure keeps the original block (unlike the prompt path, which drops it)
    assert normalized[0].data == bad


# ---------------------------------------------------------------------------
# prompt-time image normalization
# ---------------------------------------------------------------------------


async def test_prompt_normalizes_images(tmp_path):
    models, registration = _models_with_faux([faux_assistant_message("ok")], input=["text", "image"])
    session = await _open(tmp_path, models, registration)
    big = base64.b64encode(_png_bytes(4000, 3000)).decode()
    await session.prompt("what is this", images=[ImageContent(data=big, mime_type="image/png")])
    user = next(m for m in session.agent.state.messages if getattr(m, "role", None) == "user")
    text_block = next(b for b in user.content if getattr(b, "type", None) == "text")
    assert "Multiply coordinates by" in text_block.text  # resize hint appended to text
    image_block = next(b for b in user.content if getattr(b, "type", None) == "image")
    assert image_block.data != big


async def test_prompt_drops_failed_image_into_hints(tmp_path):
    models, registration = _models_with_faux([faux_assistant_message("ok")])
    session = await _open(tmp_path, models, registration)
    bad = base64.b64encode(b"\x00garbage").decode()
    await session.prompt("look", images=[ImageContent(data=bad, mime_type="image/x-weird")])
    user = next(m for m in session.agent.state.messages if getattr(m, "role", None) == "user")
    assert all(getattr(b, "type", None) != "image" for b in user.content)
    text_block = next(b for b in user.content if getattr(b, "type", None) == "text")
    assert "could not be converted" in text_block.text


async def test_read_tool_processes_images(tmp_path):
    models, registration = _models_with_faux(
        [faux_assistant_message("done")], input=["text", "image"]
    )
    session = await _open(tmp_path, models, registration)
    (tmp_path / "big.png").write_bytes(_png_bytes(4000, 3000))
    read_tool = next(t for t in session.tools if t.name == "read")
    result = await read_tool.execute("call-1", {"path": "big.png"}, None, None)
    text = next(b.text for b in result.content if getattr(b, "type", None) == "text")
    assert "Read image file" in text
    assert any(getattr(b, "type", None) == "image" for b in result.content)


def _image_size(base64_data):
    from PIL import Image

    with Image.open(io.BytesIO(base64.b64decode(base64_data))) as image:
        return image.size


async def test_read_tool_resize_profile_follows_the_model(tmp_path):
    """pi resolves the read resize profile per call from the current model."""
    from karen_ai import ModelImageInputLimits, ModelImageResizeOptions, ModelInputLimits

    models, registration = _models_with_faux([])
    session = await _open(tmp_path, models, registration)
    (tmp_path / "big.png").write_bytes(_png_bytes(4000, 3000))
    read_tool = next(t for t in session.tools if t.name == "read")

    result = await read_tool.execute("call-1", {"path": "big.png"}, None, None)
    image = next(b for b in result.content if getattr(b, "type", None) == "image")
    assert _image_size(image.data) == (2000, 1500)  # pi's default profile

    narrow = registration.get_model().model_copy(
        update={
            "input_limits": ModelInputLimits(
                images=ModelImageInputLimits(resize=ModelImageResizeOptions(max_width=100, max_height=100))
            )
        }
    )
    session.set_model(narrow)

    result = await read_tool.execute("call-2", {"path": "big.png"}, None, None)
    image = next(b for b in result.content if getattr(b, "type", None) == "image")
    assert _image_size(image.data) == (100, 75)  # the new model's profile


async def test_auto_resize_images_disabled_keeps_original(tmp_path):
    """`images.autoResize=false` (pi's `getImageAutoResize`) skips the resize path."""
    models, registration = _models_with_faux([faux_assistant_message("ok")], input=["text", "image"])
    session = await _open(tmp_path, models, registration, auto_resize_images=False)
    payload = base64.b64encode(_png_bytes(4000, 3000)).decode()
    await session.prompt("what is this", images=[ImageContent(data=payload, mime_type="image/png")])
    user = next(m for m in session.agent.state.messages if getattr(m, "role", None) == "user")
    image_block = next(b for b in user.content if getattr(b, "type", None) == "image")
    assert image_block.data == payload  # untouched, no re-encode
    text_block = next(b for b in user.content if getattr(b, "type", None) == "text")
    assert "Multiply coordinates" not in text_block.text


async def test_resize_fails_when_exif_application_fails(monkeypatch):
    """An EXIF failure aborts the resize instead of sending a sideways image."""
    from karen_coding_agent.utils import image_process

    def boom(image, original_bytes):
        raise RuntimeError("transpose failed")

    monkeypatch.setattr(image_process, "apply_exif_orientation", boom)
    payload = _png_bytes(4000, 3000)
    assert image_process.resize_image_in_process(payload, "image/png") is None
    assert image_process.convert_image_bytes_to_png(payload) is None
    processed = await process_image(payload, "image/png")
    assert not processed.ok
    assert "could not be resized below the inline image size limit" in processed.message


# ---------------------------------------------------------------------------
# images.blockImages
# ---------------------------------------------------------------------------


def _image_content(width=4, height=4):
    return ImageContent(data=base64.b64encode(_png_bytes(width, height)).decode(), mime_type="image/png")


async def test_block_images_replaces_and_dedupes(tmp_path):
    from karen_ai import ToolResultMessage, UserMessage

    models, registration = _models_with_faux([])
    session = await _open(tmp_path, models, registration, block_images=True)
    messages = [
        UserMessage(content=[TextContent(text="look"), _image_content(), _image_content()], timestamp=1),
        ToolResultMessage(tool_call_id="c1", tool_name="read", content=[_image_content()], timestamp=2),
    ]
    converted = session._convert_to_llm(messages)
    assert [b.type for b in converted[0].content] == ["text", "text"]
    assert converted[0].content[0].text == "look"
    assert converted[0].content[1].text == BLOCKED_IMAGE_PLACEHOLDER
    assert [b.type for b in converted[1].content] == ["text"]
    assert converted[1].content[0].text == BLOCKED_IMAGE_PLACEHOLDER


async def test_block_images_leaves_text_only_and_other_roles_alone(tmp_path):
    from karen_ai import AssistantMessage, UserMessage

    models, registration = _models_with_faux([])
    session = await _open(tmp_path, models, registration, block_images=True)
    plain = UserMessage(content=[TextContent(text="no images here")], timestamp=1)
    assistant = AssistantMessage(
        content=[TextContent(text="reply")], api="faux", provider="faux", model="faux", timestamp=2
    )
    converted = session._convert_to_llm([plain, assistant])
    assert converted[0] is plain
    assert converted[1] is assistant


async def test_block_images_is_read_per_request(tmp_path):
    from karen_ai import UserMessage

    models, registration = _models_with_faux([])
    session = await _open(tmp_path, models, registration)
    messages = [UserMessage(content=[TextContent(text="x"), _image_content()], timestamp=1)]
    assert [b.type for b in session._convert_to_llm(messages)[0].content] == ["text", "image"]
    session.block_images = True  # mid-session settings change
    assert [b.type for b in session._convert_to_llm(messages)[0].content] == ["text", "text"]


def test_image_settings_helpers():
    assert image_auto_resize(None) is True
    assert image_auto_resize({}) is True
    assert image_auto_resize({"autoResize": False}) is False
    assert image_block_images(None) is False
    assert image_block_images({"blockImages": True}) is True


# ---------------------------------------------------------------------------
# thinking level persistence
# ---------------------------------------------------------------------------


async def test_thinking_level_recorded_in_transcript(tmp_path):
    models, registration = _models_with_faux([], reasoning=True)
    session = await _open(tmp_path, models, registration)
    await session.set_thinking_level("high")
    custom = [
        entry
        for entry in await session.entries()
        if getattr(entry, "type", None) == "custom"
        and getattr(entry, "custom_type", None) == THINKING_LEVEL_ENTRY
    ]
    assert [entry.data for entry in custom] == [{"thinkingLevel": "high"}]
    # re-setting the same level records nothing new
    await session.set_thinking_level("high")
    assert len([e for e in await session.entries() if getattr(e, "type", None) == "custom"]) == 1


async def test_thinking_level_restored_on_resume(tmp_path):
    models, registration = _models_with_faux([], reasoning=True)
    session = await _open(tmp_path, models, registration)
    await session.set_thinking_level("high")
    await session.close()

    resumed = await _open(tmp_path, models, registration, fresh=False)
    assert resumed.agent.state.thinking_level == "high"
    await resumed.close()


async def test_thinking_level_restore_clamps_to_model(tmp_path):
    """A recorded level the current model can't do is clamped on load."""
    models, registration = _models_with_faux([], reasoning=True)
    session = await _open(tmp_path, models, registration)
    await session.set_thinking_level("high")
    await session.close()

    narrow, narrow_registration = _models_with_faux([], reasoning=False)
    resumed = AgentSession(
        cwd=str(tmp_path),
        models=narrow,
        model=narrow_registration.get_model(),
        sessions_root=str(tmp_path / "sessions"),
    )
    await resumed.open()
    assert resumed.agent.state.thinking_level == "off"
    await resumed.close()


async def test_thinking_level_follows_a_session_switch(tmp_path):
    """The level belongs to the session, not the runtime: every path that
    adopts another session (`switch_session`/`fork`/`clone`, i.e. `/resume`,
    `/fork`, `/clone` and the RPC commands) re-reads it from that branch."""
    models, registration = _models_with_faux([], reasoning=True)
    first = await _open(tmp_path, models, registration)
    await first.set_thinking_level("high")
    first_metadata = first.session.metadata
    await first.close()

    second = await _open(tmp_path, models, registration, fresh=True)
    await second.set_thinking_level("low")
    assert second.session.metadata.id != first_metadata.id

    await second.switch_session(first_metadata)

    assert second.session.metadata.id == first_metadata.id
    assert second.agent.state.thinking_level == "high"
    await second.close()


async def test_thinking_level_follows_a_clone(tmp_path):
    models, registration = _models_with_faux([], reasoning=True)
    session = await _open(tmp_path, models, registration)
    await session.set_thinking_level("high")
    # a stale live value: what any rebind that skipped the restore leaves behind
    session.agent.state.thinking_level = "off"

    await session.clone()

    assert await session._thinking_level_entries()  # the entry was copied along
    assert session.agent.state.thinking_level == "high"
    await session.close()


# ---------------------------------------------------------------------------
# settings writes: never clobber a broken file
# ---------------------------------------------------------------------------


def test_update_settings_refuses_malformed_file(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"defaultModel": "m1",,}', encoding="utf-8")
    monkeypatch.setenv("KAREN_SETTINGS_PATH", str(settings_path))
    with pytest.raises(ValueError) as error:
        update_settings({"defaultModel": "m2"})
    assert "refusing to overwrite" in str(error.value)
    assert settings_path.read_text(encoding="utf-8") == '{"defaultModel": "m1",,}'  # untouched


def test_update_settings_refuses_non_object_file(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setenv("KAREN_SETTINGS_PATH", str(settings_path))
    with pytest.raises(ValueError):
        update_settings({"defaultModel": "m2"})
    assert settings_path.read_text(encoding="utf-8") == "[1, 2, 3]"


# ---------------------------------------------------------------------------
# HTML export: self-contained + only the leaf branch
# ---------------------------------------------------------------------------


def _payload(html):
    return json.loads(base64.b64decode(_extract_payload(html)).decode("utf-8"))


async def test_export_to_html_has_no_external_resources(tmp_path):
    models, registration = _models_with_faux([faux_assistant_message("hi there")])
    session = await _open(tmp_path, models, registration)
    await session.prompt("hello")
    html = open(await session.export_to_html(str(tmp_path / "e.html")), encoding="utf-8").read()
    # no CDN scripts, no remote stylesheets/images — the report opens offline
    assert "<script src=" not in html
    assert "cdn.jsdelivr.net" not in html
    assert "http://" not in html and "https://" not in html
    await session.close()


async def test_export_to_html_renders_only_the_leaf_branch(tmp_path):
    from karen_coding_agent.session_export import export_session_to_html

    entries = [
        {"id": "a", "parentId": None, "type": "message", "message": {"role": "user", "content": "first"}},
        {"id": "b", "parentId": "a", "type": "message", "message": {"role": "assistant", "content": "answer"}},
        {"id": "c", "parentId": "a", "type": "message", "message": {"role": "user", "content": "abandoned"}},
    ]
    path = export_session_to_html({"id": "s1", "cwd": str(tmp_path)}, entries, "b", str(tmp_path / "e.html"))
    data = _payload(open(path, encoding="utf-8").read())
    assert [entry["id"] for entry in data["entries"]] == ["a", "b"]


def test_export_to_html_unknown_leaf_falls_back_to_all_entries(tmp_path):
    from karen_coding_agent.session_export import export_session_to_html

    entries = [{"id": "a", "parentId": None, "type": "message"}]
    path = export_session_to_html({"id": "s1"}, entries, "missing", str(tmp_path / "e.html"))
    assert [entry["id"] for entry in _payload(open(path, encoding="utf-8").read())["entries"]] == ["a"]


#: Runs the report's own script against a minimal DOM so the renderer is
#: actually exercised (a browser-only failure would otherwise ship silently).
_DOM_HARNESS_JS = r"""
import { readFileSync } from "node:fs";
const html = readFileSync(process.argv[2], "utf8");
const js = html.match(/<script>\r?\n([\s\S]*?)\r?\n<\/script>/)[1];
class El {
  constructor(tag) { this.tag = tag; this.children = []; this.className = ""; this._html = ""; }
  set innerHTML(value) { this._html = String(value); this.children = []; }
  get innerHTML() { return this._html; }
  appendChild(child) {
    if (!child || typeof child.dump !== "function") {
      throw new TypeError("appendChild got a non-node: " + JSON.stringify(child));
    }
    this.children.push(child);
    return child;
  }
  querySelectorAll() { return []; }
  dump() { return this._html + this.children.map((c) => c.dump()).join(""); }
}
const app = new El("div");
globalThis.document = {
  getElementById: (id) => (id === "app" ? app : null),
  createElement: (tag) => new El(tag),
};
eval(js);
process.stdout.write(app.dump());
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the report renderer")
def test_export_html_renderer_runs(tmp_path):
    import re
    import subprocess

    from karen_coding_agent.session_export import export_session_to_html

    entries = [
        {
            "id": "e1",
            "parentId": None,
            "type": "message",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "**bold** and `code` and <script>alert(1)</script>"}],
            },
        },
        {
            "id": "e1b",
            "parentId": "e1",
            "type": "message",
            # the payload is UTF-8 behind base64: `atob` alone would mojibake this
            "message": {"role": "user", "content": [{"type": "text", "text": "中文测试 ✨ café 日本語"}]},
        },
        {
            "id": "e2",
            "parentId": "e1b",
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolName": "bash",
                "content": [{"type": "text", "text": "output"}],
            },
        },
        {
            "id": "e3",
            "parentId": "e2",
            "type": "message",
            # A fence only opens at the start of a line (CommonMark), so this one
            # stays inline text instead of vanishing into a dropped block.
            "message": {"role": "assistant", "content": "Here is code: ```py\nx=1\n``` and done."},
        },
        {
            "id": "e4",
            "parentId": "e3",
            "type": "message",
            "message": {"role": "assistant", "content": "before\n```py\nx=1\n```\nafter"},
        },
        {
            "id": "e5",
            "parentId": "e4",
            "type": "message",
            "message": {"role": "assistant", "content": "```\nunterminated"},
        },
        {
            "id": "e6",
            "parentId": "e5",
            "type": "message",
            "message": {
                "role": "bashExecution",
                "command": "echo hello",
                "output": "hello\n",
                "exitCode": 0,
                "cancelled": False,
                "truncated": False,
                "timestamp": 1,
            },
        },
        {
            "id": "e7",
            "parentId": "e6",
            "type": "message",
            "message": {
                "role": "bashExecution",
                "command": "boom",
                "output": "bad",
                "exitCode": 3,
                "cancelled": False,
                "truncated": False,
                "timestamp": 2,
            },
        },
    ]
    path = export_session_to_html({"id": "s1"}, entries, "e7", str(tmp_path / "e.html"))
    html = open(path, encoding="utf-8").read()
    (tmp_path / "harness.mjs").write_text(_DOM_HARNESS_JS, encoding="utf-8")

    result = subprocess.run(
        [shutil.which("node"), str(tmp_path / "harness.mjs"), path],
        capture_output=True,
        text=True,
        encoding="utf-8",  # the renderer emits UTF-8; the Windows locale codec would mangle it
    )

    assert result.returncode == 0, result.stderr
    rendered = result.stdout
    assert "<strong>bold</strong>" in rendered and "<code>code</code>" in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered  # escaped, not injected
    assert '<details class="tool"><summary>bash</summary>' in rendered  # tool result as a node
    assert re.search(r"<script>alert", rendered) is None
    # the base64 payload is UTF-8: non-ASCII session text survives the round trip
    assert "中文测试 ✨ café 日本語" in rendered
    # no placeholder ever leaks, and no text loses content to a swallowed fence
    assert "\x00" not in rendered
    assert "Here is code: ```py" in rendered and "x=1" in rendered
    assert '<pre><code class="lang-py">x=1</code></pre>' in rendered
    assert '<pre><code>unterminated</code></pre>' in rendered
    # side-channel bash output is rendered, failed ones flagged
    assert "<div class=\"cmd\">$ echo hello</div>" in rendered
    assert "<pre>hello\n</pre>" in rendered
    assert "<div class=\"cmd\">$ boom</div>" in rendered
    assert "(exit 3)" in rendered


async def test_export_html_from_file_reads_a_native_session(tmp_path):
    """`karen --export FILE`: the on-disk storage log, not just the v3 export."""
    from karen_coding_agent.session_export import export_html_from_file

    models, registration = _models_with_faux(
        [faux_assistant_message("first"), faux_assistant_message("second")]
    )
    session = await _open(tmp_path, models, registration)
    await session.prompt("one")
    await session.prompt("two")
    # move the tip back so the second turn is an abandoned branch
    entries = await session.entries()
    await session.navigate_tree(entries[1].id)
    await session.prompt("three")
    native_path = session.session.metadata.path
    session_id = session.session.metadata.id
    await session.close()

    out = export_html_from_file(native_path, str(tmp_path / "out.html"))
    data = _payload(open(out, encoding="utf-8").read())
    texts = [
        block["text"]
        for entry in data["entries"]
        for block in (entry.get("message") or {}).get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    assert "one" in texts and "first" in texts  # kept: the tip's ancestors
    assert "three" in texts  # the new turn on the restored branch
    assert "two" not in texts  # off-branch, like pi's getPath
    assert data["header"]["id"] == session_id


async def test_export_html_from_file_reads_a_v3_export(tmp_path):
    models, registration = _models_with_faux([faux_assistant_message("reply")])
    session = await _open(tmp_path, models, registration)
    await session.prompt("question")
    jsonl = await session.export_to_jsonl(str(tmp_path / "v3.jsonl"))
    await session.close()

    from karen_coding_agent.session_export import export_html_from_file

    data = _payload(open(export_html_from_file(jsonl, str(tmp_path / "v3.html")), encoding="utf-8").read())
    roles = [(entry.get("message") or {}).get("role") for entry in data["entries"]]
    assert roles == ["user", "assistant"]


def test_cli_export_flag_writes_html(tmp_path, capsys, monkeypatch):
    session_file = tmp_path / "native.jsonl"
    session_file.write_text(
        "\n".join(
            [
                json.dumps({"kind": "header", "id": "s1", "cwd": str(tmp_path)}),
                json.dumps({"kind": "value", "namespace": "pi.branch.tip", "key": "main", "value": "b"}),
                json.dumps(
                    [
                        {"kind": "entry", "id": "a", "seq": 1, "type": "message",
                         "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}},
                        {"kind": "entry", "id": "b", "parentId": "a", "seq": 2, "type": "message",
                         "message": {"role": "assistant", "content": [{"type": "text", "text": "yo"}]}},
                    ]
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "out.html"

    code = karen_cli.main(["--export", str(session_file), str(out_path)])

    assert code == 0
    assert "Exported to:" in capsys.readouterr().out
    assert [e["id"] for e in _payload(out_path.read_text(encoding="utf-8"))["entries"]] == ["a", "b"]


def test_cli_export_flag_reports_a_missing_file(tmp_path, capsys):
    code = karen_cli.main(["--export", str(tmp_path / "nope.jsonl")])
    assert code == 1
    assert "File not found" in capsys.readouterr().err

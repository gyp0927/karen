"""End-to-end check of M3 (compaction + hooks + prompt templates) against real DeepSeek.

Scenario: a session with a long history is compacted for real (the model writes
the summary), the compaction entry is persisted to a JSONL session, the session
is reopened (simulated restart), the rebuilt context is fed back into the agent
loop, and the model answers a question whose answer only survives in the
summary. Also exercises branch summarization, the hook registry, and prompt
template loading.

Credentials: same precedence as examples/agent_smoke.py (credentials file, then
DEEPSEEK_API_KEY).

Usage:
    python examples/agent_compaction_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

from karen_ai import CreateModelsOptions, JsonFileCredentialStore, TextContent, ToolCall, UserMessage, create_models
from karen_ai.providers import deepseek_provider, faux_assistant_message
from karen_agent import (
    AgentContext,
    AgentLoopConfig,
    BeforeCompactionEvent,
    HookRegistry,
    agent_loop,
    convert_to_llm,
    format_prompt_template_invocation,
    load_prompt_templates,
    models_stream_fn,
)
from karen_agent.compaction import (
    CompactionSettings,
    GenerateBranchSummaryOptions,
    compact,
    generate_branch_summary,
    prepare_compaction,
)
from karen_agent.result import Err
from karen_agent.session import (
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionRepo,
)
from karen_agent.session.commit import insert_entry
from karen_agent.session.ids import uuid7
from karen_agent.session.types import BranchScan, CompactionEntry
from karen_agent.session.values import branch_tip, set_value
from karen_agent.session.context import build_session_context

MODEL_ID = "deepseek-v4-pro"
DEFAULT_CREDENTIALS = Path.home() / ".karen/credentials.json"

FILLER = "This is filler discussion about weather, sports, and cooking recipes to bulk up the transcript. " * 4


def _build_models():
    override = os.environ.get("KAREN_CREDENTIALS_PATH")
    path = Path(override).expanduser() if override else DEFAULT_CREDENTIALS
    if path.exists():
        return create_models(CreateModelsOptions(credentials=JsonFileCredentialStore(path))), f"credentials file {path}"
    return create_models(), "DEEPSEEK_API_KEY"


def _now() -> int:
    return int(time.time() * 1000)


async def _append(branch, message):
    return await branch.append_message(message)


async def main() -> int:
    models, source = _build_models()
    models.set_provider(deepseek_provider())
    model = models.get_model("deepseek", MODEL_ID)
    print(f"-> deepseek/{model.id}  auth from {source}")

    sessions_dir = tempfile.mkdtemp(prefix="karen-compaction-")
    repo = JsonlSessionRepo(sessions_dir)
    session = await repo.create(JsonlSessionCreateOptions(cwd=sessions_dir))
    branch = await session.create_branch("main", None)

    # --- a transcript whose key fact will fall inside the compacted range ----
    await _append(branch, UserMessage(
        content="Please create a file named project-notes.txt containing exactly: alpha beta gamma.",
        timestamp=_now(),
    ))
    await _append(branch, faux_assistant_message([
        TextContent(text="Creating the file now."),
        ToolCall(id="t1", name="write", arguments={"path": "project-notes.txt", "content": "alpha beta gamma"}),
    ]))
    for i in range(6):
        await _append(branch, UserMessage(content=f"{FILLER} (round {i})", timestamp=_now()))
        await _append(branch, faux_assistant_message(f"Acknowledged round {i}. {FILLER}"))
    await _append(branch, UserMessage(content="Thanks, all done!", timestamp=_now()))
    await _append(branch, faux_assistant_message("You're welcome!"))

    path_entries = await branch.find_entries(BranchScan(order="oldestFirst"))
    print(f"-> transcript: {len(path_entries)} entries")

    # --- prompt templates + hooks (local machinery) --------------------------
    template_dir = Path(sessions_dir) / "templates"
    template_dir.mkdir()
    (template_dir / "recap.md").write_text("Recap $1 in one sentence.", encoding="utf-8")
    templates = load_prompt_templates(str(template_dir))
    assert templates.prompt_templates and templates.prompt_templates[0].name == "recap"
    print(f"-> prompt template: {format_prompt_template_invocation(templates.prompt_templates[0], ['the session'])}")

    hook_events = []

    def observe_compaction(event):
        hook_events.append((event.reason, len(event.preparation.messages_to_summarize)))

    hooks = HookRegistry(lambda error, hook: print(f"hook error in {hook}: {error}"))
    hooks.on("before_compaction", observe_compaction)

    # --- compaction for real -------------------------------------------------
    settings = CompactionSettings(keep_recent_tokens=400)
    preparation_result = prepare_compaction(path_entries, settings)
    if isinstance(preparation_result, Err) or preparation_result.value is None:
        print("FAILED: prepare_compaction returned nothing to compact", file=sys.stderr)
        return 1
    preparation = preparation_result.value
    print(
        f"-> compaction plan: summarize {len(preparation.messages_to_summarize)} messages, "
        f"retain {len(preparation.retained_tail)}, ~{preparation.tokens_before} tokens before"
    )
    await hooks.run(
        "before_compaction", BeforeCompactionEvent(reason="threshold", preparation=preparation)
    )
    assert hook_events == [("threshold", len(preparation.messages_to_summarize))]

    print("-> asking the model to summarize (compact)...")
    compact_result = await compact(preparation, models, model)
    if isinstance(compact_result, Err):
        print(f"FAILED: compact error {compact_result.error.code}: {compact_result.error}", file=sys.stderr)
        return 1
    compacted = compact_result.value
    print("--- summary ---------------------------------------------------------")
    print(compacted.summary)
    print("---------------------------------------------------------------------")
    print(f"-> usage: {compacted.usage.total_tokens} tokens; details: {compacted.details}")
    assert compacted.usage and compacted.usage.total_tokens > 0
    assert compacted.details == {"readFiles": [], "modifiedFiles": ["project-notes.txt"]}

    # --- persist the compaction entry, then simulate a restart ---------------
    compaction_id = uuid7()

    async def append_compaction(mutator):
        tip = await mutator.get_value(branch_tip("main"))
        await mutator.commit([
            insert_entry(CompactionEntry(
                id=compaction_id,
                parent_id=tip.value,
                summary=compacted.summary,
                retained_tail=compacted.retained_tail,
                tokens_before=compacted.tokens_before,
                details=compacted.details,
                usage=compacted.usage,
                from_hook=False,
            )),
            set_value(branch_tip("main"), compaction_id),
        ])

    await session.mutate(append_compaction)
    await session.close()

    metadata = (await repo.list(JsonlSessionListOptions(cwd=sessions_dir)))[0]
    session = await repo.open(metadata)
    branch = await session.branch("main")
    reopened_entries = await branch.find_entries(BranchScan(order="oldestFirst"))
    context_messages = await build_session_context(reopened_entries)
    print(f"-> reopened: {len(reopened_entries)} stored entries -> {len(context_messages)} context messages")
    assert context_messages[0].role == "compactionSummary"
    assert context_messages[0].summary == compacted.summary
    llm_messages = convert_to_llm(context_messages)
    assert llm_messages[0].content[0].text.startswith(
        "The conversation history before this point was compacted into the following summary:"
    )

    # --- continue the conversation from the compacted context -----------------
    context = AgentContext(messages=context_messages)
    config = AgentLoopConfig(model=model, convert_to_llm=convert_to_llm)
    question = UserMessage(
        content="What is the name of the project file, and what exact content did we write into it?",
        timestamp=_now(),
    )
    stream = agent_loop([question], context, config, None, models_stream_fn(models))
    async for event in stream:
        if event.type == "message_update" and event.assistant_message_event.type == "text_delta":
            print(event.assistant_message_event.delta, end="", flush=True)
    print()
    messages = await stream.result()
    final = messages[-1]
    answer = "".join(getattr(c, "text", "") for c in getattr(final, "content", []))
    if getattr(final, "stop_reason", None) == "error":
        print(f"FAILED: {final.error_message}", file=sys.stderr)
        return 1
    if "project-notes.txt" not in answer:
        print(f"FAILED: answer does not know the file name: {answer!r}", file=sys.stderr)
        return 1
    print("-> the model recovered the file name from the compaction summary")

    # --- branch summarization for real ----------------------------------------
    print("-> asking the model to summarize the branch...")
    branch_result = await generate_branch_summary(
        reopened_entries, GenerateBranchSummaryOptions(models=models, model=model)
    )
    if isinstance(branch_result, Err):
        print(f"FAILED: branch summary error {branch_result.error.code}: {branch_result.error}", file=sys.stderr)
        return 1
    assert branch_result.value.summary.startswith("The user explored a different conversation branch")
    print(f"-> branch summary: {branch_result.value.summary[:200]}...")

    await session.close()
    print("OK: compaction + hooks + prompt templates all verified against the real model")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

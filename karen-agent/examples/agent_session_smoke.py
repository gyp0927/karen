"""End-to-end check of the agent loop + M1 session persistence against real DeepSeek.

Phase 1 runs the echo-tool loop and persists every produced message into a JSONL
session branch (entry + branch tip + usage row in one commit per message).
Phase 2 simulates a process restart: a fresh repo discovers and reopens the
session, loads the transcript from disk, and continues the conversation with a
follow-up that is only answerable from the persisted history.

Credentials: same precedence as examples/agent_smoke.py (credentials file, then
DEEPSEEK_API_KEY).

Usage:
    python examples/agent_session_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

from karen_ai import CreateModelsOptions, JsonFileCredentialStore, TextContent, UserMessage, create_models
from karen_ai.providers import deepseek_provider
from karen_agent import AgentContext, AgentLoopConfig, AgentTool, AgentToolResult, agent_loop, models_stream_fn
from karen_agent.session import (
    BranchScan,
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionRepo,
    MessageEntry,
    SessionInvariantError,
    UsageRow,
    branch_tip,
    insert_entry,
    insert_usage,
    set_value,
)

MODEL_ID = "deepseek-v4-pro"
SECRET = "karen-session-42"
DEFAULT_CREDENTIALS = Path.home() / ".karen" / "credentials.json"


def _build_models():
    override = os.environ.get("KAREN_CREDENTIALS_PATH")
    path = Path(override).expanduser() if override else DEFAULT_CREDENTIALS
    if path.exists():
        return create_models(CreateModelsOptions(credentials=JsonFileCredentialStore(path))), f"credentials file {path}"
    return create_models(), "DEEPSEEK_API_KEY"


def _echo_tool() -> AgentTool:
    async def execute(tool_call_id, params, signal, on_update):
        return AgentToolResult(content=[TextContent(text=f"echo: {params['value']}")])

    return AgentTool(
        name="echo",
        description="Echo back the given value verbatim.",
        label="Echo",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string", "description": "The text to echo back"}},
            "required": ["value"],
        },
        execute=execute,
    )


def _convert_to_llm(messages):
    return [m for m in messages if getattr(m, "role", None) in ("system", "user", "assistant", "toolResult")]


async def _run_agent(prompt_text, history, models, model):
    """One agent-loop run; returns the run's new messages."""
    context = AgentContext(messages=list(history), tools=[_echo_tool()])
    config = AgentLoopConfig(model=model, convert_to_llm=_convert_to_llm)
    prompt = UserMessage(content=prompt_text, timestamp=int(time.time() * 1000))
    stream = agent_loop([prompt], context, config, None, models_stream_fn(models))
    async for event in stream:
        if event.type == "message_update" and event.assistant_message_event.type == "text_delta":
            print(event.assistant_message_event.delta, end="", flush=True)
        elif event.type == "tool_execution_start":
            print(f"\n[tool ->] {event.tool_name}({event.args})")
        elif event.type == "tool_execution_end":
            text = "".join(getattr(c, "text", "") for c in event.result.content)
            print(f"[tool <-] {event.tool_name}: {text}")
    print()
    return await stream.result()


async def _append_message(session, branch_name, message):
    """Persist one message: entry + branch tip + (for assistants) usage row, one commit."""
    entry_id = session.id_generator.next()

    async def do(mutator):
        tip = await mutator.get_value(branch_tip(branch_name))
        if tip is None:
            raise SessionInvariantError(f"Unknown branch: {branch_name}")
        writes = [insert_entry(MessageEntry(id=entry_id, parent_id=tip.value, message=message)),
                  set_value(branch_tip(branch_name), entry_id)]
        usage = getattr(message, "usage", None)
        if getattr(message, "role", None) == "assistant" and usage is not None and usage.total_tokens:
            writes.append(
                insert_usage(
                    UsageRow(id=session.id_generator.next(), usage=usage, entry_id=entry_id, adjustment=False)
                )
            )
        await mutator.commit(writes)

    await session.mutate(do)
    return entry_id


async def main() -> int:
    models, source = _build_models()
    models.set_provider(deepseek_provider())
    model = models.get_model("deepseek", MODEL_ID)
    print(f"-> deepseek/{model.id}  auth from {source}")

    root = tempfile.mkdtemp(prefix="karen-agent-session-")
    cwd = os.getcwd()

    # ---- phase 1: run + persist ------------------------------------------------
    repo = JsonlSessionRepo(root)
    session = await repo.create(JsonlSessionCreateOptions(cwd=cwd))
    await session.create_branch("main", None)

    print(f"\n== phase 1: run + persist (session {session.metadata.id})")
    run1 = await _run_agent(f'Call the echo tool with value="{SECRET}", then repeat what it returned.', [], models, model)
    for message in run1:
        await _append_message(session, "main", message)
    stats = await session.get_stats()
    print(
        f"persisted {len(run1)} messages; session stats: {stats.message_count} messages, "
        f"{stats.usage.total_tokens} tokens, cost total={stats.usage.cost.total}"
    )
    path = session.metadata.path
    await session.close()

    # ---- phase 2: restart + resume --------------------------------------------
    repo2 = JsonlSessionRepo(root)  # fresh repo, as after a process restart
    listed = await repo2.list(JsonlSessionListOptions(cwd=cwd))
    resumed = await repo2.open(listed[0])
    branch = await resumed.branch("main")
    entries = await branch.find_entries(BranchScan(order="oldestFirst"))
    history = [e.message for e in entries]
    print(f"\n== phase 2: resumed {len(history)} messages from {path}")
    print(f"   roles on disk -> memory: {[getattr(m, 'role', '?') for m in history]}")

    follow_up = (
        "Without calling any tool: what was the exact value I asked you to pass to the echo tool earlier? "
        "Answer with just the value."
    )
    run2 = await _run_agent(follow_up, history, models, model)
    for message in run2:
        await _append_message(resumed, "main", message)
    stats2 = await resumed.get_stats()
    print(f"after resume: {stats2.message_count} messages, {stats2.usage.total_tokens} tokens cumulative")
    await resumed.close()

    # ---- verify -----------------------------------------------------------------
    last = run2[-1]
    answer = "".join(getattr(c, "text", "") for c in getattr(last, "content", []))
    print("-" * 60)
    print(f"follow-up answer: {answer!r}")
    if getattr(last, "stop_reason", None) == "error":
        print(f"FAILED: {last.error_message}", file=sys.stderr)
        return 1
    if SECRET not in answer:
        print(f"FAILED: answer does not contain {SECRET!r} (history did not survive the round trip)",
              file=sys.stderr)
        return 1
    if stats2.usage.total_tokens <= stats.usage.total_tokens:
        print("FAILED: cumulative usage did not grow after the resumed run", file=sys.stderr)
        return 1
    print("OK: transcript + usage survived restart; follow-up answered from persisted history")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

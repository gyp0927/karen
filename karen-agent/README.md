# karen-agent

Agent loop on top of [karen-ai](../karen-ai) — a Python port of
[`@earendil-works/pi`](https://github.com/earendil-works/pi)'s `packages/agent`
(core loop layer only; the coding-agent app, MCP, and TUI are out of scope for now).

The loop works with `AgentMessage` throughout and transforms to karen-ai `Message`
objects only at the LLM call boundary, so applications can keep custom message
shapes (compaction markers, notifications, …) in the transcript and decide in
`convert_to_llm` how — or whether — each one reaches the model.

## Layout

| Module | Contents |
| --- | --- |
| `types.py` | `AgentContext`, `AgentTool`, `AgentToolResult`, `AgentLoopConfig`, hook payloads, the 10 `AgentEvent` types |
| `stream_fn.py` | default stream-fn registry + `models_stream_fn()` (bridges a karen-ai `Models` registry into the loop) |
| `agent_loop.py` | `agent_loop` / `agent_loop_continue` and their `run_*` async variants — the full port of pi's `agent-loop.ts` |
| `session/` | durable session persistence (pi's `harness/session/`): entries, branches, typed values/lists, usage rows, fork, resume |
| `session/jsonl/` | format-4 JSONL storage + `JsonlSessionRepo` (one file per session under a sessions root) |
| `session/memory.py` | `MemoryStorage` / `MemorySessionRepo` (in-memory backend, same `Storage`/`Session` contract) |
| `utils/usage.py` | `empty_usage` / `add_usage` (pi's `harness/utils/usage.ts`) |

## Sessions (M1)

`karen_agent.session` ports pi's session persistence layer:

```python
from karen_agent.session import JsonlSessionRepo, JsonlSessionCreateOptions, JsonlSessionListOptions

repo = JsonlSessionRepo("~/.karen/sessions")
session = await repo.create(JsonlSessionCreateOptions(cwd=os.getcwd()))
branch = await session.create_branch("main", None)
entry_id = await branch.append_message(UserMessage(content="hi", timestamp=ms))
await session.close()

# resume: discover + reopen, entries/values/lists/usage replay from the file
metadata = (await repo.list(JsonlSessionListOptions(cwd=os.getcwd())))[0]
session = await repo.open(metadata)

# fork: TreeForkOptions() copies the whole tree; BranchForkOptions(branch, entry_id, position)
# copies one branch ancestry up to an entry (lane state restarts idle, op/usage state excluded)
fork = await repo.fork(session.metadata, BranchForkOptions(branch="main", entry_id=entry_id))
```

- **Storage model**: every commit is one JSONL line — entries (`message` / `compaction` /
  `branch_summary` / `custom`), usage rows, scalar value set/delete, list append/delete.
  All writes go through a per-session `MutationLine`; a mutator allows exactly one commit.
- **Wire format**: pi's format v4 — camelCase keys, header line with `v`/`kind`/`id`/
  `storageVersion`/`createdAt`/`cwd`, single-write transactions as bare objects. Loaded
  messages are coerced back into karen-ai models via the `Message` role union; unknown
  shapes stay plain dicts.
- **Crash safety**: appends are line-atomic; a torn final line is discarded and the file
  repaired on open; snapshot rewrites (fork) publish via temp-file + rename.
- **Not ported** (out of M1 scope): the durable runtime operation state machine
  (`OperationState` leaves, `pi.op.*` payloads are typed `Any`), legacy v3 session
  migration (v3 files are detected and rejected), pi's `FileSystem` capability abstraction
  (direct `pathlib` I/O instead), and the chord `Context` parameter.

See [`examples/session_smoke.py`](examples/session_smoke.py) for a runnable
create → fork → resume round trip (no API key needed).

## Usage

```python
import asyncio
from karen_ai import Context, TextContent, UserMessage, create_models
from karen_agent import AgentContext, AgentLoopConfig, AgentTool, AgentToolResult, agent_loop, models_stream_fn

async def execute(tool_call_id, params, signal, on_update):
    return AgentToolResult(content=[TextContent(text=f"echo: {params['value']}")])

echo = AgentTool(
    name="echo",
    description="Echo back the given value.",
    label="Echo",
    parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
    execute=execute,
)

async def main():
    models = create_models()  # credentials via ~/.karen/credentials.json or env vars
    model = models.get_model("deepseek", "deepseek-v4-pro")
    context = AgentContext(messages=[], tools=[echo])
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda ms: [m for m in ms if getattr(m, "role", None) in ("user", "assistant", "toolResult")],
    )
    stream = agent_loop([UserMessage(content="Say hi via the echo tool", timestamp=0)],
                        context, config, None, models_stream_fn(models))
    async for event in stream:
        ...  # agent_start / turn_start / message_* / tool_execution_* / turn_end / agent_end
    messages = await stream.result()

asyncio.run(main())
```

See [`examples/agent_smoke.py`](examples/agent_smoke.py) for a runnable DeepSeek
round trip (question → tool call → answer).

## What the loop gives you

- **Tool execution**: JSON-schema validation + JS-semantics coercion of arguments
  (via karen-ai's `validate_tool_arguments`), optional `prepare_arguments` shim,
  parallel execution with completion-order `tool_execution_end` events and
  source-order result messages; a single `execution_mode="sequential"` tool
  serializes the whole batch.
- **Safety**: a `"length"`-stopped (truncated) assistant message fails all of its
  tool calls instead of executing possibly-borked arguments.
- **Hooks**: `before_tool_call` (block / mutate args), `after_tool_call`
  (field-wise result overrides), `finish_turn` / `prepare_next_turn` /
  `prepare_request` (turn lifecycle, model + thinking-level swaps),
  `transform_context` (prune/compact before conversion), `get_api_key`
  (per-call key resolution), `get_steering_messages` / `get_follow_up_messages`
  (mid-run injection and run extension).
- **Termination control**: the batch stops early only when *every* finalized
  tool result has `terminate=True`.

## Development

```bash
../.venv/Scripts/python.exe -m pip install -e ./karen-agent --no-deps  # karen-ai is local
cd karen-agent && ../.venv/Scripts/python.exe -m pytest -q
```

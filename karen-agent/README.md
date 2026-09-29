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

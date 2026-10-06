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
| `agent.py` | `Agent` (pi's `agent.ts`): stateful wrapper with transcript state, event subscription, steering/follow-up queues, abort and `wait_for_idle` |
| `config.py` | harness config validation (pi's `harness/config.ts`): tool-name/retry/compaction validation, `RetryPolicy` + defaults |
| `skills.py` | skill loading (pi's `harness/skills.ts`): `SKILL.md` discovery, ignore files, metadata diagnostics, `format_skill_invocation` |
| `env/` | execution environment (pi's `harness/types.ts` FileSystem/Shell + `harness/env/nodejs.ts`): `ExecutionEnv` protocols, stable `FileError`/`ExecutionError` codes, `LocalExecutionEnv` |
| `search.py` | session search service interfaces (pi's `search/index.ts`) — protocols only |
| `proxy.py` | `stream_proxy` (pi's `proxy.ts`): stream fn that routes LLM calls through a proxy server speaking the `data: `-line event protocol |
| `messages.py` | harness message shapes (`bashExecution`/`custom`/`branchSummary`/`compactionSummary`) + `convert_to_llm` (pi's `harness/messages.ts`) |
| `result.py` | `Result`/`Ok`/`Err` + `CompactionError`/`BranchSummaryError` (pi's `harness/types.ts` result helpers) |
| `hooks.py` | `HookRegistry`: pi's 11 harness hooks with their aggregation semantics + `HarnessStreamOptions`/`StreamOptionsPatch` |
| `resources.py` | `Skill` / `PromptTemplate` / `Resources` |
| `prompt_templates.py` | prompt-template loading (frontmatter, diagnostics) + `$1`/`$@`/`$ARGUMENTS`/`${@:N:L}` substitution |
| `system_prompt.py` | `format_skills_for_system_prompt` (the `<available_skills>` block) |
| `compaction/` | compaction + branch summarization (pi's `harness/compaction/`): cut points, token estimation, summary generation, file-op tracking |
| `tools/` | built-in tools (pi's `harness/tools/`): `read`, `write`, `edit`, `bash` + supporting utils (edit-diff, image detection, path utils, file mutation queue, local shell) |
| `session/` | durable session persistence (pi's `harness/session/`): entries, branches, typed values/lists, usage rows, fork, resume |
| `session/context.py` | project session entries into model-context messages (pi's `harness/session/context.ts`) |
| `session/jsonl/` | format-4 JSONL storage + `JsonlSessionRepo` (one file per session under a sessions root) |
| `session/memory.py` | `MemoryStorage` / `MemorySessionRepo` (in-memory backend, same `Storage`/`Session` contract) |
| `utils/` | `usage.py` (pi's usage.ts), `truncate.py` (line/byte truncation), `output_capture.py` (bounded shell-output views), `adaptive_publisher.py` (rate-limited publishing), `shell_output.py` (pi's shell-output.ts capture collector) |

## Compaction, hooks, prompt templates (M3)

`karen_agent.compaction` ports pi's context compaction:

```python
from karen_agent.compaction import CompactionSettings, compact, prepare_compaction
from karen_agent.result import Err

preparation = prepare_compaction(path_entries, CompactionSettings()).value  # None when not applicable
await hooks.run("before_compaction", BeforeCompactionEvent(reason="threshold", preparation=preparation))
result = await compact(preparation, models, model)  # the model writes the summary
if not isinstance(result, Err):
    ...  # persist CompactionEntry(summary, retained_tail, tokens_before, details, usage)
```

- **Cut points & budgets**: `estimate_context_tokens` (provider usage + trailing char/4
  estimate, images = 4800 chars), `find_cut_point` (keeps ~`keep_recent_tokens`,
  never cuts at tool results, splits oversized turns at the turn start),
  `should_compact` (`tokens > context_window - reserve_tokens`, defaults
  16384/20000).
- **Summaries**: `compact()` runs pi's structured-checkpoint prompt (or the
  update variant when a previous summary exists), and a second turn-prefix
  summary when the cut splits a turn. File operations from read/write/edit tool
  calls are tracked across compactions and appended as `<read-files>` /
  `<modified-files>` tags. All summary requests go through a caller-owned
  one-request boundary (`compact_with_request` / `generate_summary_with_request`)
  with `cache_retention="none"` + a fresh uuid7 `session_id`.
- **Branch summaries**: `collect_entries_for_branch_summary` +
  `generate_branch_summary` summarize an abandoned branch before navigation.
- **`karen_agent.messages.convert_to_llm`** is the harness-grade boundary:
  `compactionSummary`/`branchSummary` become `<summary>`-wrapped user messages,
  `bashExecution` renders as `Ran \`cmd\` + output`, `custom` becomes a user
  message, unknown roles drop. Session loading keeps these as plain dicts, so
  all accessors accept models *and* camelCase dicts.
- **`karen_agent.session.context`**: `build_context_entries` (latest compaction
  + tail only), `session_entry_to_context_messages` (assistant
  error/aborted/deferred messages drop out), `build_session_context` (custom
  entries resolve through `entry_projectors`).
- **`HookRegistry`** (pi's `harness/hooks.ts`): 11 hooks — `before_run`
  (prompt chaining + injection), `before_drive` (fail-closed), `before_run_end`
  (last follow-up wins), `transform_context`, `before_request` (stream-options
  patches merged, diff returned), `before_payload`, `after_response`,
  `before_tool` (args chaining, first block wins, handler errors block),
  `after_tool` (field-wise merge), `before_compaction`/`before_navigation`
  (first decline-or-result wins). Handler errors go to the `report_error`
  callback; `close()` makes later `on`/`run` raise.
- **Prompt templates** (`prompt_templates.py`): load `.md` files (direct
  children only) with YAML frontmatter, per-file diagnostics instead of
  exceptions, `parse_command_args` shell-style quoting and `substitute_args`
  placeholders. `system_prompt.py` renders `<available_skills>`.

M3 deviations from pi, all documented at the port sites: no chord `Context`
parameter (an explicit `signal=` keyword threads aborts into summary requests);
pi-ai's assistant-call retry layer (`retryAssistantCall`/`RetryPolicy`) is not
ported — karen-ai adapters retry transient HTTP errors via `max_retries`; the
hook registry drops pi's lanes/effect-gates/telemetry spans (events are exactly
the `HookMap` payloads, handlers take just the event); template loading uses
synchronous `pathlib` I/O and PyYAML. Prompt constants are byte-identical to pi.

See [`examples/agent_compaction_smoke.py`](examples/agent_compaction_smoke.py)
for the real-API check: compact a session with a live model, persist the
compaction entry, reopen the session, and continue the conversation from the
summary (verified against DeepSeek).

## Demo CLI (M4)

[`examples/karen_cli.py`](examples/karen_cli.py) puts the whole stack behind an
interactive terminal agent:

```bash
python examples/karen_cli.py [--cwd PATH] [--model ID] [--new]
printf 'hello\n/quit\n' | python examples/karen_cli.py --new   # scripted/piped
```

- Streams replies, prints tool calls (`[tool ->] write(path='a.py', ...)`), and
  a context-token estimate after every turn.
- Resumes the most recent session for the working directory (sessions under
  `~/.karen/sessions`, override with `KAREN_SESSIONS_ROOT`); `/new` starts
  fresh. Every message and usage row is persisted as it arrives.
- **Auto-compaction**: when estimated context tokens cross
  `context_window - reserve_tokens`, the model writes a structured summary that
  is persisted as a compaction entry and the context rebuilds from it.
  `/compact [focus]` triggers it manually. A context overflow instead compacts
  with reason `"overflow"` and retries the turn once (see M7).
- **Prompt templates**: `/templates` lists templates from `.karen/prompts`
  (project) and `~/.karen/prompts` (user); `/name args...` expands `$1`, `$@`,
  `${@:N:L}` and sends the result.
- **Hooks**: a `HookRegistry` is bridged into the loop's `before_tool_call` /
  `after_tool_call` hooks — the demo registers a path guard (write/edit outside
  the working directory is blocked) and a tool-call counter.
- The leading system message carries prompt + tool declarations (pi's
  `initialState` pattern), so resumed sessions replay tools without extra
  system messages.

Verified against DeepSeek: piped session runs a write tool call, compacts,
answers a question from the summary, and a second process resumes the session
and answers from the compacted history.

## Agent class, skills, config (M5)

`karen_agent.agent` ports pi's `agent.ts` — the stateful, app-facing wrapper
around the loop:

```python
from karen_agent import Agent, AgentInitialState, create_builtin_tools, models_stream_fn

agent = Agent(
    stream_fn=models_stream_fn(models),
    initial_state=AgentInitialState(system_prompt=prompt, model=model, tools=create_builtin_tools(cwd)),
)
unsubscribe = agent.subscribe(lambda event, signal: ...)   # awaited in subscription order
await agent.prompt("hello")        # str | AgentMessage | list, plus optional images
agent.steer(UserMessage(...))      # injected after the current turn
agent.follow_up(UserMessage(...))  # runs when the agent would otherwise stop
agent.abort()                      # then: await agent.wait_for_idle()
agent.reset()                      # keeps the replayed prompt/tool baseline
await agent.continue_()            # pi's continue(); renamed — `continue` is a keyword
```

- `AgentInitialState` seeds the leading system message from `system_prompt` +
  `tools` (unless `messages` already starts with one); `state.system_prompt`
  replays it, and assigning `state.tools` / `state.messages` copies the list.
- Steering and follow-up queues drain `"one-at-a-time"` (default) or `"all"`;
  `has_queued_messages()` / `peek_queued_messages()` / `clear_all_queues()`.
- Failed runs follow pi: the error becomes an assistant message with stop reason
  `"error"`/`"aborted"` emitted through the normal message/turn/agent events
  (also recorded in `state.error_message`) instead of raising out of `prompt()`.
- `state.pending_tool_calls` / `state.is_streaming` / `state.streaming_message`
  track the live run; `wait_for_idle()` resolves after `agent_end` listeners
  settle.

`karen_agent.skills` ports `harness/skills.ts`: `load_skills()` walks directories
recursively for `SKILL.md` (and direct root `.md` files with frontmatter), honors
`.gitignore` / `.ignore` / `.fdignore`, validates names (`a-z0-9-`, ≤64 chars,
must match the directory) and descriptions (≤1024 chars) as warnings, and returns
`SkillDiagnostic`s instead of raising. `format_skill_invocation()` renders the
explicit-invocation prompt; `format_skills_for_system_prompt()` (M3) offers the
loaded skills to the model. `load_sourced_skills()` tags results with
application-defined provenance values.

`karen_agent.config` ports `harness/config.ts`: `validate_tool_names`,
`validate_retry_policy`, `validate_compaction_settings`, and `RetryPolicy` +
`DEFAULT_RETRY_POLICY` (3 retries, 1s base delay, 60s agent-delay cap).

M5 deviations: `RetryPolicy` is declared and validated but not yet consumed by
the loop (karen-ai adapters retry via `max_retries`); pi's `RangeError`/`TypeError`
become Python `ValueError`/`TypeError`; skill loading uses synchronous `pathlib`
I/O, PyYAML (parser error text differs), and pathspec's `GitIgnoreSpec` in place
of the `ignore` npm package; no chord `Context` parameter.

See [`examples/agent_class_smoke.py`](examples/agent_class_smoke.py) — verified
against DeepSeek: the agent writes a file, a steered question is injected
mid-run, and the model answers it by reading a loaded skill file.

## Execution environment, search, proxy (M6)

`karen_agent.env` ports pi's `ExecutionEnv` capability interface
(`harness/types.ts`: `FileSystem` + `Shell`) and its Node implementation
(`harness/env/nodejs.ts`) as `LocalExecutionEnv`:

```python
from karen_agent import LocalExecutionEnv, ShellExecOptions, execute_shell_with_capture
from karen_agent.result import Ok

env = LocalExecutionEnv("/path/to/project")
await env.write_file("notes/todo.txt", "ship it\n")          # creates parents
info = (await env.file_info("notes/todo.txt")).value          # FileInfo(kind="file", ...)
reader = (await env.open_text_line_reader("notes/todo.txt")).value  # strict-LF, torn-tail aware

result = await env.exec("cat notes/todo.txt", ShellExecOptions(timeout=5))
captured = await execute_shell_with_capture(env, "make build")  # pi's default tail capture + spill
await env.cleanup()  # kills any processes still running from exec
```

- **Never-throws contract**: every operation returns `Result` with stable,
  backend-independent error codes — `FileError.code` ∈
  `aborted|not_found|permission_denied|not_directory|is_directory|invalid|not_supported|unknown`,
  `ExecutionError.code` ∈ `aborted|timeout|shell_unavailable|spawn_error|callback_error|unknown`.
- `exec` reuses the M2 shell engine (bash resolution, process-tree kills,
  spill) and adds env merging (`inherit_env` + per-call `env`), per-command
  `cwd`, bounded capture (`ShellOutputLimits`) and active-process tracking for
  `cleanup()`. Path handling ports pi's `resolvePath` (`~` expansion, `file://`
  URLs, cwd-relative).
- `utils/shell_output.py`'s `execute_shell_with_capture` is the compatibility
  collector from pi's `shell-output.ts`: runs `exec` with pi's default
  tail-retained 50KB/2000-line capture + spill and folds aborts
  (`cancelled=True`) and — with `return_execution_errors=True` — execution
  failures into one `ShellCaptureResult`.
- `karen_agent.search` ports pi's `search/index.ts` as pure protocols:
  `SearchQuery` / `SessionSearchHit` / `EntrySearchHit` /
  `SessionSearchService`; implementations live outside karen-agent.
- `karen_agent.proxy` ports `proxy.ts`: `stream_proxy(model, context,
  ProxyStreamOptions(auth_token=..., proxy_url=...))` POSTs
  `{model, context, options}` to `{proxyUrl}/api/stream` with Bearer auth and
  reconstructs the partial assistant message client-side from the server's
  `data: `-line events (the server strips `partial` to save bandwidth). A clean
  EOF without a terminal `done`/`error` surfaces as
  `"Connection closed by proxy server before the response completed"`. Use it
  as an `Agent`'s `stream_fn` when provider credentials live behind a server.

M6 deviations: no chord `Context` parameter — an explicit `signal=` keyword
threads aborts into every env method; `ShellExecOptions.on_update` still
receives complete snapshots (M2 deviation), so `on_chunk` diffs consecutive
snapshots and reports nothing when the retained window slides; text file I/O
disables newline translation (Node byte-verbatim semantics); the proxy keeps
only a connect timeout (pi's `fetch` sets none).

See [`examples/agent_env_proxy_smoke.py`](examples/agent_env_proxy_smoke.py) —
keyless verification: env ops against the real local bash, and an `Agent`
driven by `stream_proxy` against a scripted loopback proxy server.

## Overflow detection & recovery (M7)

M7 closes the last pi-ai gap: `karen_ai.utils.overflow` ports
`utils/overflow.ts` — `is_context_overflow(message, context_window)` detects
provider context-overflow errors (23 message patterns plus Cerebras' bodyless
400/413, minus throttling/rate-limit lookalikes), silent overflow (a successful
response whose `usage.input + cache_read` exceeds the context window, z.ai
style), and length-stop overflow (Xiaomi MiMo's zero-output `"length"` stop on
a filled context). `is_recoverable_length(message, desired_max_output)` flags
truncated responses worth one compact-and-retry attempt. In pi, only the
durable runtime and the coding-agent app consume this; karen has no durable
runtime, so the consumer is the demo CLI, which now implements the overflow
branch of pi's `AgentSession._checkCompaction`:

- an overflow **error** or recoverable length stop drops the failed attempt
  from the transcript (`_omit_final_attempt`, pi's `_omitRecoveryAttempt`),
  compacts with reason `"overflow"`, and retries the turn once via
  `agent_loop_continue`; a second overflow keeps the failure and gives up with
  pi's "recovery failed after one compact-and-retry attempt" message;
- a **silent overflow** on a successful response compacts with reason
  `"overflow"` without retrying (a completed response cannot be continued);
- the check skips aborted messages and messages from a different model (pi's
  model-switch guard).

Simplifications, documented at `_overflow_action`: the CLI transcript is
append-only, so pi's projection/context-edit retention guards are trivially
true and dropped; and only fresh post-run messages are checked, so pi's stale
pre-compaction guard cannot trigger. This also retires the dead `"overflow"`
literal the `before_compaction` hook declared since M3 — something can finally
produce it.

## Built-in tools (M2)

`karen_agent.tools` ports pi's four built-in tools. Factories take an optional
`cwd` (default: the process cwd at call time) in place of pi's
`ExecutionToolContext`; `create_builtin_tools(cwd)` returns all four:

```python
from karen_agent.tools import create_builtin_tools

context = AgentContext(messages=[], tools=create_builtin_tools("/path/to/project"))
```

- **read** — text files (offset/limit, 2000-line/50KB head truncation with
  continuation hints) and images (jpg/png/gif/webp/bmp magic-byte detection,
  base64 attachments, optional `image_processor` hook for conversion/resizing).
- **write** — creates parent directories, writes raw UTF-8 (no newline
  translation), serialized per path through a mutation queue.
- **edit** — exact-unique-match replacements with fuzzy fallback (smart quotes,
  dashes, NBSP folding) that preserves untouched lines byte-for-byte, CRLF/BOM
  round-tripping, `prepare_arguments` tolerating JSON-string/single-object/
  legacy `oldText`/`newText` shapes, diff + unified patch in `details`.
- **bash** — runs through a real bash (Git Bash on Windows, /bin/bash→PATH→sh
  on POSIX), combined stdout+stderr, tail truncation (last 2000 lines/50KB)
  with full output spilled to a temp file, timeout and abort kill the whole
  process tree, streaming partial results via `on_update`.

Error messages, truncation footers, and result `details` shapes match pi
byte-for-byte (camelCase keys). Differences from pi, all consequences of
dropping the `ExecutionEnv` capability layer (consistent with M1): direct
`pathlib`/subprocess I/O instead of a `FileSystem`/`Shell` interface; shell
output updates publish complete snapshots in-process instead of pi's
replace/append/slide delta protocol; pi's 2s durable-checkpoint throttle is
dropped (karen's update callback has no checkpoint channel); stderr merges
into stdout at OS level (`stderr=STDOUT`) instead of interleaving two pipes;
spill files are named `karen-output-*.log`.

See [`examples/agent_tools_smoke.py`](examples/agent_tools_smoke.py) for a
real-API check: the model writes a file, edits it, and verifies it with bash
(verified against DeepSeek).

> **Note:** your `convert_to_llm` must keep `"system"` messages. The loop
> declares the executable tool set via `tools_added` on a system message —
> filtering system messages out hides all tools from the model.

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
create → fork → resume round trip (no API key needed), and
[`examples/agent_session_smoke.py`](examples/agent_session_smoke.py) for a
real-API integration check: an agent-loop run persisted to a session, then a
simulated restart that reopens the session and continues the conversation from
the persisted transcript (verified against DeepSeek).

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
        convert_to_llm=lambda ms: [m for m in ms if getattr(m, "role", None) in ("system", "user", "assistant", "toolResult")],
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

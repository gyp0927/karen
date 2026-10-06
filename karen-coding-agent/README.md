# karen-coding-agent

The `karen` CLI coding assistant — the application layer on top of
[karen-agent](../karen-agent), and the karen equivalent of
[`@earendil-works/pi`](https://github.com/earendil-works/pi)'s
`packages/coding-agent` (much smaller scope: no TUI, RPC, MCP, or extensions
yet — see the milestone list).

## M1: AgentSession + CLI

`AgentSession` (`src/karen_coding_agent/agent_session.py`) binds
karen_agent's `Agent` to durable session persistence and automatic
compaction — the karen equivalent of pi coding-agent's
`core/agent-session.ts`, simplified:

```python
from karen_coding_agent import AgentSession

session = AgentSession(cwd=os.getcwd(), models=models, model=model)
await session.open()                     # resumes the most recent session for cwd
session.subscribe(lambda event, signal: ...)   # agent events (deltas, tool calls)
await session.prompt("fix the tests")    # runs to completion, incl. recovery
await session.close()
```

- **Persistence as it happens**: every message is appended to the JSONL
  session (with usage rows) as its `message_end` event arrives, so a crash
  mid-run loses nothing.
- **Threshold auto-compaction**: after each run, when estimated context
  tokens cross `context_window - reserve_tokens`, the model writes a
  structured summary that is persisted as a compaction entry and the context
  rebuilds from it.
- **Overflow recovery** (the overflow branch of pi's
  `AgentSession._checkCompaction`, via karen-ai's
  `is_context_overflow`/`is_recoverable_length`): a context-overflow error or
  recoverable length stop **rewinds the branch tip** to persistently omit the
  failed attempt (pi's `_omitRecoveryAttempt`), compacts with reason
  `"overflow"`, and retries the turn once via `agent.continue_()`; a second
  overflow keeps the failure and emits `overflow_give_up`. A silent overflow
  on a *successful* response (usage over the window) compacts without
  retrying. Aborted messages and messages from a different model are skipped,
  like pi.
- **Hooks**: a `HookRegistry` is bridged into the agent's
  `before_tool_call`/`after_tool_call`, and `before_compaction` can decline
  or replace any compaction.
- **Lifecycle events** for the UI via `listener`: `session_opened`,
  `compaction_start`/`compaction_end`, `overflow_retry`, `overflow_give_up`.

Simplifications vs pi (documented at the port sites): no settings manager,
session projections, context edits, auto-retry, or extension events; pi's
retention/staleness guards are dropped because karen sessions are append-only
and only fresh post-run messages are checked.

### The `karen` CLI

```bash
karen [--cwd PATH] [--model ID] [--new]     # interactive REPL
karen -p "summarize this repo"              # headless print mode
printf 'hello\n/quit\n' | karen --new       # piped REPL (how the smokes drive it)
```

- Streams replies, prints tool calls (`[tool ->] write(path='a.py', ...)`)
  and a context-token estimate after every turn. In print mode tool chatter
  goes to stderr, leaving stdout for the reply.
- Sessions live under `~/.karen/sessions` (override with
  `KAREN_SESSIONS_ROOT`), resumed per working directory; `/new` starts fresh.
- Slash commands: `/help`, `/new`, `/compact [focus]`, `/templates`,
  `/quit`; `/name args...` invokes a prompt template from `.karen/prompts`
  (project) or `~/.karen/prompts` (user), with `$1`/`$@`/`${@:N:L}`
  substitution.
- Ctrl-C aborts the running turn (`agent.abort()` + `wait_for_idle()`).

Credentials come from `~/.karen/credentials.json` (override with
`KAREN_CREDENTIALS_PATH`) or provider environment variables; the default
model is `deepseek/deepseek-v4-pro` (override with `--model` or
`KAREN_MODEL`).

## Roadmap

Later milestones (tracked in the repo root README): find/grep/ls/powershell
tools (M2), print/JSON machine-readable output (M3), settings file (M4), RPC
mode / extensions / MCP / TUI (unscheduled).

## Development

```bash
../.venv/Scripts/python.exe -m pip install -e ./karen-coding-agent --no-deps  # karen-ai/karen-agent are local
cd karen-coding-agent && ../.venv/Scripts/python.exe -m pytest -q
```

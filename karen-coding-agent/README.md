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

Simplifications vs pi (documented at the port sites): no settings manager on
the session itself (the CLI loads the settings files, see M4), no session
projections, context edits, auto-retry, or extension events; pi's
retention/staleness guards are dropped because karen sessions are append-only
and only fresh post-run messages are checked.

### The `karen` CLI

```bash
karen [--cwd PATH] [--model ID] [--new]     # interactive REPL
karen -p "summarize this repo"              # headless text mode: final reply on stdout
karen "one" "two"                           # headless: prompts run sequentially
karen --mode json "prompt"                  # headless JSON event stream
karen --mode rpc                            # JSON command protocol on stdin/stdout (M5)
printf 'hello\n/quit\n' | karen --new       # piped REPL (how the smokes drive it)
```

- Interactive mode streams replies live and prints tool calls
  (`[tool ->] write(path='a.py', ...)`) plus a context-token estimate after
  every turn.
- Headless modes (pi's `runPrintMode`; M3): **text** (default) prints only the
  final assistant message's text to stdout — tool chatter and session notices
  go to stderr — and exits 1 when the final message is an error or aborted;
  **json** (`--mode json`) emits the session header line followed by one JSON
  event per line (pi's `json-event.ts` shape: `message_update` carries only
  cumulative `usage` + the delta sub-event with `partial` stripped,
  `toolcall_start` gains `id`/`toolName`) and stays machine-parseable end to
  end. Deviations from pi: no TTY auto-detection (karen's REPL is designed to
  be piped), positional prompts imply print mode instead of becoming an
  interactive initial message, and `@file`/image arguments are not supported
  yet.
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

## M2: find/grep/ls/powershell tools

`src/karen_coding_agent/tools/` ports pi coding-agent's `core/tools/`
application-level tools, and `create_default_tools(cwd)` is now
`AgentSession`'s default tool set in pi's order: **read, bash, powershell
(Windows only), edit, write, grep, find, ls**.

- **find / grep** (`find.py`, `grep.py`): pi shells out to `fd`/`rg` binaries
  downloaded on demand from GitHub releases; karen walks and searches
  **in-process** (`walk.py` + `globs.py`) with the same semantics — pi's
  glob dialects (fd-style full-path patterns get an implicit `**/` prefix),
  gitignore-aware pruning (`.git` always pruned, hidden files included,
  deepest `.gitignore`/`.ignore` wins, `.ignore` outranks `.gitignore` in the
  same directory), `path:line: text` match / `path-line- text` context
  format, long-line truncation to 500 chars, binary files searched only up
  to the first NUL byte, and pi's exact notices/details
  (`resultLimitReached`/`matchLimitReached`/`linesTruncated`/`truncation`).
  Known deviations: ignore files *above* the search root and the global
  gitignore are not consulted, `.fdignore` is not read, results are sorted
  case-insensitively (fd/rg order is unspecified), and regex syntax is
  Python `re`.
- **ls** (`ls.py`): pure-Python like pi's pure-JS original.
- **powershell** (`powershell.py`, Windows only): karen-agent's
  `create_shell_tool` over PowerShell with pi's
  `-NoProfile -NonInteractive -ExecutionPolicy Bypass -Command` invocation
  and UTF-8 output prefix (`pwsh.exe` preferred, `powershell.exe` fallback).
  pi's `PI_*` session env vars and spawn hooks are not ported yet.

Verified against real DeepSeek: print mode had the model call
`find(pattern='src/**/*.py')` → `grep(pattern='TODO', path='src')` and
answer from the tool results.

## M4: settings files

`src/karen_coding_agent/settings.py` ports the mechanism of pi's
`core/settings-manager.ts`, scoped down to what karen can consume. Global
`~/.karen/settings.json` (override with `KAREN_SETTINGS_PATH`) plus project
`<cwd>/.karen/settings.json`; the project file deep-merges over the global
one (nested objects merge recursively; `defaultTools` gets pi's special
`+name`/`-name` modifier merge). Like pi, files are not schema-validated:
unknown keys are ignored, wrong-typed values dropped, and a malformed file
yields a startup `[settings warning: ...]` on stderr and is skipped.

Ported subset (camelCase wire keys, like pi):

```json
{
  "defaultProvider": "deepseek",
  "defaultModel": "deepseek-v4-pro",
  "sessionDir": "~/.karen/sessions",
  "shellPath": "C:/Program Files/Git/bin/bash.exe",
  "shellCommandPrefix": "shopt -s expand_aliases",
  "compaction": { "enabled": true, "reserveTokens": 16384, "keepRecentTokens": 20000 },
  "prompts": ["~/extra-prompts"],
  "defaultTools": ["-powershell"]
}
```

- `defaultProvider`/`defaultModel` resolve after `--provider`/`--model` and
  `KAREN_PROVIDER`/`KAREN_MODEL`, before the built-in defaults.
- `sessionDir` applies when `KAREN_SESSIONS_ROOT` is not set; `shellPath` and
  `shellCommandPrefix` configure the bash tool; `prompts` adds extra template
  directories (lowest precedence); `defaultTools` selects the session's tool
  set (`["read", "grep"]` replaces the defaults, `["-powershell", "+ls"]`
  modifies them).
- `~` is expanded in path settings. Writes (pi's `/settings` command) and the
  rest of pi's Settings (TUI, extensions, analytics, retry, themes, …) are
  not ported.

## M5: RPC mode

`src/karen_coding_agent/rpc.py` ports pi's `modes/rpc/` protocol onto karen's
`AgentSession`: commands are one JSON object per line on stdin, responses and
events are one JSON object per line on stdout.

```bash
karen --mode rpc [--new] [--cwd PATH]
printf '%s\n' '{"type":"prompt","message":"say hi","id":"1"}' '{"type":"get_state","id":"2"}' \
  | karen --mode rpc --new
```

- **stdout** starts with the session header line (same as `--mode json`) and
  then carries only two kinds of lines:
  - responses: `{"id", "type":"response", "command", "success":true, "data":{…}}`
    or `{"…", "success":false, "error":"…"}`; `id` echoes the command's id
    (`null` for parse errors);
  - events: the `--mode json` shapes (`agent_start`, `message_update` with
    delta sub-events, `tool_execution_start/end`, `agent_end`, …) plus
    `compaction_start`/`compaction_end`/`overflow_retry`/`overflow_give_up`.
  Everything else (session notices, hook errors) goes to stderr.
- **Commands**: `prompt`, `steer`, `follow_up`, `abort`, `clear_queue`,
  `new_session`, `get_state`, `set_model`, `set_steering_mode`,
  `set_follow_up_mode`, `get_available_models`, `get_messages`,
  `get_last_assistant_text`, `get_entries`, `compact`,
  `set_auto_compaction`. Unknown commands and malformed lines are answered
  with `success:false` (`command:"parse"` for JSON errors) — the loop keeps
  running; stdin EOF exits 0.
- **`prompt` while a run is active** is queued as a steering message and
  answered `{"disposition":"queued"}` (pi's default `streamingBehavior:
  "steer"`); when idle it answers `{"disposition":"started"}` and the run's
  progress arrives on the event stream. Deviation: karen reports the
  disposition at command time, where pi resolves its response after preflight.
- **`new_session`** opens a brand-new session for the cwd (closing the old
  one) and emits the new header line; pi's `parentSession` (fork) is not
  supported. `get_state` reports model, thinkingLevel, isStreaming,
  isCompacting, steeringMode/followUpMode, sessionId/sessionFile,
  autoCompactionEnabled, messageCount and pendingMessageCount.
- **`set_auto_compaction`** gates the post-run threshold check only; overflow
  recovery (compact-and-retry) always runs. Manual `compact` answers
  `{"compacted": bool}`.
- `get_entries` answers branch entry *ids* + `leafId` instead of pi's
  `SessionEntry` objects (karen's AgentSession emits no entry events).
- Not ported: the thinking-level commands, auto-retry, pi's bash side
  channel, fork/clone/switch/export-html/session-stats, `get_commands` and
  the extension UI sub-protocol (karen has no extensions or TUI yet), and
  image inputs on `prompt`/`steer`/`follow_up`.

Verified against real DeepSeek over real stdio: header → `prompt` → full event
stream → `get_last_assistant_text` = `"RPC-OK"`, model switching,
`new_session` rebinding (second header), unknown-command/parse-error
responses, and a clean exit 0 on EOF.

## Roadmap

Later milestones (tracked in the repo root README): extensions / MCP / TUI /
image input / fork and branch-navigation RPC commands (unscheduled).

## Development

```bash
../.venv/Scripts/python.exe -m pip install -e ./karen-coding-agent --no-deps  # karen-ai/karen-agent are local
cd karen-coding-agent && ../.venv/Scripts/python.exe -m pytest -q
```

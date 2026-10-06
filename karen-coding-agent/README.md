# karen-coding-agent

The `karen` CLI coding assistant — the application layer on top of
[karen-agent](../karen-agent), and the karen equivalent of
[`@earendil-works/pi`](https://github.com/earendil-works/pi)'s
`packages/coding-agent` (much smaller scope: no TUI, MCP, or extensions yet —
see the milestone list).

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
- **Lifecycle events** for the UI via `listener`: `session_opened` (with the
  `new`/`fork`/`clone`/`switch` reason), `compaction_start`/`compaction_end`,
  `overflow_retry`, `overflow_give_up`, `session_tree`, `session_info_changed`.

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
- Slash commands: `/help`, `/new`, `/compact [focus]`, `/tree [--summarize]
  [id]`, `/fork [n|<id>]`, `/clone`, `/sessions`, `/resume <n|id>`,
  `/name [text]`, `/session`, `/templates`, `/skills`, `/quit` (the session
  navigation set arrived in M7); `/<template> args...` invokes a prompt
  template from `.karen/prompts` (project) or `~/.karen/prompts` (user), with
  `$1`/`$@`/`${@:N:L}` substitution, or a skill from `.karen/skills` /
  `~/.karen/skills` (see M6). Built-in command names win over templates and
  skills of the same name, like pi's built-in slash commands.
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
- `get_entries` answers the session's entries (with `since` slicing after a
  named entry); M7 turned the earlier id-only shape into pi's entry objects.
- Not ported: the thinking-level commands, auto-retry, pi's bash side
  channel, export-html, `get_commands` and the extension UI sub-protocol
  (karen has no extensions or TUI yet), and image inputs on
  `prompt`/`steer`/`follow_up`. Fork/clone/switch/session-stats arrived in M7
  (see below).

Verified against real DeepSeek over real stdio: header → `prompt` → full event
stream → `get_last_assistant_text` = `"RPC-OK"`, model switching,
`new_session` rebinding (second header), unknown-command/parse-error
responses, and a clean exit 0 on EOF.

## M6: structured system prompt, context files and skills

`src/karen_coding_agent/prompt.py` ports pi's `core/system-prompt.ts`, and
`src/karen_coding_agent/resources.py` the context-file/skill discovery of
`core/resource-loader.ts`. The session's system message now carries pi's
structured sections (`SystemMessage.sections`, `content` empty) instead of one
flat string, so request rendering matches pi's transcript exactly:

| section | source |
| --- | --- |
| `preamble` (untagged) | the default karen preamble, or `SYSTEM.md` |
| `tools` | one line per selected tool that has a snippet (pi's `promptSnippet`s) |
| `rules` | shell-fallback rule, per-tool guidelines, then "Be concise…"/"Show file paths…" (deduplicated, order kept) |
| `addendum` | `APPEND_SYSTEM.md` |
| `project_context` | `<project_instructions path="…">` per AGENTS.md/CLAUDE.md |
| `skills` | `<available_skills>` block (only with a read/bash tool, like pi) |
| `cwd` | the working directory, forward-slashed |

- **Context files**: `~/.karen/AGENTS.md` first, then every ancestor of the
  cwd outermost-first; per directory the first of
  `AGENTS.override.md`, `AGENTS.md`, `AGENTS.MD`, `CLAUDE.md`, `CLAUDE.MD`
  wins. Paths are deduplicated by real path and read as UTF-8 with the BOM
  stripped. Deviations: the git-worktree "shadowed context file" rule is not
  ported.
- **`SYSTEM.md` / `APPEND_SYSTEM.md`**: project (`<cwd>/.karen/`) wins over
  global (`~/.karen/`). pi gates the project files behind its project-trust
  prompt; karen has no trust system, so they always apply.
- **Skills**: `<cwd>/.karen/skills` and `~/.karen/skills` (SKILL.md, loaded by
  karen-agent's loader, diagnostics on stderr). Skills are advertised to the
  model in the `skills` section and can be invoked by name — `/token-skill
  please` expands to `format_skill_invocation(skill, "please")`; `/skills`
  lists them. Templates and skills share one `/name` namespace (templates
  win), like pi's slash commands.
- Not ported: pi's `docs` section (no karen docs tree), `forceSystemPrompt`
  (extension hook), the `PI_*` session-env guideline the shell tools
  contribute, and `/reload` (`/new` rebuilds the prompt).

Verified against real DeepSeek: an AGENTS.md rule and an `APPEND_SYSTEM.md`
rule both showed up in the reply, and `/token-skill please` made the model
answer with the skill's mandated token.

## M7: session navigation (tree, fork, clone, switch)

`src/karen_coding_agent/navigation.py` holds the pure helpers (pi's
`SessionManager.getTree` plus the fork-selector views) and `AgentSession`
gained the operations pi's `navigateTree`/runtime fork/clone/switch perform.

- **Tree** (`session_tree()`, `render_tree`): every entry becomes a node,
  children oldest-first, entries whose parent is not in the file become roots
  (pi's orphan rule). `/tree` renders it — `●` current tip, `○` other branch
  tips, `·` interior nodes, `[label]` for labelled entries. Lines show the
  id's **last** 8 characters (karen's uuid7 ids share a long timestamp
  prefix, so the head does not distinguish entries); ids resolve by full id,
  unique prefix or unique suffix.
- **`navigate_tree(targetId)`** moves the branch tip inside the same session
  file, exactly like pi: a no-op at the current tip; a user message moves the
  tip to its parent and returns the text as `editorText` for editing; any
  other entry becomes the tip. `summarize=True` condenses the abandoned
  branch (old tip back to the common ancestor) into a `branch_summary` entry
  placed at the *target* position, with `label` attaching a tree label to it.
  Context is rebuilt from the new ancestry; rewound entries stay in the file
  and reappear as siblings/orphans in the tree.
- **Fork / clone** (`fork()`, `clone()`): a fork copies the current branch
  into a new session file and rebinds to it, recording the source as the new
  header's `parent_session_id`; `position="before"` (pi's default) requires a
  user message and returns its text as `selectedText`, `position="at"` copies
  through the selected entry, and `clone()` is `at` on the tip. Deviation:
  only entries on the *current* branch can be forked (karen-agent's fork
  copies a branch path, pi's `createBranchedSession` copies any entry's
  ancestry), so `user_messages_for_forking()` lists the current branch's
  user messages — navigate to another branch first to fork off it.
- **Switch** (`list_sessions()`, `switch_session(metadata)`): opens another
  session file for the same cwd and rebinds; switching to the open session is
  a no-op. **`session_stats()`** ports pi's `getSessionStats` (user/
  assistant/tool-call/tool-result counts, token totals, cost) summed from the
  entries — a fork copies entries, not karen's usage rows — omitting pi's
  `contextUsage`.
- **Names and labels**: `set_session_name` / `session_name` (pi's
  `pi.session.name` value, empty names rejected) and `set_label`
  (`pi.entry.label`). Because karen's fork only copies *configured lanes*,
  `open()`/`_rebind()` now write the branch's `pi.lane.config`/`pi.lane.state`
  pair (model, thinking level, tool names) when they are missing — the same
  binding pi's runtime performs — leaving existing values untouched.
- **REPL**: `/tree [--summarize] [id]`, `/fork [n|<id>]` (lists the forkable
  user messages, then reprints the text to edit), `/clone`, `/sessions`,
  `/resume <n|id>`, `/name [text]`, `/session`.
- **RPC**: `get_tree`, `get_fork_messages`, `fork` (`entryId`, optional
  `position`), `clone`, `switch_session` (`sessionPath`, absolute path or
  bare id), `set_session_name`, `get_session_stats`; `get_entries` now
  returns the entries themselves and honours `since`. `session_tree` and
  `session_info_changed` are forwarded on the event stream, and every command
  that replaces the session re-emits the JSONL header line.
- Not ported: pi's `/import` (JSONL import), HTML export, and the interactive
  full-screen tree/fork selector (karen's REPL takes ids and indices); the new
  built-in command names shadow templates or skills of the same name.

Verified against real DeepSeek: a two-turn session rendered its tree; a fork
reproduced the first turn in a new file with `parent_session_id` set;
`/tree --summarize` had the model write a real branch summary that appeared
in the context as a `branchSummary` message; navigating onto a user message
returned its text and emptied the context; clone and switch round-tripped;
and the RPC surface answered `get_tree`/`fork`/`clone`/`switch_session`/
`get_session_stats` over real stdio with a header per session.

## Roadmap

Later milestones (tracked in the repo root README): extensions / MCP / TUI /
image input / auto-retry (unscheduled).

## Development

```bash
../.venv/Scripts/python.exe -m pip install -e ./karen-coding-agent --no-deps  # karen-ai/karen-agent are local
cd karen-coding-agent && ../.venv/Scripts/python.exe -m pytest -q
```

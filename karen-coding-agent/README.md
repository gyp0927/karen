# karen-coding-agent

The `karen` CLI coding assistant — the application layer on top of
[karen-agent](../karen-agent), and the karen equivalent of
[`@earendil-works/pi`](https://github.com/earendil-works/pi)'s
`packages/coding-agent` (smaller scope: no extensions; the TUI is hand-rolled
in pure stdlib rather than ported from pi's React/Ink framework).

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
  overflow keeps the failure and emits `overflow_give_up`. The rewind stops at
  the last entry that is *not* part of the attempt, so anything recorded
  alongside it (a thinking-level change, a label, a model switch) stays on the
  branch — pi's context edit unparents nothing. A silent overflow
  on a *successful* response (usage over the window) compacts without
  retrying. Aborted messages and messages from a different model are skipped,
  like pi.
- **Hooks**: a `HookRegistry` is bridged into the agent's
  `before_tool_call`/`after_tool_call`, and `before_compaction` can decline
  or replace any compaction.
- **Lifecycle events** for the UI via `listener`: `session_opened` (with the
  `new`/`fork`/`clone`/`switch` reason), `compaction_start`/`compaction_end`,
  `overflow_retry`, `overflow_give_up`, the `auto_retry_*` and
  `summarization_retry_*` events (M8), `session_tree`,
  `session_info_changed`.

Simplifications vs pi (documented at the port sites): no settings manager on
the session itself (the CLI loads the settings files, see M4), no session
projections, context edits, or extension events; pi's retention/staleness
guards are dropped because karen sessions are append-only and only fresh
post-run messages are checked.

### The `karen` CLI

```bash
karen                                       # interactive: alt-screen TUI (plain REPL when piped)
karen --repl                                # force the plain line-based REPL instead
karen --tui                                 # force the alt-screen TUI
karen -p "summarize this repo"              # headless text mode: final reply on stdout
karen "one" "two"                           # headless: prompts run sequentially
karen --mode json "prompt"                  # headless JSON event stream
karen --mode rpc                            # JSON command protocol on stdin/stdout (M5)
karen --export session.jsonl out.html       # session file -> standalone HTML report (M9)
printf 'hello\n/quit\n' | karen --new       # piped REPL (how the smokes drive it)
```

`--cwd PATH`, `--model ID` and `--new` apply to every interactive and headless
mode.

- Interactive mode streams replies live and prints tool calls
  (`[tool ->] write(path='a.py', ...)`) plus a context-token estimate after
  every turn.
- Interactive mode picks its front-end at startup (`tui/terminal.py`'s
  `supports_tty()`): the alt-screen **TUI when stdin *and* stdout are a
  terminal**, otherwise the plain REPL plus a one-line notice saying why. Both
  ends are required — the alt-screen escapes go to stdout, and the Windows key
  path (`msvcrt.getwch()`, which has no EOF path) needs a real *console*, so on
  Windows the gate asks `GetConsoleMode` rather than trusting `isatty()`: the
  NUL device is a character device and passes `isatty()`, and Git Bash (mintty)
  hands a native program pipes, not a console. Force either side with
  `--tui`/`--repl`, or persist it with `"tui": false` in a settings file;
  precedence is flag > settings > default.
- Headless modes (pi's `runPrintMode`; M3): **text** (default) prints only the
  final assistant message's text to stdout — tool chatter and session notices
  go to stderr — and exits 1 when the final message is an error or aborted;
  **json** (`--mode json`) emits the session header line followed by one JSON
  event per line (pi's `json-event.ts` shape: `message_update` carries only
  cumulative `usage` + the delta sub-event with `partial` stripped,
  `toolcall_start` gains `id`/`toolName`) and stays machine-parseable end to
  end. Deviations from pi: positional prompts imply print mode instead of
  becoming an interactive initial message, and `@file`/image arguments are not
  supported yet.
- Sessions live under `~/.karen/sessions` (override with
  `KAREN_SESSIONS_ROOT`), resumed per working directory; `/new` starts fresh.
- Slash commands: `/help`, `/new`, `/compact [focus]`, `/retry [on|off]`,
  `/tree [--summarize] [id]`, `/fork [n|<id>]`, `/clone`, `/sessions`,
  `/resume <n|id>`,
  `/name [text]`, `/session`, `/templates`, `/skills`, `/quit` (the session
  navigation set arrived in M7, `/retry` in M8, `/thinking`, `/export` and
  `/settings` in M9); `/<template> args...` invokes
  a prompt
  template from `.karen/prompts` (project) or `~/.karen/prompts` (user), with
  `$1`/`$@`/`${@:N:L}` substitution, or a skill from `.karen/skills` /
  `~/.karen/skills` (see M6). Built-in command names win over templates and
  skills of the same name, like pi's built-in slash commands.
- Ctrl-C aborts the running turn (`agent.abort()` + `wait_for_idle()`), and
  also cancels a retry backoff that is in progress (M8).

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
  "retry": { "enabled": true, "maxRetries": 3, "baseDelayMs": 2000, "maxAgentDelayMs": 60000 },
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
- `retry` (M8) is pi's coding-agent retry block — per-field fallbacks are
  `enabled: true`, `maxRetries: 3`, `baseDelayMs: 2000`,
  `maxAgentDelayMs: 60000`; wrong types fall back instead of failing.
- `~` is expanded in path settings. `tui` (a karen addition) keeps `karen` in
  the plain REPL when it is `false`; it is read at startup, like the model and
  shell settings. Writes (pi's `/settings` command) and the rest of pi's
  Settings (extensions, analytics, themes, `retry.provider`, …) are not ported.

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
  `set_auto_compaction`, `set_auto_retry`, `abort_retry`. Unknown commands and
  malformed lines are answered with `success:false` (`command:"parse"` for
  JSON errors) — the loop keeps running; stdin EOF exits 0.
- **`prompt` while a run is active** requires an explicit `streamingBehavior`
  (`"steer"` or `"followUp"`); karen answers `{"disposition":"queued"}` and
  routes the message accordingly, or fails with pi's
  `"Agent is already processing. Specify streamingBehavior ('steer' or
  'followUp') to queue the message."` when it is missing (see M9 — the earlier
  implicit steer is gone). When idle it answers `{"disposition":"started"}` and
  the run's progress arrives on the event stream. Deviation: karen reports the
  disposition at command time, where pi resolves its response after preflight.
  A spawned prompt task counts as busy from the moment it is created (before it
  has run a step), so a burst of buffered prompts is never answered `started`
  and then dropped: every one after the first gets the busy error, or is queued
  when it names a `streamingBehavior`.
- **`new_session`** opens a brand-new session for the cwd (closing the old
  one) and emits the new header line; pi's `parentSession` (fork) is not
  supported. `get_state` reports model, thinkingLevel, isStreaming,
  isCompacting, steeringMode/followUpMode, sessionId/sessionFile,
  autoCompactionEnabled, messageCount and pendingMessageCount.
- **`set_auto_compaction`** gates the post-run threshold check and the
  overflow recovery; auto-retry (M8) keeps running, like pi's independent
  flags. Manual `compact` answers `{"compacted": bool}`.
- `get_entries` answers the session's entries (with `since` slicing after a
  named entry); M7 turned the earlier id-only shape into pi's entry objects.
- Not ported at M5: the thinking-level commands, pi's bash side channel,
  export-html, `get_commands` and the extension UI sub-protocol (karen has no
  extensions or TUI yet), and image inputs on `prompt`/`steer`/`follow_up`.
  Fork/clone/switch/session-stats arrived in M7, retry control in M8, and the
  rest of that list in M9 (below) — only `get_commands` and the extension UI
  sub-protocol remain.

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
- Not ported: pi's `/import` (JSONL import; HTML export and the interactive
  full-screen tree/fork selector arrived later — see M9 and the roadmap). The
  new built-in command names shadow templates or skills of the same name.

Verified against real DeepSeek: a two-turn session rendered its tree; a fork
reproduced the first turn in a new file with `parent_session_id` set;
`/tree --summarize` had the model write a real branch summary that appeared
in the context as a `branchSummary` message; navigating onto a user message
returned its text and emptied the context; clone and switch round-tripped;
and the RPC surface answered `get_tree`/`fork`/`clone`/`switch_session`/
`get_session_stats` over real stdio with a header per session.

## M8: auto-retry

A transient provider failure (overloaded, rate limit, 5xx, dropped
connection, …) no longer ends the turn: the failed attempt is dropped, the
backoff is awaited, and the turn is re-run — pi's `AgentSession` auto-retry.

- **karen-ai** gained the assistant-call layer of pi-ai's `utils/retry.ts`
  next to the existing provider-request retry: `RetryPolicy`
  (`enabled`/`maxRetries`/`baseDelayMs`/`maxAgentDelayMs`), `retry_delay_ms`
  (`baseDelayMs * 2^(attempt-1)`, capped at 60s), `RetryCallbacks`
  (`onRetryScheduled`/`onRetryAttemptStart`/`onRetryFinished`, sync or
  async), `retry_assistant_call(produce, policy, signal=…, callbacks=…)` and
  `is_retryable_assistant_error(message)` with pi's verbatim pattern lists —
  account/billing/quota limits (`insufficient_quota`, "Monthly usage limit
  reached", `GoUsageLimitError`, …) are **not** retried, so deterministic
  failures still fail fast. An abort during the backoff is normalized to an
  aborted message with the error message stripped.
- **karen-agent**: `compact()`, `generate_summary*()` and
  `GenerateBranchSummaryOptions` take `retry`/`callbacks` and run every
  summary request through `retry_assistant_call` (pi's
  `completeSimpleWithRetries`), so one dropped stream no longer loses a whole
  compaction. `RetryPolicy`/`DEFAULT_MAX_AGENT_RETRY_DELAY_MS` are now
  karen-ai's (karen-agent's `config.py` re-exports them).
- **AgentSession**: after each run the last assistant message is classified
  (context overflow is excluded — compaction owns that); a retryable failure
  emits `auto_retry_start {attempt, maxAttempts, delayMs, errorMessage}`,
  durably omits the failed attempt (same branch-tip rewind as overflow
  recovery), sleeps the backoff, then re-runs the turn. A later successful
  response emits `auto_retry_end {success: true, attempt}`; an exhausted
  budget keeps the failure and emits `{success: false, attempt, finalError}`.
  `abort_retry()` (also called by `abort()`, so Ctrl-C works during a
  backoff) cancels the sleep and emits `finalError: "Retry cancelled"`;
  `is_retrying`, `auto_retry_enabled` and `set_auto_retry_enabled()` expose
  the state. Forwarded `agent_end` events carry pi's `willRetry` flag, so a
  run that is about to be retried is not reported as a failure.
- **Summaries report their own retries** via
  `summarization_retry_scheduled`/`summarization_retry_attempt_start` (with
  `source: "compaction" | "branchSummary"`)/`summarization_retry_finished`.
- **REPL**: failures print `[retrying (attempt n/max) in Xs: …]` (stderr in
  headless mode), successful retries `[retry succeeded on attempt n]`, and
  `/retry [on|off]` shows or toggles the policy. `--mode json` prints the
  retry events; **RPC** forwards them and answers `set_auto_retry` /
  `abort_retry`.
- **Settings**: the `retry` block (pi's coding-agent defaults: enabled, 3
  retries, 2s base delay, 60s cap) configures the session's policy.
- Not ported: pi's per-request `retry.provider` knobs (karen-ai adapters take
  those per request), the TUI retry indicator, and `_runAutoCompaction`'s
  `willRetry` reshuffling (karen's overflow recovery is a separate path).

Verified against real DeepSeek with injected transient errors ("Error 503
Service Unavailable"): a failing turn was retried and answered by the real
model with the failed attempt gone from the transcript; a manual compaction
whose summary call failed once produced `summarization_retry_*` events and a
real summary; the next turn answered from that summary; and a 30s backoff was
cancelled by `abort_retry` with `finalError: "Retry cancelled"`. A real
project `settings.json` drove `/retry` (maxRetries 5, baseDelayMs 250).

## M9: images, export, bash side channel, thinking levels, settings writes

CA-M9 closes out the non-extension app-layer surface pi's coding-agent has
that karen had not ported yet.

- **Image input** (`utils/image_process.py`, `utils/exif_orientation.py`,
  `utils/tool_result_images.py`, plus the `read` tool's image processor): pi's
  `utils/image-*.ts`, with Pillow standing in for the photon Rust/WASM module
  and `asyncio.to_thread` for pi's worker thread (Pillow releases the GIL
  around decode/resize/encode). `processImage` normalizes to a supported
  inline format (PNG/JPEG/GIF/WebP; anything else — including BMP — is
  converted to PNG or omitted with pi's exact hint strings), `resizeImage`
  applies EXIF orientation then downscales to fit 2000×2000 and a 4.5MB base64
  ceiling (pi's `Math.round` semantics, PNG-then-JPEG candidate order,
  `[q, 85, 70, 55, 40]` quality ladder, 0.75 shrink loop), and
  `formatDimensionNote` tells the model the coordinate mapping. The **prompt
  path** (`AgentSession._normalize_prompt_images`, pi's `_normalizePromptImages`)
  resizes each image to the current model's `input_limits.images.resize`
  profile, drops failures into text hints, and appends the hints to the user
  text; `steer()`/`follow_up()` keep images raw, exactly like pi. The
  **tool-result path** (`AgentSession._after_tool_call`) normalizes images
  tools produced *after* the hooks run (pi's ordering), keeping the original
  block on failure (unlike the prompt path). The **read tool** now wires its
  `image_processor` hook so `read` on an image returns a resized `ImageContent`
  (BMP goes through PNG conversion like pi instead of an omission note).
  `images.autoResize` (default true) skips the resize path entirely when off, and
  `images.blockImages` (default false) strips images from the LLM request in a
  `convert_to_llm` wrapper — pi's `convertToLlmWithBlockImages`, including its
  "Image reading is disabled." placeholder and consecutive-placeholder dedupe.
  Both are read live (a `/settings` write applies to the running session) and at
  session open from the merged settings files. An EXIF transpose failure aborts
  the resize (pi folds decode + EXIF into one try/catch that yields null) rather
  than sending a silently unrotated image.
- **Session export** (`session_export.py`): `export_to_jsonl` writes the
  current branch in pi's importable v3 wire format — a
  `{"type":"session","version":3,...}` header then the branch re-parented into
  one linear chain — so pi can re-open it; `export_to_html` writes one
  self-contained `.html` report: the session data is base64-embedded, only the
  `leafId` branch path is included (pi's `getPath`), and the template renders it
  with its own markdown-subset renderer — no CDN scripts, so the report opens
  offline and makes no network requests. pi inlines vendored
  marked/highlight.js instead; karen trades syntax highlighting for not shipping
  a 165KB JS blob. The template's renderer handles `bashExecution` messages
  (`$ command`, output, `(exit n)`/`(cancelled)`/truncated notes) and extracts
  fenced code the way CommonMark reads it — a fence opens only at the start of a
  line and an unterminated one runs to the end — so a mid-line triple-backtick
  can neither swallow text nor leak a placeholder. The base64 payload is decoded
  as UTF-8 (`atob` alone yields Latin-1, which mojibakes every non-ASCII
  string), matching pi's `TextDecoder` step. pi sources the export theme from its
  TUI theme system; karen ships a neutral dark default until the TUI milestone
  lands.
- **Bash side channel** (`bash_executor.py`): `AgentSession.execute_bash` /
  `abort_bash` run a shell command outside the agent loop (pi's
  `core/bash-executor.ts`), streaming sanitized output to a callback and
  `bash_execution_update` events (each carrying the command id and the fresh
  chunk — pi's `onChunk`, so appending the deltas reproduces the output rather
  than repeating it), spilling
  oversized output to a temp file, and reporting pi's `BashResult`
  (`output`/`exitCode`/`cancelled`/`truncated`/`fullOutputPath`). The result is
  recorded in the transcript as a `BashExecutionMessage` (pi's
  `recordBashResult`, so the model sees it next turn; `excludeFromContext` keeps
  it out of the LLM conversion), queued while a run is in flight so
  tool_use/tool_result ordering can't break and appended once the run settles.
  Built on karen-agent's
  `run_shell_command` + `OutputCapture`.
- **Thinking levels**: `set_thinking_level` (clamped to the model's
  capabilities via karen-ai's `clamp_thinking_level`, emits
  `thinking_level_changed` and appends a `thinking_level_change` session entry
  on change), `cycle_thinking_level`, `get_available_thinking_levels`,
  `supports_thinking`. A resumed session restores the last recorded level on its
  branch (pi's `getSessionContextSettings`). Deviation: pi also seeds that entry
  into every new session from its `defaultThinkingLevel` setting; karen has no
  such setting, so it records actual changes only — no metadata node at the root
  of every tree.
- **RPC**: the new commands `set_thinking_level`, `cycle_thinking_level`,
  `get_available_thinking_levels`, `bash`, `abort_bash`, `export_html`;
  `images` (`[{data, mimeType}]`) is accepted on `prompt`/`steer`/`follow_up`;
  `thinking_level_changed`/`bash_execution_update` are forwarded; `get_state`
  now reports the real `sessionFile` path and `sessionName` (it previously
  echoed the session id as `sessionFile`). `prompt` honors `streamingBehavior`:
  while a run is in flight it is required (pi's exact error) and routes the
  message to `steer` or `followUp`. `/new` rebinding carries the image and retry
  settings across.
- **Settings writes** (`settings.update_settings`, behind `/settings`): merge
  camelCase keys back into the global or project file atomically (nested dicts
  deep-merge, unknown keys kept), pi's write path minus its lock file and
  modified-field bookkeeping (karen runs single-process). A file that exists but
  fails to parse is never overwritten (pi's `save()` returns early on a load
  error) — `update_settings` raises instead, and the CLI reports it.
- **CLI**: `/thinking [level|cycle]`, `/export [jsonl|html] [path]`,
  `/settings [show | global|project key=value ...]`, and pi's
  `karen --export FILE [OUT.html]` (an existing session file — the native
  storage log or a v3 export — to a standalone report, no model needed).
  Handler failures (unwritable export path, malformed settings file) print
  `[error: ...]` and leave the REPL running. A `/settings` write re-reads the
  file into the CLI's own settings snapshot (pi reads its settings manager on
  every use), so the next `/new` — and the tool set it rebuilds — sees it
  instead of silently reverting to the startup values.

Two parity details pi splits across two call sites, both now matched: the
pending bash queue is flushed **when the run settles and again before a new
prompt** (pi `agent-session.ts` flushes in the run's `finally` — after
auto-retry and overflow recovery are through, *not* at each attempt's
`agent_end` — and in `prompt()`), so a result recorded after the settle flush
still reaches the model in the next turn rather than after it, and a result
recorded during an attempt that gets retried is not cast away with that
attempt; and `switch_session`/`fork`/`clone` re-read the branch's
last `thinking_level_change` (pi rebuilds its AgentSession on every switch,
restoring the level in the constructor) instead of leaving the outgoing
session's level in place. The RPC prompt path also treats a spawned-but-unstarted
prompt task as busy: `create_task` does not yield, so a burst of buffered
prompts used to answer every one of them `disposition: started` while all but
the first died in a swallowed `RuntimeError`.

Verified with 57 new offline tests (`tests/test_m9.py`, including a node-driven
render of the HTML report against a DOM stub) plus the full suite (260 passed)
and karen-agent/karen-ai suites (332 / 443+1). New runtime dependency: Pillow.

## TUI: alt-screen interactive mode

`karen` runs the interactive session inside a hand-rolled alternate-screen
viewport (and `--tui` forces it when auto-detection says no), pi's
`modes/interactive/` minus the React/Ink framework pi builds on
(its `@earendil-works/pi-tui` is ~1.3MB and not portable, so this is pure
stdlib). The layout mirrors pi's chat viewport (`chat-viewport.ts`): a
scrollable transcript on top, a fixed input dock below.

```
❯ what is in this repo?

Looking around:▌

[tool <-] ls
    path='.'

It looks like a Python monorepo with packages karen-ai, karen-agent,
karen-coding-agent and karen-mcp.

! python -m pytest -q
[bash <-] python -m pytest -q
    ................................
    1 failed, 371 passed in 32.8s
    exit 1

                                model: deepseek/deepseek-v4-pro | tools: 8 | cwd: E:\karen
> ▌
enter: send  ctrl+j: newline  up/down: scroll  ctrl+c: quit  !cmd: shell
```

The package (`src/karen_coding_agent/tui/`) is split so the logic is testable
without a TTY:

| module | role |
| --- | --- |
| `transcript.py` | pure model: agent/session events → user / assistant / tool / bash / notice messages |
| `layout.py` | pure renderer: messages + terminal size → exactly `height` ANSI lines |
| `terminal.py` | the only TTY-touching module: alt screen (`CSI ?1049h/l`), raw mode, key decoding |
| `app.py` | `TuiApp` (state + editor) and `LiveTui` (the async driver) |

- **Alt screen**: `--tui` enters `CSI ?1049h`, hides the cursor, and restores
  both on exit, so the shell's scrollback is untouched. Raw mode is enabled on
  entry and restored on exit — POSIX via `termios` cbreak, Windows via
  `SetConsoleMode` (raw console input plus `ENABLE_VIRTUAL_TERMINAL_PROCESSING`
  so the VT escapes are interpreted rather than printed).
- **Responsive while streaming**: the key loop runs in an executor thread
  (`asyncio.to_thread`) and each prompt runs as a task, so the viewport keeps
  repainting, scrolling and accepting keys while the model streams. A prompt
  submitted while a run is in flight is queued and sent as soon as that run
  finishes (the status/dock shows it), rather than racing a second run against
  the same session.
- **Editor**: multi-line, with `ctrl+j` inserting a newline and `enter`
  submitting a single-line buffer (or inserting a newline once it is
  multi-line); `ctrl+enter` always submits. Cursor movement (`left`/`right`/
  `ctrl+a`/`ctrl+e`), `backspace` (joining lines at a line start), `ctrl+k`
  (kill to end of line), `ctrl+u` (clear to line start), `tab`, and
  `up`/`down` scrolling of the transcript.
- **Event-driven**, off the same stream the plain REPL prints: `text_delta`s
  accumulate into one streaming assistant block with a cursor, tool calls
  render as `[tool ->] name` / `[tool <-] name` (matched by `tool_call_id`, so
  parallel executions of the same tool can't cross-attribute), a failed run
  attaches its error to that run's block, and every session notice
  (`session_opened`, compaction, overflow, `auto_retry_*`,
  `summarization_retry_*`) lands in the transcript. Nothing prints to
  stdout/stderr while the alt screen is live — slash-command output is captured
  and reposted as notice lines.
- **Slash commands** work in the TUI through the same router the REPL uses:
  `/new`, `/compact`, `/retry`, `/tree`, `/fork`, `/clone`, `/sessions`,
  `/resume`, `/name`, `/session`, `/thinking`, `/export`, `/settings`,
  `/templates`, `/skills`, `/help`, `/quit`, and `/<template>`/`<skill>`
  invocation.
- **Shell bypass** (`!command`, pi's `!` — TUI-only, the plain REPL has no
  equivalent): the command runs through `AgentSession.execute_bash`, so it stays
  outside the agent loop, streams into the transcript as a `[bash ->]` block
  that becomes `[bash <-]` when it finishes, and is recorded in the session so
  the model sees the command and its output on the next turn
  (`recordBashResult`). A non-zero exit adds `exit N`; an executor spill adds
  `output truncated (full output: …)`. The live view is bounded on three axes so
  a flooding command cannot stall the frame loop: 400 retained lines, 1024
  characters per retained line (a `base64 -w0` stream or a `\r`-redrawn progress
  bar has no line breaks at all, and `textwrap` on a whitespace-free megabyte is
  quadratic), and a 50 ms repaint throttle. Whatever falls off is reported: a
  dropped-lines marker, or a leading `…` on a truncated line.
- **A submitted line that is a command never stays `(pending)` — and never
  costs the answer it interrupts its block**: slash-command lines are resolved
  on the spot (they never become a run, so no `agent_start` will clear them),
  and a `!command` line is not echoed as a user message at all — the bash block
  already carries the command. The in-flight run's assistant block is parked
  while that line is in doubt and handed back when it turns out not to be a
  run, so the answer keeps streaming into one block instead of splitting in two
  with the first half holding a cursor nothing would clear.
- **Ctrl+C takes the running command with it**: a `!command` still in flight
  when the TUI quits is killed at the process-tree level by
  `run_shell_command`'s own unwind, so no shell (and no child of it) keeps
  writing files behind a dead app.
- **Layout accounting** is exact: the dock is rendered and measured first, the
  transcript fills the remaining rows, and `render` always returns exactly
  `height` lines (a tall dock on a tiny terminal keeps the footer by trimming
  from the top). Agent messages start pending and clear when the run starts, so
  an interrupted or failed prompt reads `(aborted)` / `(error)` instead of a
  bare line.
- Not ported: pi's session-tree/fork *selectors* (karen's are text commands),
  the theme system, search-in-transcript, mouse selection/copy, widgets, and
  pi's incremental (damage-region) painting — karen repaints whole frames,
  which is what makes the renderer testable without a TTY.

Verified with 98 offline tests (`tests/test_tui.py`, `tests/test_tui_app.py`:
transcript lifecycle, the three bash-output bounds, layout row accounting, the
editor's key handling, CSI decoding, the TTY gate, and an end-to-end async
driver over a fake terminal including the `!command` path) plus the full suite;
a `--tui` run was also driven against a real terminal. The cancel path is
covered with a real shell — the child writes a marker file a second in, and
cancelling the caller must leave it unwritten (karen-agent's
`tests/test_local_shell.py`, and the same chain through `execute_bash`).

## Roadmap

Later milestones: extensions (unscheduled). MCP tool integration and the TUI
both landed (above/below).

## MCP: tools from external servers

`karen_coding_agent.mcp` connects to Model Context Protocol servers (stdio or
Streamable HTTP) via `karen_mcp`, wraps each of their tools as an `AgentTool`
the loop can call, and projects `tools/call` results into LLM content with
`to_llm_content` (text and images pass through; audio, resource links, and
binary resources become short placeholders, exactly as `karen_mcp` documents).

Configuration lives in the settings files' `mcpServers` block (global
`~/.karen/settings.json` and project `<cwd>/.karen/settings.json`), one entry
per server:

```json
{
  "mcpServers": {
    "filesystem": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"] },
    "remote": { "url": "https://mcp.example.com/mcp", "headers": { "Authorization": "Bearer …" } }
  }
}
```

The CLI builds an `McpToolManager`, connects every configured server, and
passes it to `AgentSession`, which merges the wrapped tools into the default
tool set (so a server's tool is callable by name and its result reaches the
model as text/images). A server that fails to connect or list its tools is
skipped with a stderr diagnostic rather than aborting the session. An HTTP
server that needs OAuth takes an `AuthProvider` (from
`karen_mcp.oauth.adapt_oauth_provider`) in place of a static header.

`mcp_server_configs_from_settings` turns the settings block into
`McpServerConfig` objects; `McpToolManager(configs)` does the rest.

## Development

```bash
../.venv/Scripts/python.exe -m pip install -e ./karen-coding-agent --no-deps  # karen-ai/karen-agent are local
cd karen-coding-agent && ../.venv/Scripts/python.exe -m pytest -q
```

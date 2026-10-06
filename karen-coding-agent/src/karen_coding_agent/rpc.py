"""RPC mode: embeddable headless operation over JSON lines (pi's
`modes/rpc/rpc-mode.ts`).

Protocol: commands are one JSON object per line on stdin; responses
(``type: "response"`` with `command`/`success`/optional `data`/`error`,
carrying the command's `id` when given) and agent events stream as one
JSON object per line on stdout — the same event shapes as `--mode json`,
plus the session header as the first line and compaction lifecycle
events as they happen.

Ported commands — the subset karen's AgentSession/Agent can serve:

    prompt, steer, follow_up, abort, clear_queue, new_session,
    get_state, set_model, set_steering_mode, set_follow_up_mode,
    get_available_models, get_messages, get_last_assistant_text,
    get_entries, compact, set_auto_compaction

Deviations from pi (all documented here and in the README):
- the thinking-level commands, auto-retry, bash, fork/clone/switch,
  session-stats/export-html, get_commands and the extension UI
  sub-protocol are out of scope (karen has no extensions/TUI yet and no
  bash-executor side channel).
- `images` on prompt/steer/follow_up are not accepted (karen's app layer
  does not wire image input yet).
- `get_entries` returns branch entry ids instead of pi's SessionEntry
  objects (karen's AgentSession emits no entry_appended events).
- `new_session` starts a brand-new session for the cwd; `parentSession`
  is not supported.
- `prompt` reports `started`/`queued` at command time (pi resolves the
  response after preflight); the authoritative outcome still arrives
  through the event stream.
- `set_auto_compaction` gates the threshold auto-compaction driver;
  overflow recovery always runs.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any, Dict, List, Optional

from karen_ai import TextContent, UserMessage
from karen_agent.messages import message_field
from karen_agent.session import BranchScan
from karen_agent.session.jsonl import to_jsonable
from karen_agent.types import AgentMessage

from .agent_session import AgentSession
from .json_events import to_json_event

_UNSET = object()


def _success(command_id: Optional[str], command: str, data: Any = _UNSET) -> Dict[str, Any]:
    if data is _UNSET:
        return {"id": command_id, "type": "response", "command": command, "success": True}
    return {"id": command_id, "type": "response", "command": command, "success": True, "data": data}


def _error(command_id: Optional[str], command: str, message: str) -> Dict[str, Any]:
    return {"id": command_id, "type": "response", "command": command, "success": False, "error": message}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _queue_mode(value: Any) -> str:
    if value not in ("all", "one-at-a-time"):
        raise ValueError(f"Invalid queue mode: {value!r} (expected 'all' or 'one-at-a-time')")
    return value


def _user_message(text: str) -> UserMessage:
    """Same shape `Agent._normalize_prompt_input` builds for a string prompt."""
    return UserMessage(content=[TextContent(text=text)], timestamp=_now_ms())


def _message_text(message: AgentMessage) -> str:
    return "".join(
        block.text
        for block in message.content
        if message_field(block, "type") == "text" and message_field(block, "text")
    )


def _last_assistant_text(messages: List[AgentMessage]) -> Optional[str]:
    for message in reversed(messages):
        if message_field(message, "role") != "assistant":
            continue
        text = _message_text(message)
        return text or None
    return None


class RpcSession:
    """Session ownership plus `new_session` rebinding for the RPC loop.

    The starting `agent_session` is wrapped; `new_session` rebinds the
    holder to a fresh AgentSession on the same cwd (like pi's
    runtimeHost rebind), closing the old one. The JSONL header of each
    (re)opened session is emitted by the RpcServer (see `_emit_header`).
    """

    def __init__(self, agent_session: AgentSession, *, cwd: Optional[str] = None) -> None:
        self.agent_session = agent_session
        self.cwd = cwd or agent_session.cwd

    async def new_session(self) -> AgentSession:
        """Create a brand-new on-disk session and rebind to it."""
        old = self.agent_session
        if old.session is not None:
            await old.close()
        session = AgentSession(
            cwd=self.cwd,
            models=old.models,
            model=old.model,
            sessions_root=old.sessions_root,
            fresh=True,
            branch_name=old.branch_name,
            system_prompt=old.system_prompt_text,
            system_prompt_sections=old.system_prompt_sections,
            tools=old.tools,
            hooks=old.hooks,
            compaction_settings=old.settings,
            stream_fn=old._stream_fn,
        )
        await session.open()
        self.agent_session = session
        return session


class RpcServer:
    """Runs the protocol over given input lines and an output sink.

    `run()` reads one line at a time from `input_iter` (async or sync
    iterable of str) and writes responses/events through `emit(line)`.
    A `prompt` command may arrive while another is running: it is queued
    (steered into the active run), reported `queued`, and the event
    stream carries the outcome — pi resolves its response only after
    preflight, karen does not.
    """

    def __init__(self, holder: RpcSession, *, input_iter, emit, on_command_error=None) -> None:
        self.holder = holder
        self.input_iter = input_iter
        self.emit = emit
        self.on_command_error = on_command_error
        self.session = holder.agent_session
        self.auto_compaction_enabled = True
        self.is_compacting = False
        self._prompt_task: Optional[asyncio.Task] = None
        self._agent_unsub: Optional[Any] = None
        self._previous_listener = None

    # -- plumbing ---------------------------------------------------------------

    def _out(self, obj: Dict[str, Any]) -> None:
        self.emit(json.dumps(obj, ensure_ascii=False) + "\n")

    def _on_agent_event(self, event, signal) -> None:
        self._out(to_json_event(event))

    def _on_session_event(self, event) -> None:
        event_type = event.get("type")
        if event_type == "compaction_start":
            self.is_compacting = True
        elif event_type == "compaction_end":
            self.is_compacting = False
        if event_type in ("compaction_start", "compaction_end", "overflow_retry", "overflow_give_up"):
            self._out(event)

    def _emit_header(self) -> None:
        """Emit the current session's JSONL header (pi's json-mode first line).

        Emitted by the server rather than driven by `session_opened`: the
        session is opened by the caller before the server subscribes, and
        `new_session` rebinds after `open()` has already fired.
        """
        header = self.session.session_header()
        if header is not None:
            self._out(header)

    def _subscribe(self) -> None:
        if self._agent_unsub is not None:
            self._agent_unsub()
            self._agent_unsub = None
        if self._previous_listener is not None:
            self.session._listener = self._previous_listener  # type: ignore[misc]
        self._previous_listener = self.session._listener
        self.session._listener = self._on_session_event  # type: ignore[misc]
        self._agent_unsub = self.session.subscribe(self._on_agent_event)

    # -- commands ---------------------------------------------------------------

    async def handle_command(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch one command, emit its response, and return it.

        The return value is the same object that went out on the wire, so
        embedded callers (and tests) can drive the server without parsing
        its output stream.
        """
        command_id = raw.get("id")
        command_type = raw.get("type")
        try:
            response = await self._dispatch(command_id, command_type, raw)
        except Exception as error:
            response = _error(command_id, str(command_type), str(error))
        self._out(response)
        return response

    async def _dispatch(self, command_id, command_type, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if command_type == "prompt":
            return await self._cmd_prompt(command_id, raw)
        if command_type == "steer":
            self.session.steer(_user_message(str(raw.get("message", ""))))
            return _success(command_id, "steer", {"disposition": "queued"})
        if command_type == "follow_up":
            self.session.follow_up(_user_message(str(raw.get("message", ""))))
            return _success(command_id, "follow_up", {"disposition": "queued"})
        if command_type == "abort":
            self.session.abort()
            return _success(command_id, "abort")
        if command_type == "clear_queue":
            agent = self.session.agent
            drained_steering = [m for m in agent._steering_queue.drain()]
            drained_follow = [m for m in agent._follow_up_queue.drain()]
            return _success(
                command_id,
                "clear_queue",
                {
                    "steering": [_message_text(m) for m in drained_steering],
                    "followUp": [_message_text(m) for m in drained_follow],
                },
            )
        if command_type == "new_session":
            self.session = await self.holder.new_session()
            self._subscribe()
            self._emit_header()
            return _success(command_id, "new_session", {"cancelled": False})
        if command_type == "get_state":
            agent = self.session.agent
            state = agent.state
            header = self.session.session_header()
            return _success(
                command_id,
                "get_state",
                {
                    "model": to_jsonable(self.session.model),
                    "thinkingLevel": state.thinking_level,
                    "isStreaming": state.is_streaming,
                    "isCompacting": self.is_compacting,
                    "steeringMode": agent.steering_mode,
                    "followUpMode": agent.follow_up_mode,
                    "sessionId": self.session.session.metadata.id,
                    "sessionFile": (header or {}).get("id"),
                    "autoCompactionEnabled": self.auto_compaction_enabled,
                    "messageCount": len(state.messages),
                    "pendingMessageCount": len(agent.peek_queued_messages()),
                },
            )
        if command_type == "set_model":
            provider = raw.get("provider")
            model_id = raw.get("modelId")
            model = self.session.models.get_model(provider, model_id)
            if model is None:
                return _error(command_id, "set_model", f"Model not found: {provider}/{model_id}")
            self.session.set_model(model)
            return _success(command_id, "set_model", to_jsonable(model))
        if command_type == "set_steering_mode":
            self.session.agent.steering_mode = _queue_mode(raw.get("mode"))
            return _success(command_id, "set_steering_mode")
        if command_type == "set_follow_up_mode":
            self.session.agent.follow_up_mode = _queue_mode(raw.get("mode"))
            return _success(command_id, "set_follow_up_mode")
        if command_type == "get_available_models":
            available = await self.session.models.get_available()
            return _success(
                command_id,
                "get_available_models",
                {"models": [to_jsonable(model) for model in available]},
            )
        if command_type == "get_messages":
            messages = self.session.agent.state.messages
            return _success(command_id, "get_messages", {"messages": [to_jsonable(m) for m in messages]})
        if command_type == "get_last_assistant_text":
            text = _last_assistant_text(self.session.agent.state.messages)
            return _success(command_id, "get_last_assistant_text", {"text": text})
        if command_type == "get_entries":
            branch = await self.session.session.branch(self.session.branch_name)
            entries = await branch.find_entries(BranchScan(order="oldestFirst"))
            ids = [entry.id for entry in entries if entry.type in ("message", "compaction")]
            tip = await branch.get_tip_id()
            return _success(command_id, "get_entries", {"entries": ids, "leafId": tip})
        if command_type == "compact":
            compacted = await self.session.run_compaction(
                "manual", custom_instructions=raw.get("customInstructions")
            )
            return _success(command_id, "compact", {"compacted": compacted})
        if command_type == "set_auto_compaction":
            self.auto_compaction_enabled = bool(raw.get("enabled"))
            return _success(command_id, "set_auto_compaction")
        return _error(command_id, str(command_type), f"Unknown command: {command_type}")

    async def _cmd_prompt(self, command_id: Optional[str], raw: Dict[str, Any]) -> Dict[str, Any]:
        text = str(raw.get("message", ""))
        busy = self.session.agent._active_run is not None
        if not busy:
            self._prompt_task = asyncio.get_running_loop().create_task(self._run_prompt(text))
            return _success(command_id, "prompt", {"disposition": "started"})
        self.session.steer(_user_message(text))
        return _success(command_id, "prompt", {"disposition": "queued"})

    async def wait_for_idle(self) -> None:
        """Wait for the in-flight prompt task, if any (for embedders and tests)."""
        task = self._prompt_task
        if task is None:
            await self.session.wait_for_idle()
            return
        try:
            await task
        finally:
            if self._prompt_task is task:
                self._prompt_task = None

    async def _run_prompt(self, text: str) -> None:
        try:
            await self.session.prompt(text, auto_compact=self.auto_compaction_enabled)
        except Exception as error:  # prompt failures surface through events; log elsewhere
            if self.on_command_error is not None:
                self.on_command_error(error)

    # -- run loop ---------------------------------------------------------------

    async def run(self) -> int:
        self._subscribe()
        self._emit_header()
        input_queue: asyncio.Queue = asyncio.Queue()

        async def feed() -> None:
            try:
                if hasattr(self.input_iter, "__aiter__"):
                    async for line in self.input_iter:
                        input_queue.put_nowait(line)
                else:
                    for line in self.input_iter:
                        input_queue.put_nowait(line)
            finally:
                input_queue.put_nowait(None)

        feeder = asyncio.get_running_loop().create_task(feed())
        while True:
            line = await input_queue.get()
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as parse_error:
                self._out(_error(None, "parse", f"Failed to parse command: {parse_error}"))
                continue
            if not isinstance(raw, dict) or "type" not in raw:
                self._out(_error(None, "parse", "Command must be a JSON object with a `type`"))
                continue
            await self.handle_command(raw)
        await feeder
        return 0


async def run_rpc_mode(
    session: AgentSession,
    input_iter=None,
    emit=None,
    *,
    stdin=None,
    stdout=None,
    on_command_error=None,
) -> int:
    """Run the RPC loop until stdin closes.

    `input_iter` (an async or sync iterable of lines) and `emit(line)`
    are injectable for tests; by default lines are read from `stdin`
    (blocking reads on an executor, like pi's attachJsonlLineReader)
    and each emitted object is flushed as one JSON line to `stdout`.
    """
    holder = RpcSession(session)
    server = RpcServer(
        holder,
        input_iter=input_iter or _AsyncStdinLines(stdin or sys.stdin),
        emit=emit or _StdoutEmitter(stdout or sys.stdout),
        on_command_error=on_command_error,
    )
    return await server.run()


class _AsyncStdinLines:
    """Reads physical stdin line by line (pi's attachJsonlLineReader)."""

    def __init__(self, stdin) -> None:
        self._stdin = stdin

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        loop = asyncio.get_running_loop()
        line = await loop.run_in_executor(None, self._stdin.readline)
        if line == "":
            raise StopAsyncIteration
        return line.rstrip("\r\n")


class _StdoutEmitter:
    def __init__(self, stdout) -> None:
        self._stdout = stdout

    def __call__(self, line: str) -> None:
        print(line, end="", file=self._stdout, flush=True)


__all__ = ["RpcServer", "RpcSession", "run_rpc_mode"]

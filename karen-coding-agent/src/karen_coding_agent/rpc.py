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
    get_entries, get_tree, get_fork_messages, fork, clone,
    switch_session, set_session_name, get_session_stats,
    compact, set_auto_compaction, set_auto_retry, abort_retry,
    set_thinking_level, cycle_thinking_level, get_available_thinking_levels,
    bash, abort_bash, export_html

`images` (a list of `{data, mimeType}`) is accepted on prompt/steer/follow_up;
prompt normalizes them (convert/resize, failures dropped into text hints),
steer/follow_up queue them raw — pi's exact split.

Deviations from pi (all documented here and in the README):
- get_commands and the extension UI sub-protocol are out of scope (karen has
  no extensions/TUI yet).
- `get_entries` returns karen's session entries (ids, types and message
  payloads) rather than pi's `SessionEntry` objects; `since` slices after
  the named entry, like pi.
- `get_session_stats` omits pi's `contextUsage` object.
- `fork` accepts an optional `position` ("before", the default, or "at"),
  so clone-like forks over other entry types work; `clone` is the same
  operation pinned to the current tip.
- `switch_session` matches `sessionPath` against this cwd's session files
  by absolute path, and also accepts a bare session id.
- `new_session` starts a brand-new session for the cwd; `parentSession`
  is not supported.
- `prompt` reports `started`/`queued` at command time (pi resolves the
  response after preflight); the authoritative outcome still arrives
  through the event stream.
- `set_auto_compaction` gates the threshold auto-compaction driver and the
  overflow recovery; auto-retry keeps running (pi's flags are independent).
- `set_auto_retry` toggles the assistant-call retry policy, `abort_retry`
  cancels an in-progress retry backoff; both match pi's commands.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from karen_ai import ImageContent, TextContent, UserMessage
from karen_agent.messages import message_field
from karen_agent.session.jsonl import to_jsonable
from karen_agent.types import AgentMessage

from .agent_session import AgentSession
from .json_events import to_json_event
from .navigation import TreeNode

_UNSET = object()

#: Session lifecycle events the RPC stream forwards (pi forwards every session
#: event; karen's `session_opened` is covered by the JSONL header line).
_FORWARDED_SESSION_EVENTS = (
    "compaction_start",
    "compaction_end",
    "overflow_retry",
    "overflow_give_up",
    "auto_retry_start",
    "auto_retry_end",
    "summarization_retry_scheduled",
    "summarization_retry_attempt_start",
    "summarization_retry_finished",
    "session_tree",
    "session_info_changed",
    "thinking_level_changed",
    "bash_execution_update",
)


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


def _parse_images(raw: Any) -> Optional[List[ImageContent]]:
    """Validate an `images` payload ([{data, mimeType}, ...]) into ImageContent."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("images must be a list of {data, mimeType}")
    images: List[ImageContent] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("data"), str):
            raise ValueError("each image must be an object with a base64 `data` string")
        mime = item.get("mimeType") or item.get("mime_type")
        if not isinstance(mime, str):
            raise ValueError("each image must have a `mimeType` string")
        images.append(ImageContent(data=item["data"], mime_type=mime))
    return images


def _user_message(text: str, images: Optional[List[ImageContent]] = None) -> UserMessage:
    """Same shape `Agent._normalize_prompt_input` builds for a string prompt
    (pi queues steer/follow-up images raw, no normalization)."""
    content: List[Any] = [TextContent(text=text)]
    if images:
        content.extend(images)
    return UserMessage(content=content, timestamp=_now_ms())


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
            shell_path=old.shell_path,
            shell_command_prefix=old.shell_command_prefix,
            auto_resize_images=old.auto_resize_images,
            block_images=old.block_images,
            retry_policy=old.retry,
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
    A `prompt` command may arrive while another is running: it must carry a
    `streamingBehavior` ("steer" or "followUp"), is reported `queued`, and the
    event stream carries the outcome — pi resolves its response only after
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
        if event_type in _FORWARDED_SESSION_EVENTS:
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
            self.session.steer(_user_message(str(raw.get("message", "")), _parse_images(raw.get("images"))))
            return _success(command_id, "steer", {"disposition": "queued"})
        if command_type == "follow_up":
            self.session.follow_up(_user_message(str(raw.get("message", "")), _parse_images(raw.get("images"))))
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
            metadata = self.session.session.metadata
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
                    "sessionId": metadata.id,
                    "sessionFile": getattr(metadata, "path", None),
                    "sessionName": await self.session.session_name(),
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
            entries = await self.session.entries()
            since = raw.get("since")
            if since is not None:
                index = next((i for i, entry in enumerate(entries) if entry.id == since), -1)
                if index == -1:
                    return _error(command_id, "get_entries", f"Entry not found: {since}")
                entries = entries[index + 1 :]
            return _success(
                command_id,
                "get_entries",
                {
                    "entries": [to_jsonable(entry) for entry in entries],
                    "leafId": await self.session.branch_tip_id(),
                },
            )
        if command_type == "get_tree":
            tree = await self.session.session_tree()
            return _success(
                command_id,
                "get_tree",
                {
                    "tree": [self._tree_node(node) for node in tree],
                    "leafId": await self.session.branch_tip_id(),
                },
            )
        if command_type == "get_fork_messages":
            messages = await self.session.user_messages_for_forking()
            return _success(command_id, "get_fork_messages", {"messages": messages})
        if command_type == "fork":
            position = raw.get("position", "before")
            if position not in ("before", "at"):
                return _error(command_id, "fork", f"Invalid fork position: {position!r}")
            result = await self.session.fork(raw.get("entryId"), position=position)
            self._emit_header()
            return _success(
                command_id,
                "fork",
                {"text": result["selectedText"], "cancelled": result["cancelled"]},
            )
        if command_type == "clone":
            try:
                await self.session.clone()
            except ValueError as error:
                return _error(command_id, "clone", str(error))
            self._emit_header()
            return _success(command_id, "clone", {"cancelled": False})
        if command_type == "switch_session":
            session_ref = raw.get("sessionPath")
            if not session_ref:
                return _error(command_id, "switch_session", "sessionPath is required")
            metadata = await self._find_session(str(session_ref))
            if metadata is None:
                return _error(command_id, "switch_session", f"Session not found: {session_ref}")
            await self.session.switch_session(metadata)
            self._emit_header()
            return _success(command_id, "switch_session", {"cancelled": False})
        if command_type == "set_session_name":
            name = str(raw.get("name", "")).strip()
            if not name:
                return _error(command_id, "set_session_name", "Session name cannot be empty")
            await self.session.set_session_name(name)
            return _success(command_id, "set_session_name")
        if command_type == "get_session_stats":
            stats = await self.session.session_stats()
            return _success(command_id, "get_session_stats", stats)
        if command_type == "compact":
            compacted = await self.session.run_compaction(
                "manual", custom_instructions=raw.get("customInstructions")
            )
            return _success(command_id, "compact", {"compacted": compacted})
        if command_type == "set_auto_compaction":
            self.auto_compaction_enabled = bool(raw.get("enabled"))
            return _success(command_id, "set_auto_compaction")
        if command_type == "set_auto_retry":
            self.session.set_auto_retry_enabled(bool(raw.get("enabled")))
            return _success(command_id, "set_auto_retry")
        if command_type == "abort_retry":
            self.session.abort_retry()
            return _success(command_id, "abort_retry")
        if command_type == "set_thinking_level":
            await self.session.set_thinking_level(str(raw.get("level", "off")))
            return _success(command_id, "set_thinking_level")
        if command_type == "cycle_thinking_level":
            level = await self.session.cycle_thinking_level()
            if level is None:
                return _success(command_id, "cycle_thinking_level", None)
            return _success(command_id, "cycle_thinking_level", {"level": level})
        if command_type == "get_available_thinking_levels":
            levels = self.session.get_available_thinking_levels()
            return _success(command_id, "get_available_thinking_levels", {"levels": levels})
        if command_type == "bash":
            command = raw.get("command")
            if not command:
                return _error(command_id, "bash", "command is required")
            result = await self.session.execute_bash(
                str(command),
                exclude_from_context=bool(raw.get("excludeFromContext")),
                command_id=command_id,
            )
            return _success(command_id, "bash", result.to_dict())
        if command_type == "abort_bash":
            self.session.abort_bash()
            return _success(command_id, "abort_bash")
        if command_type == "export_html":
            path = await self.session.export_to_html(raw.get("outputPath"))
            return _success(command_id, "export_html", {"path": path})
        return _error(command_id, str(command_type), f"Unknown command: {command_type}")

    @staticmethod
    def _tree_node(node: TreeNode) -> Dict[str, Any]:
        """Serialise one tree node the way pi's `SessionTreeNode` JSON does."""
        data: Dict[str, Any] = {
            "entry": to_jsonable(node.entry),
            "children": [RpcServer._tree_node(child) for child in node.children],
        }
        if node.label:
            data["label"] = node.label
        return data

    async def _find_session(self, session_ref: str):
        """Resolve a session file path or bare session id for `switch_session`."""
        wanted = os.path.normcase(os.path.abspath(session_ref))
        for metadata in await self.session.list_sessions():
            path = getattr(metadata, "path", None)
            if metadata.id == session_ref:
                return metadata
            if path and os.path.normcase(os.path.abspath(path)) == wanted:
                return metadata
        return None

    async def _cmd_prompt(self, command_id: Optional[str], raw: Dict[str, Any]) -> Dict[str, Any]:
        text = str(raw.get("message", ""))
        images = _parse_images(raw.get("images"))
        streaming_behavior = raw.get("streamingBehavior")
        if streaming_behavior is not None and streaming_behavior not in ("steer", "followUp"):
            return _error(
                command_id, "prompt", f"Invalid streamingBehavior: {streaming_behavior!r} (expected 'steer' or 'followUp')"
            )
        # A prompt task that is spawned but has not run yet still counts as a run
        # in flight: `create_task` does not yield to the loop, so without this a
        # second buffered prompt would also be told `started` and then vanish into
        # a RuntimeError that `_run_prompt` swallows. pi answers that prompt with
        # the busy error instead (agent-session.ts prompt()).
        task = self._prompt_task
        busy = self.session.agent._active_run is not None or (task is not None and not task.done())
        if not busy:
            self._prompt_task = asyncio.get_running_loop().create_task(self._run_prompt(text, images))
            return _success(command_id, "prompt", {"disposition": "started"})
        # Streaming: pi requires an explicit streamingBehavior and routes
        # 'followUp' to the after-run queue (agent-session.ts prompt()).
        if streaming_behavior is None:
            return _error(
                command_id,
                "prompt",
                "Agent is already processing. Specify streamingBehavior ('steer' or 'followUp') to queue the message.",
            )
        if streaming_behavior == "followUp":
            self.session.follow_up(_user_message(text, images))
        else:
            self.session.steer(_user_message(text, images))
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

    async def _run_prompt(self, text: str, images: Optional[List[ImageContent]] = None) -> None:
        try:
            await self.session.prompt(text, images=images, auto_compact=self.auto_compaction_enabled)
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

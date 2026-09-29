"""Codex WebSocket transport primitives.

Ports the WebSocket half of pi-ai's `api/openai-codex-responses.ts`: at most one
idle socket per (session, account) pair, reused for follow-up turns. When the
next request body matches the previous one except for appended input, only the
delta plus `previous_response_id` is sent. Sockets are dropped after five
minutes idle or fifty-five minutes of age, and a session whose WebSocket attempt
fails before any event is pinned to SSE for the rest of its life.

`websockets` is imported defensively: without it the transport reports itself as
unavailable and the caller falls back to SSE, matching pi-ai's runtime probing.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple

from ..abort import AbortSignal
from ..errors import AbortError
from ..session_resources import register_session_resource_cleanup
from ..utils.diagnostics import format_thrown_value
from .codex_errors import CodexProtocolError

try:  # pragma: no cover - availability depends on the installed websockets release
    from websockets.asyncio.client import connect as _ws_connect
    from websockets.protocol import State as _WebSocketState

    WEBSOCKETS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ws_connect = None  # type: ignore[assignment]
    _WebSocketState = None  # type: ignore[assignment]
    WEBSOCKETS_AVAILABLE = False

OPENAI_BETA_RESPONSES_WEBSOCKETS = "responses_websockets=2026-02-06"
SESSION_WEBSOCKET_CACHE_TTL_MS = 5 * 60 * 1000
SESSION_WEBSOCKET_MAX_AGE_MS = 55 * 60 * 1000
DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS = 15_000
WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE = 1009

#: Codex events are small, but encrypted reasoning payloads are not.
WEBSOCKET_MAX_MESSAGE_BYTES = 16 * 1024 * 1024

_DONE = object()


class WebSocketUnavailableError(RuntimeError):
    """Raised when no WebSocket implementation is importable in this runtime."""


class WebSocketCloseError(Exception):
    """The socket closed (or failed) before the response completed."""

    def __init__(
        self,
        message: str,
        code: Optional[int] = None,
        reason: Optional[str] = None,
        was_clean: Optional[bool] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.reason = reason
        self.was_clean = was_clean


@dataclass
class OpenAICodexWebSocketDebugStats:
    """Per-session counters; mirrors pi-ai's `OpenAICodexWebSocketDebugStats`."""

    requests: int = 0
    connections_created: int = 0
    connections_reused: int = 0
    cached_context_requests: int = 0
    store_true_requests: int = 0
    full_context_requests: int = 0
    delta_requests: int = 0
    last_input_items: int = 0
    last_delta_input_items: Optional[int] = None
    last_previous_response_id: Optional[str] = None
    websocket_failures: int = 0
    sse_fallbacks: int = 0
    websocket_fallback_active: Optional[bool] = None
    last_websocket_error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """camelCase view matching pi-ai's debug struct."""
        result: Dict[str, Any] = {
            "requests": self.requests,
            "connectionsCreated": self.connections_created,
            "connectionsReused": self.connections_reused,
            "cachedContextRequests": self.cached_context_requests,
            "storeTrueRequests": self.store_true_requests,
            "fullContextRequests": self.full_context_requests,
            "deltaRequests": self.delta_requests,
            "lastInputItems": self.last_input_items,
            "websocketFailures": self.websocket_failures,
            "sseFallbacks": self.sse_fallbacks,
        }
        if self.last_delta_input_items is not None:
            result["lastDeltaInputItems"] = self.last_delta_input_items
        if self.last_previous_response_id is not None:
            result["lastPreviousResponseId"] = self.last_previous_response_id
        if self.websocket_fallback_active is not None:
            result["websocketFallbackActive"] = self.websocket_fallback_active
        if self.last_websocket_error is not None:
            result["lastWebSocketError"] = self.last_websocket_error
        return result


@dataclass
class CachedWebSocketContinuation:
    """What the backend still remembers from the previous turn on this socket."""

    last_request_body: Dict[str, Any]
    last_response_id: str
    last_response_items: List[Any]


@dataclass
class CachedWebSocketConnection:
    socket: Any
    busy: bool
    created_at: float
    idle_handle: Optional[asyncio.TimerHandle] = None
    continuation: Optional[CachedWebSocketContinuation] = None


_session_cache: Dict[str, Dict[str, CachedWebSocketConnection]] = {}
_debug_stats: Dict[str, OpenAICodexWebSocketDebugStats] = {}
_sse_fallback_sessions: Set[str] = set()


def _drop_entry(session_id: str, account_id: str, target: CachedWebSocketConnection) -> None:
    current = _session_cache.get(session_id)
    if current is not None and current.get(account_id) is target:
        current.pop(account_id, None)
    if current is not None and not current:
        _session_cache.pop(session_id, None)


# ---------------------------------------------------------------------------
# Debug stats and fallback bookkeeping
# ---------------------------------------------------------------------------


def _get_or_create_stats(session_id: str) -> OpenAICodexWebSocketDebugStats:
    stats = _debug_stats.get(session_id)
    if stats is None:
        stats = OpenAICodexWebSocketDebugStats()
        _debug_stats[session_id] = stats
    return stats


def get_openai_codex_websocket_debug_stats(session_id: str) -> Optional[OpenAICodexWebSocketDebugStats]:
    stats = _debug_stats.get(session_id)
    if stats is None:
        return None
    return OpenAICodexWebSocketDebugStats(**vars(stats))


def reset_openai_codex_websocket_debug_stats(session_id: Optional[str] = None) -> None:
    if session_id:
        _debug_stats.pop(session_id, None)
        _sse_fallback_sessions.discard(session_id)
        return
    _debug_stats.clear()
    _sse_fallback_sessions.clear()


def is_websocket_sse_fallback_active(session_id: Optional[str]) -> bool:
    return bool(session_id) and session_id in _sse_fallback_sessions


def record_websocket_sse_fallback(session_id: Optional[str]) -> None:
    if not session_id:
        return
    stats = _get_or_create_stats(session_id)
    stats.sse_fallbacks += 1
    stats.websocket_fallback_active = is_websocket_sse_fallback_active(session_id)


def record_websocket_failure(session_id: Optional[str], error: BaseException) -> None:
    if not session_id:
        return
    _sse_fallback_sessions.add(session_id)
    stats = _get_or_create_stats(session_id)
    stats.websocket_failures += 1
    stats.last_websocket_error = format_thrown_value(error)
    stats.websocket_fallback_active = True


def record_websocket_request(
    session_id: Optional[str],
    reused: bool,
    use_cached_context: bool,
    request_body: Dict[str, Any],
) -> None:
    """Updates the per-session counters for one outgoing WebSocket request."""
    if not session_id:
        return
    stats = _get_or_create_stats(session_id)
    stats.requests += 1
    if reused:
        stats.connections_reused += 1
    else:
        stats.connections_created += 1
    if use_cached_context:
        stats.cached_context_requests += 1
    if request_body.get("store") is True:
        stats.store_true_requests += 1
    input_items = request_body.get("input") or []
    stats.last_input_items = len(input_items)
    if request_body.get("previous_response_id"):
        stats.delta_requests += 1
        stats.last_delta_input_items = len(input_items)
        stats.last_previous_response_id = request_body["previous_response_id"]
    else:
        stats.full_context_requests += 1
        stats.last_delta_input_items = None
        stats.last_previous_response_id = None


def websocket_session_ids() -> List[str]:
    """Sessions currently holding cached sockets (diagnostics helper)."""
    return list(_session_cache)


# ---------------------------------------------------------------------------
# Socket lifecycle
# ---------------------------------------------------------------------------


def _get_ready_state(socket: Any) -> Optional[str]:
    state = getattr(socket, "state", None)
    return getattr(state, "name", None) if state is not None else None


def is_websocket_reusable(socket: Any) -> bool:
    if _WebSocketState is None:
        return getattr(socket, "state", None) is None
    return getattr(socket, "state", None) is _WebSocketState.OPEN


def _is_session_expired(entry: CachedWebSocketConnection) -> bool:
    return (time.time() * 1000 - entry.created_at) >= SESSION_WEBSOCKET_MAX_AGE_MS


def close_websocket_silently(socket: Any, code: int = 1000, reason: str = "done") -> None:
    try:
        result = socket.close(code, reason)
    except Exception:
        return
    if inspect.isawaitable(result):

        async def swallow() -> None:
            try:
                await result
            except Exception:
                pass

        try:
            asyncio.ensure_future(swallow())
        except RuntimeError:  # pragma: no cover - no running loop
            pass


def _close_error(error: BaseException) -> BaseException:
    code = getattr(error, "code", None)
    reason = getattr(error, "reason", None)
    if code is None and reason is None:
        return error if isinstance(error, BaseException) else RuntimeError(str(error))
    code_text = f" {code}" if isinstance(code, int) else ""
    reason_text = f" {reason}" if isinstance(reason, str) and reason else ""
    if not reason_text and code == WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE:
        reason_text = " message too big"
    return WebSocketCloseError(
        f"WebSocket closed{code_text}{reason_text}".strip(),
        code=code if isinstance(code, int) else None,
        reason=reason if isinstance(reason, str) and reason else None,
        was_clean=type(error).__name__ == "ConnectionClosedOK",
    )


def _resolve_proxy(url: str, env: Optional[Dict[str, str]]) -> Any:
    """`websockets` reads the process proxy env itself; provider env overrides win."""
    if not env:
        return True
    for key in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        if env.get(key):
            return env[key]
    return True


async def connect_websocket(
    url: str,
    headers: Dict[str, str],
    signal: Optional[AbortSignal] = None,
    connect_timeout_ms: Optional[int] = DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS,
    env: Optional[Dict[str, str]] = None,
) -> Any:
    """Opens one socket; the connection stays open until it is closed explicitly."""
    if not WEBSOCKETS_AVAILABLE:
        raise WebSocketUnavailableError("WebSocket transport is not available in this runtime")

    connect_headers = {key: value for key, value in headers.items() if key.lower() != "openai-beta"}
    # `websockets` owns the User-Agent header on the handshake, so it is passed
    # through the dedicated parameter rather than as an extra header.
    user_agent = next((value for key, value in connect_headers.items() if key.lower() == "user-agent"), None)
    connect_headers = {key: value for key, value in connect_headers.items() if key.lower() != "user-agent"}
    timeout_ms = connect_timeout_ms if connect_timeout_ms is not None else DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS
    timeout_s = timeout_ms / 1000 if timeout_ms and timeout_ms > 0 else None

    connect_task = asyncio.ensure_future(
        _ws_connect(
            url,
            additional_headers=connect_headers,
            user_agent_header=user_agent,
            open_timeout=timeout_s if timeout_s is not None else DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS / 1000,
            close_timeout=10,
            max_size=WEBSOCKET_MAX_MESSAGE_BYTES,
            proxy=_resolve_proxy(url, env),
        )
    )
    abort_task = asyncio.ensure_future(signal.wait()) if signal is not None else None
    try:
        waiters: Set[asyncio.Future] = {connect_task}
        if abort_task is not None:
            waiters.add(abort_task)
        done, _ = await asyncio.wait(waiters, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)

        if abort_task is not None and abort_task in done:
            raise AbortError("Request was aborted")
        if connect_task in done:
            try:
                return connect_task.result()
            except asyncio.TimeoutError as error:  # websockets' own open_timeout
                raise TimeoutError(f"WebSocket connect timeout after {timeout_ms}ms") from error
        raise TimeoutError(f"WebSocket connect timeout after {timeout_ms}ms")
    except asyncio.TimeoutError as error:
        raise TimeoutError(f"WebSocket connect timeout after {timeout_ms}ms") from error
    finally:
        for task in (connect_task, abort_task):
            if task is not None and not task.done():
                task.cancel()


def _schedule_session_expiry(session_id: str, account_id: str, entry: CachedWebSocketConnection) -> None:
    if entry.idle_handle is not None:
        entry.idle_handle.cancel()

    def expire() -> None:
        entry.idle_handle = None
        if entry.busy:
            return
        close_websocket_silently(entry.socket, 1000, "idle_timeout")
        account_entries = _session_cache.get(session_id)
        if account_entries is not None and account_entries.get(account_id) is entry:
            account_entries.pop(account_id, None)
        if account_entries is not None and not account_entries:
            _session_cache.pop(session_id, None)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - sockets are only cached inside a loop
        return
    entry.idle_handle = loop.call_later(SESSION_WEBSOCKET_CACHE_TTL_MS / 1000, expire)


async def close_openai_codex_websocket_sessions(session_id: Optional[str] = None) -> None:
    """Closes cached sockets for one session (or all sessions)."""
    entries: List[CachedWebSocketConnection] = []
    if session_id:
        entries = list((_session_cache.get(session_id) or {}).values())
        _session_cache.pop(session_id, None)
    else:
        for account_entries in _session_cache.values():
            entries.extend(account_entries.values())
        _session_cache.clear()

    for entry in entries:
        if entry.idle_handle is not None:
            entry.idle_handle.cancel()
            entry.idle_handle = None
        close_websocket_silently(entry.socket, 1000, "debug_close")


register_session_resource_cleanup(close_openai_codex_websocket_sessions)


async def acquire_websocket(
    url: str,
    headers: Dict[str, str],
    session_id: Optional[str],
    account_id: str,
    signal: Optional[AbortSignal] = None,
    connect_timeout_ms: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[Any, Optional[CachedWebSocketConnection], bool, Callable[..., Any]]:
    """Returns `(socket, entry, reused, release)`.

    Without a session id every request gets a fresh socket that is closed when
    released. With one, an idle reusable socket is handed back instead of
    connecting, and `release(keep=True)` returns it to the cache.
    """
    if not session_id:
        socket = await connect_websocket(url, headers, signal, connect_timeout_ms, env)

        async def release_fresh(**_: Any) -> None:
            close_websocket_silently(socket)

        return socket, None, False, release_fresh

    account_entries = _session_cache.get(session_id)
    cached = account_entries.get(account_id) if account_entries else None

    if cached is not None:
        if cached.idle_handle is not None:
            cached.idle_handle.cancel()
            cached.idle_handle = None
        if not cached.busy and _is_session_expired(cached):
            close_websocket_silently(cached.socket, 1000, "connection_age_limit")
            _drop_entry(session_id, account_id, cached)
        elif not cached.busy and is_websocket_reusable(cached.socket):
            cached.busy = True

            async def release_reused(keep: bool = False) -> None:
                if not keep or not is_websocket_reusable(cached.socket):
                    close_websocket_silently(cached.socket)
                    cached.continuation = None
                    _drop_entry(session_id, account_id, cached)
                    return
                cached.busy = False
                _schedule_session_expiry(session_id, account_id, cached)

            return cached.socket, cached, True, release_reused
        elif cached.busy:
            busy_socket = await connect_websocket(url, headers, signal, connect_timeout_ms, env)

            async def release_busy(**_: Any) -> None:
                close_websocket_silently(busy_socket)

            return busy_socket, None, False, release_busy
        elif not is_websocket_reusable(cached.socket):
            close_websocket_silently(cached.socket)
            _drop_entry(session_id, account_id, cached)

    socket = await connect_websocket(url, headers, signal, connect_timeout_ms, env)
    entry = CachedWebSocketConnection(socket=socket, busy=True, created_at=time.time() * 1000)
    account_entries = _session_cache.get(session_id)
    if account_entries is None:
        account_entries = {}
        _session_cache[session_id] = account_entries
    account_entries[account_id] = entry

    async def release_new(keep: bool = False) -> None:
        if not keep or not is_websocket_reusable(entry.socket):
            close_websocket_silently(entry.socket)
            if entry.idle_handle is not None:
                entry.idle_handle.cancel()
                entry.idle_handle = None
            _drop_entry(session_id, account_id, entry)
            return
        entry.busy = False
        _schedule_session_expiry(session_id, account_id, entry)

    return socket, entry, False, release_new


# ---------------------------------------------------------------------------
# Stream parsing
# ---------------------------------------------------------------------------


async def _next_event(
    queue: "asyncio.Queue[Any]",
    socket: Any,
    signal: Optional[AbortSignal],
    abort_task: Optional[asyncio.Future],
    idle_timeout_s: Optional[float],
) -> Any:
    get_task = asyncio.ensure_future(queue.get())
    waiters: Set[asyncio.Future] = {get_task}
    if abort_task is not None and not abort_task.done():
        waiters.add(abort_task)
    try:
        done, _ = await asyncio.wait(waiters, timeout=idle_timeout_s, return_when=asyncio.FIRST_COMPLETED)
        if not done:
            timeout_ms = int((idle_timeout_s or 0) * 1000)
            close_websocket_silently(socket, 1000, "idle_timeout")
            raise TimeoutError(f"WebSocket idle timeout after {timeout_ms}ms")
        if abort_task is not None and abort_task in done:
            raise AbortError("Request was aborted")
        return get_task.result()
    finally:
        if not get_task.done():
            get_task.cancel()


async def parse_websocket(
    socket: Any,
    signal: Optional[AbortSignal] = None,
    idle_timeout_ms: Optional[int] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Yields raw Codex events until the terminal response event arrives."""
    queue: "asyncio.Queue[Any]" = asyncio.Queue()
    state: Dict[str, Any] = {"failed": None, "saw_completion": False}

    async def reader() -> None:
        try:
            async for raw in socket:
                text = raw if isinstance(raw, str) else bytes(raw).decode("utf-8", "replace")
                try:
                    parsed = json.loads(text)
                except ValueError as cause:
                    state["failed"] = CodexProtocolError(
                        f"Invalid Codex WebSocket JSON: {format_thrown_value(cause)}", payload=text
                    )
                    return
                if not isinstance(parsed, dict):
                    continue
                if parsed.get("type") in ("response.completed", "response.done", "response.incomplete"):
                    state["saw_completion"] = True
                    queue.put_nowait(parsed)
                    return
                queue.put_nowait(parsed)
        except asyncio.CancelledError:
            raise
        except BaseException as error:  # transport failures surface as close errors
            if state["failed"] is None and not state["saw_completion"]:
                state["failed"] = _close_error(error)
        finally:
            queue.put_nowait(_DONE)

    reader_task = asyncio.ensure_future(reader())
    abort_task = asyncio.ensure_future(signal.wait()) if signal is not None else None
    idle_timeout_s = idle_timeout_ms / 1000 if idle_timeout_ms and idle_timeout_ms > 0 else None
    try:
        while True:
            if signal is not None and signal.aborted:
                raise AbortError("Request was aborted")
            item = await _next_event(queue, socket, signal, abort_task, idle_timeout_s)
            if item is _DONE:
                break
            yield item

        if state["failed"] is not None:
            raise state["failed"]
        if not state["saw_completion"]:
            raise RuntimeError("WebSocket stream closed before response.completed")
    finally:
        reader_task.cancel()
        if abort_task is not None:
            abort_task.cancel()
        await asyncio.gather(reader_task, return_exceptions=True)


# ---------------------------------------------------------------------------
# Cached-context request bodies
# ---------------------------------------------------------------------------


def _body_without_input(body: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in body.items() if key not in ("input", "previous_response_id")}


def _inputs_equal(a: Any, b: Any) -> bool:
    return json.dumps(a if a is not None else [], separators=(",", ":")) == json.dumps(
        b if b is not None else [], separators=(",", ":")
    )


def request_bodies_match_except_input(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return json.dumps(_body_without_input(a), separators=(",", ":")) == json.dumps(
        _body_without_input(b), separators=(",", ":")
    )


def get_cached_websocket_input_delta(
    body: Dict[str, Any],
    continuation: CachedWebSocketContinuation,
) -> Optional[List[Any]]:
    """Input items appended since the cached turn, or None when the cache cannot apply."""
    if not request_bodies_match_except_input(body, continuation.last_request_body):
        return None

    current_input = body.get("input") or []
    baseline = list(continuation.last_request_body.get("input") or []) + list(continuation.last_response_items)
    if len(current_input) < len(baseline):
        return None

    prefix = current_input[: len(baseline)]
    if not _inputs_equal(prefix, baseline):
        return None

    return current_input[len(baseline) :]


def build_cached_websocket_request_body(
    entry: CachedWebSocketConnection,
    body: Dict[str, Any],
) -> Dict[str, Any]:
    """Rewrites `body` as a delta request, or clears the cache and returns it unchanged."""
    continuation = entry.continuation
    if continuation is None:
        return body

    delta = get_cached_websocket_input_delta(body, continuation)
    if delta is None or not continuation.last_response_id:
        entry.continuation = None
        return body

    return {**body, "previous_response_id": continuation.last_response_id, "input": delta}


__all__ = [
    "DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS",
    "OPENAI_BETA_RESPONSES_WEBSOCKETS",
    "SESSION_WEBSOCKET_CACHE_TTL_MS",
    "SESSION_WEBSOCKET_MAX_AGE_MS",
    "WEBSOCKETS_AVAILABLE",
    "WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE",
    "CachedWebSocketConnection",
    "CachedWebSocketContinuation",
    "OpenAICodexWebSocketDebugStats",
    "WebSocketCloseError",
    "WebSocketUnavailableError",
    "acquire_websocket",
    "build_cached_websocket_request_body",
    "close_openai_codex_websocket_sessions",
    "close_websocket_silently",
    "connect_websocket",
    "get_cached_websocket_input_delta",
    "get_openai_codex_websocket_debug_stats",
    "is_websocket_reusable",
    "is_websocket_sse_fallback_active",
    "parse_websocket",
    "record_websocket_failure",
    "record_websocket_request",
    "record_websocket_sse_fallback",
    "request_bodies_match_except_input",
    "reset_openai_codex_websocket_debug_stats",
]

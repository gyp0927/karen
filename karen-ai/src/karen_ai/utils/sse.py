"""Server-Sent Events decoding, mirroring the decoder in pi-ai's anthropic-messages.ts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterable, AsyncIterator, List, Optional

from ..abort import AbortSignal


@dataclass
class ServerSentEvent:
    event: Optional[str]
    data: str
    raw: List[str] = field(default_factory=list)


@dataclass
class _SseDecoderState:
    event: Optional[str] = None
    data: List[str] = field(default_factory=list)
    raw: List[str] = field(default_factory=list)


def _flush_event(state: _SseDecoderState) -> Optional[ServerSentEvent]:
    if state.event is None and not state.data:
        return None
    event = ServerSentEvent(event=state.event, data="\n".join(state.data), raw=list(state.raw))
    state.event = None
    state.data = []
    state.raw = []
    return event


def decode_sse_line(line: str, state: _SseDecoderState) -> Optional[ServerSentEvent]:
    if line == "":
        return _flush_event(state)

    state.raw.append(line)
    if line.startswith(":"):
        return None

    delimiter_index = line.find(":")
    field_name = line if delimiter_index == -1 else line[:delimiter_index]
    value = "" if delimiter_index == -1 else line[delimiter_index + 1 :]
    if value.startswith(" "):
        value = value[1:]

    if field_name == "event":
        state.event = value
    elif field_name == "data":
        state.data.append(value)

    return None


async def iterate_sse_messages(
    chunks: AsyncIterable[str],
    signal: Optional[AbortSignal] = None,
) -> AsyncIterator[ServerSentEvent]:
    """Decode an SSE byte/text stream into events, one per blank-line-separated block."""
    state = _SseDecoderState()
    buffer = ""

    async def feed(text: str):
        nonlocal buffer
        buffer += text
        events = []
        while True:
            idx_n = buffer.find("\n")
            idx_r = buffer.find("\r")
            if idx_n == -1 and idx_r == -1:
                break
            if idx_r == -1:
                idx = idx_n
            elif idx_n == -1:
                idx = idx_r
            else:
                idx = min(idx_n, idx_r)
            line = buffer[:idx]
            next_idx = idx + 1
            if buffer[idx] == "\r" and next_idx < len(buffer) and buffer[next_idx] == "\n":
                next_idx += 1
            buffer = buffer[next_idx:]
            event = decode_sse_line(line, state)
            if event is not None:
                events.append(event)
        return events

    async for chunk in chunks:
        if signal is not None and signal.aborted:
            raise RuntimeError("Request was aborted")
        for event in await feed(chunk):
            yield event

    # Flush any trailing partial line and pending event.
    if buffer:
        event = decode_sse_line(buffer, state)
        if event is not None:
            yield event
    trailing = _flush_event(state)
    if trailing is not None:
        yield trailing

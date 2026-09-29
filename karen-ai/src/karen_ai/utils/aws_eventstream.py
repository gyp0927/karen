"""AWS event-stream (application/vnd.amazon.eventstream) decoder.

The Bedrock ConverseStream response body is a sequence of binary-framed
messages: prelude (total length + headers length), prelude CRC32, headers,
payload, message CRC32. Each message's `:message-type`/`:event-type` headers
name the event (messageStart, contentBlockDelta, ...) or exception.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class EventStreamError(ValueError):
    """Malformed event-stream framing (bad CRC, truncated message, bad header)."""


@dataclass
class EventStreamMessage:
    headers: Dict[str, Any]
    payload: bytes

    @property
    def message_type(self) -> Optional[str]:
        value = self.headers.get(":message-type")
        return value if isinstance(value, str) else None

    @property
    def event_type(self) -> Optional[str]:
        value = self.headers.get(":event-type")
        return value if isinstance(value, str) else None

    @property
    def exception_type(self) -> Optional[str]:
        value = self.headers.get(":exception-type")
        return value if isinstance(value, str) else None

    def json_payload(self) -> Any:
        if not self.payload:
            return None
        return json.loads(self.payload.decode("utf-8"))


_PRELUDE_BYTES = 8
_PRELUDE_CRC_BYTES = 4
_MESSAGE_CRC_BYTES = 4

_HEADER_VALUE_TYPES = {
    0: "bool_true",
    1: "bool_false",
    2: "byte",
    3: "short",
    4: "int",
    5: "long",
    6: "bytes",
    7: "string",
    8: "timestamp",
    9: "uuid",
}


def _decode_headers(data: bytes) -> Dict[str, Any]:
    headers: Dict[str, Any] = {}
    offset = 0
    while offset < len(data):
        if offset + 1 > len(data):
            raise EventStreamError("Truncated header name length")
        name_len = data[offset]
        offset += 1
        if offset + name_len + 1 > len(data):
            raise EventStreamError("Truncated header name")
        name = data[offset : offset + name_len].decode("utf-8")
        offset += name_len
        value_type = data[offset]
        offset += 1

        if value_type == 0:
            headers[name] = True
        elif value_type == 1:
            headers[name] = False
        elif value_type == 2:
            headers[name] = struct.unpack(">b", data[offset : offset + 1])[0]
            offset += 1
        elif value_type == 3:
            headers[name] = struct.unpack(">h", data[offset : offset + 2])[0]
            offset += 2
        elif value_type == 4:
            headers[name] = struct.unpack(">i", data[offset : offset + 4])[0]
            offset += 4
        elif value_type == 5:
            headers[name] = struct.unpack(">q", data[offset : offset + 8])[0]
            offset += 8
        elif value_type in (6, 7):
            (value_len,) = struct.unpack(">H", data[offset : offset + 2])
            offset += 2
            raw = data[offset : offset + value_len]
            offset += value_len
            headers[name] = raw.decode("utf-8") if value_type == 7 else raw
        elif value_type == 8:
            headers[name] = struct.unpack(">q", data[offset : offset + 8])[0]
            offset += 8
        elif value_type == 9:
            headers[name] = data[offset : offset + 16]
            offset += 16
        else:
            raise EventStreamError(f"Unknown header value type: {value_type}")
    return headers


def decode_message(buffer: bytes) -> Tuple[EventStreamMessage, bytes]:
    """Decode one message from the front of `buffer`, returning (message, rest).

    Raises EventStreamError on malformed framing. Callers must only invoke
    this once `len(buffer) >= total_length` (see `needed_bytes`).
    """
    if len(buffer) < _PRELUDE_BYTES + _PRELUDE_CRC_BYTES:
        raise EventStreamError("Not enough bytes for a message prelude")
    total_length, headers_length = struct.unpack(">II", buffer[:_PRELUDE_BYTES])
    if len(buffer) < total_length:
        raise EventStreamError("Not enough bytes for the full message")

    prelude_crc_expected = struct.unpack(">I", buffer[_PRELUDE_BYTES : _PRELUDE_BYTES + 4])[0]
    if zlib.crc32(buffer[:_PRELUDE_BYTES]) & 0xFFFFFFFF != prelude_crc_expected:
        raise EventStreamError("Prelude CRC mismatch")

    message_crc_expected = struct.unpack(">I", buffer[total_length - 4 : total_length])[0]
    if zlib.crc32(buffer[: total_length - 4]) & 0xFFFFFFFF != message_crc_expected:
        raise EventStreamError("Message CRC mismatch")

    headers_start = _PRELUDE_BYTES + _PRELUDE_CRC_BYTES
    headers_end = headers_start + headers_length
    payload = buffer[headers_end : total_length - _MESSAGE_CRC_BYTES]
    headers = _decode_headers(buffer[headers_start:headers_end])
    return EventStreamMessage(headers=headers, payload=payload), buffer[total_length:]


class EventStreamDecoder:
    """Incremental decoder fed with response chunks."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> List[EventStreamMessage]:
        self._buffer.extend(data)
        messages: List[EventStreamMessage] = []
        while True:
            if len(self._buffer) < _PRELUDE_BYTES + _PRELUDE_CRC_BYTES:
                break
            total_length, _headers_length = struct.unpack(">II", self._buffer[:_PRELUDE_BYTES])
            if total_length < _PRELUDE_BYTES + _PRELUDE_CRC_BYTES + _MESSAGE_CRC_BYTES:
                raise EventStreamError(f"Invalid message total length: {total_length}")
            if len(self._buffer) < total_length:
                break
            message, rest = decode_message(bytes(self._buffer))
            messages.append(message)
            self._buffer = bytearray(rest)
        return messages

    def flush(self) -> List[EventStreamMessage]:
        """Decode any complete trailing message; the stream must end on a boundary."""
        if self._buffer:
            raise EventStreamError(f"Event stream ended with {len(self._buffer)} undelivered bytes")
        return []

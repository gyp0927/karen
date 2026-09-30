"""UUIDv7 id generation (pi uses `@earendil-works/pi-ai/utils/uuid`'s `uuidv7`).

UUIDv7 (RFC 9562): 48-bit millisecond Unix timestamp prefix, so ids sort by
creation time; version/variant bits per spec; remaining bits random.
"""

from __future__ import annotations

import secrets
import time
import uuid
from typing import Optional


def uuid7(timestamp_ms: Optional[int] = None) -> str:
    ts = int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms)
    ts &= (1 << 48) - 1
    rand = secrets.token_bytes(10)
    data = bytearray(16)
    for shift, index in zip(range(40, -1, -8), range(6)):
        data[index] = (ts >> shift) & 0xFF
    data[6] = 0x70 | (rand[0] & 0x0F)
    data[7] = rand[1]
    data[8] = 0x80 | (rand[2] & 0x3F)
    data[9:] = rand[3:]
    return str(uuid.UUID(bytes=bytes(data)))


class Uuid7IdGenerator:
    """Default `IdGenerator`; accepts an optional leader timestamp in ms."""

    def next(self, timestamp_ms: Optional[int] = None) -> str:
        return uuid7(timestamp_ms)

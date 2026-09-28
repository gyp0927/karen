"""Shared abort/HTTP helpers for the OAuth flows (internal)."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Dict, Optional, TypeVar

import httpx

from ...abort import AbortSignal
from ...errors import AbortError

T = TypeVar("T")

#: Per-request timeout matching pi-ai's `AbortSignal.timeout(30_000)`.
REQUEST_TIMEOUT_SECONDS = 30.0


async def race_with_abort(
    awaitable: Awaitable[T],
    signal: Optional[AbortSignal],
    cancel_message: str = "Login cancelled",
) -> T:
    """Await `awaitable`, raising AbortError(cancel_message) if `signal` fires first.

    The underlying task is cancelled on abort, which also tears down any
    in-flight httpx request it is driving.
    """
    if signal is not None and signal.aborted:
        raise AbortError(cancel_message)
    task = asyncio.ensure_future(awaitable)
    if signal is None:
        return await task
    abort_task = asyncio.ensure_future(signal.wait())
    try:
        done, _pending = await asyncio.wait({task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return task.result()
        task.cancel()
        try:
            await task
        except BaseException:
            pass
        raise AbortError(cancel_message)
    finally:
        abort_task.cancel()
        try:
            await abort_task
        except BaseException:
            pass


async def post_form(
    url: str,
    fields: Dict[str, str],
    signal: Optional[AbortSignal],
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> httpx.Response:
    """POST application/x-www-form-urlencoded, abort-aware."""
    merged = {"Accept": "application/json", **(headers or {})}
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        return await race_with_abort(client.post(url, data=fields, headers=merged), signal)


async def post_json(
    url: str,
    body: Any,
    signal: Optional[AbortSignal],
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> httpx.Response:
    """POST a JSON body, abort-aware."""
    merged = {"Accept": "application/json", **(headers or {})}
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        return await race_with_abort(client.post(url, json=body, headers=merged), signal)


async def get(
    url: str,
    signal: Optional[AbortSignal],
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> httpx.Response:
    """GET, abort-aware."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        return await race_with_abort(client.get(url, headers=headers), signal)

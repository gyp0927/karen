"""Credential store implementations: in-memory and JSON-file backed.

Mirrors pi-ai's auth/credential-store.ts contract: `modify` is the only write
path and serializes read-modify-write per provider (asyncio locks in-process;
the file store reloads from disk inside the lock to stay correct across
processes for the common case).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Union

from pydantic import TypeAdapter

from .types import (
    ApiKeyCredential,
    AuthOperationOptions,
    Credential,
    CredentialInfo,
    OAuthCredential,
)

_credential_adapter: TypeAdapter[Credential] = TypeAdapter(Union[ApiKeyCredential, OAuthCredential])


def _clone(credential: Optional[Credential]) -> Optional[Credential]:
    if credential is None:
        return None
    return _credential_adapter.validate_python(credential.model_dump())


class InMemoryCredentialStore:
    def __init__(self) -> None:
        self._entries: Dict[str, Credential] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, provider_id: str) -> asyncio.Lock:
        return self._locks.setdefault(provider_id, asyncio.Lock())

    async def read(self, provider_id: str, options: Optional[AuthOperationOptions] = None) -> Optional[Credential]:
        if options and options.signal:
            options.signal.throw_if_aborted()
        return _clone(self._entries.get(provider_id))

    async def list(self, options: Optional[AuthOperationOptions] = None) -> List[CredentialInfo]:
        if options and options.signal:
            options.signal.throw_if_aborted()
        return [CredentialInfo(provider_id=pid, type=cred.type) for pid, cred in self._entries.items()]

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Optional[Credential]], Awaitable[Optional[Credential]]],
        options: Optional[AuthOperationOptions] = None,
    ) -> Optional[Credential]:
        if options and options.signal:
            options.signal.throw_if_aborted()
        async with self._lock_for(provider_id):
            current = _clone(self._entries.get(provider_id))
            updated = await fn(current)
            if updated is not None:
                self._entries[provider_id] = _clone(updated)
            return _clone(self._entries.get(provider_id))

    async def delete(self, provider_id: str, options: Optional[AuthOperationOptions] = None) -> None:
        if options and options.signal:
            options.signal.throw_if_aborted()
        async with self._lock_for(provider_id):
            self._entries.pop(provider_id, None)


class JsonFileCredentialStore:
    """Single JSON file holding one credential per provider id:

    ```json
    {
      "anthropic": { "type": "api_key", "key": "sk-ant-..." },
      "openai": { "type": "oauth", "refresh": "...", "access": "...", "expires": 0 }
    }
    ```
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, provider_id: str) -> asyncio.Lock:
        return self._locks.setdefault(provider_id, asyncio.Lock())

    def _load(self) -> Dict[str, Credential]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        entries: Dict[str, Credential] = {}
        for provider_id, value in raw.items():
            try:
                entries[provider_id] = _credential_adapter.validate_python(value)
            except Exception:
                continue  # Skip entries from newer/older formats.
        return entries

    def _save(self, entries: Dict[str, Credential]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {pid: cred.model_dump(by_alias=True, mode="json") for pid, cred in entries.items()}
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    async def read(self, provider_id: str, options: Optional[AuthOperationOptions] = None) -> Optional[Credential]:
        if options and options.signal:
            options.signal.throw_if_aborted()
        return _clone(self._load().get(provider_id))

    async def list(self, options: Optional[AuthOperationOptions] = None) -> List[CredentialInfo]:
        if options and options.signal:
            options.signal.throw_if_aborted()
        return [CredentialInfo(provider_id=pid, type=cred.type) for pid, cred in self._load().items()]

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Optional[Credential]], Awaitable[Optional[Credential]]],
        options: Optional[AuthOperationOptions] = None,
    ) -> Optional[Credential]:
        if options and options.signal:
            options.signal.throw_if_aborted()
        async with self._lock_for(provider_id):
            entries = self._load()
            current = _clone(entries.get(provider_id))
            updated = await fn(current)
            if updated is not None:
                entries[provider_id] = _clone(updated)
                self._save(entries)
            return _clone(entries.get(provider_id))

    async def delete(self, provider_id: str, options: Optional[AuthOperationOptions] = None) -> None:
        if options and options.signal:
            options.signal.throw_if_aborted()
        async with self._lock_for(provider_id):
            entries = self._load()
            if provider_id in entries:
                del entries[provider_id]
                self._save(entries)

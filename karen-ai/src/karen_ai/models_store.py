"""Persistent model catalogs keyed by provider id, mirroring models-store.ts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional, Protocol, Union, runtime_checkable

from pydantic import Field, TypeAdapter

from .abort import AbortSignal, operation_signal
from .types import AnyModel, ClassifierModel, ImageModel, KarenBase, Model

_any_model_adapter: TypeAdapter[AnyModel] = TypeAdapter(Union[Model, ImageModel, ClassifierModel])


class ModelsStoreEntry(KarenBase):
    """Persisted models of every type, plus remote-catalog cache metadata."""

    models: List[AnyModel] = Field(default_factory=list)
    last_modified: Optional[int] = None
    checked_at: Optional[int] = None
    etag: Optional[str] = None


class ModelsStoreOperationOptions(KarenBase):
    signal: Optional[AbortSignal] = None

    model_config = {"arbitrary_types_allowed": True}


@runtime_checkable
class ModelsStore(Protocol):
    async def read(
        self, provider_id: str, options: Optional[ModelsStoreOperationOptions] = None
    ) -> Optional[ModelsStoreEntry]: ...

    async def write(
        self, provider_id: str, entry: ModelsStoreEntry, options: Optional[ModelsStoreOperationOptions] = None
    ) -> None: ...

    async def delete(self, provider_id: str, options: Optional[ModelsStoreOperationOptions] = None) -> None: ...


def _clone_entry(entry: ModelsStoreEntry) -> ModelsStoreEntry:
    return ModelsStoreEntry.model_validate(entry.model_dump())


class InMemoryModelsStore:
    def __init__(self) -> None:
        self._entries: dict[str, ModelsStoreEntry] = {}

    async def read(
        self, provider_id: str, options: Optional[ModelsStoreOperationOptions] = None
    ) -> Optional[ModelsStoreEntry]:
        operation_signal(options.signal if options else None).throw_if_aborted()
        entry = self._entries.get(provider_id)
        return _clone_entry(entry) if entry else None

    async def write(
        self, provider_id: str, entry: ModelsStoreEntry, options: Optional[ModelsStoreOperationOptions] = None
    ) -> None:
        operation_signal(options.signal if options else None).throw_if_aborted()
        self._entries[provider_id] = _clone_entry(entry)

    async def delete(self, provider_id: str, options: Optional[ModelsStoreOperationOptions] = None) -> None:
        operation_signal(options.signal if options else None).throw_if_aborted()
        self._entries.pop(provider_id, None)


class JsonFileModelsStore:
    """One JSON file per provider under a directory."""

    def __init__(self, directory: Union[str, Path]) -> None:
        self._dir = Path(directory)
        self._locks: dict[str, asyncio.Lock] = {}

    def _path_for(self, provider_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in provider_id)
        return self._dir / f"{safe}.json"

    def _lock_for(self, provider_id: str) -> asyncio.Lock:
        return self._locks.setdefault(provider_id, asyncio.Lock())

    async def read(
        self, provider_id: str, options: Optional[ModelsStoreOperationOptions] = None
    ) -> Optional[ModelsStoreEntry]:
        operation_signal(options.signal if options else None).throw_if_aborted()
        path = self._path_for(provider_id)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        try:
            return ModelsStoreEntry.model_validate(raw)
        except Exception:
            return None

    async def write(
        self, provider_id: str, entry: ModelsStoreEntry, options: Optional[ModelsStoreOperationOptions] = None
    ) -> None:
        operation_signal(options.signal if options else None).throw_if_aborted()
        async with self._lock_for(provider_id):
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._path_for(provider_id)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(entry.model_dump_json(by_alias=True, indent=2), encoding="utf-8")
            tmp.replace(path)

    async def delete(self, provider_id: str, options: Optional[ModelsStoreOperationOptions] = None) -> None:
        operation_signal(options.signal if options else None).throw_if_aborted()
        async with self._lock_for(provider_id):
            self._path_for(provider_id).unlink(missing_ok=True)

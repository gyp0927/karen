"""Discovery, loading, and the API an extension registers against.

pi loads TypeScript modules through jiti and hands each one an ``ExtensionAPI``;
karen imports Python modules with importlib and hands each one the same API,
spelled in snake_case. What a module must expose is the mirror of pi's default
export::

    def extension(pi):
        pi.register_command("hello", handler=...)

The factory may be ``async def`` — the loader awaits it before the session
starts, which is what lets an extension fetch configuration up front.

Where extensions are found, in load order (pi's order, karen's directories):

1. ``<cwd>/.karen/extensions/``   — project, only for this working directory
2. ``<agent_dir>/extensions/``    — user, i.e. ``~/.karen/extensions/``
3. settings ``extensions: [...]`` — explicit paths, in the order written
4. ``--extension PATH``           — the command line, after the settings list

Load order decides conflicts: duplicate paths, and same-named tools and
commands, are all first-registration-wins, so a project extension beats a
``--extension`` one with a colliding tool name (pi 1.2 loads the command line
first instead; karen keeps the snapshot's order).

Inside a search directory, every direct ``*.py`` file is an extension, and every
subdirectory holding an ``index.py`` is one. Nothing is read recursively and
files whose name starts with ``_`` or ``.`` are skipped, so a helper module can
live beside an extension. For a directory extension the directory itself goes on
``sys.path`` while it loads, so ``index.py`` can ``import helper`` from beside
it — pick distinctive helper names, since those modules stay imported.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from .events import EVENT_NAMES
from .runner import ErrorSink, ExtensionRunner
from .types import (
    Command,
    Extension,
    ToolDefinition,
    maybe_await,
)

__all__ = [
    "EXTENSIONS_DIR_NAME",
    "FACTORY_NAME",
    "KAREN_DIR_NAME",
    "ExtensionAPI",
    "ExtensionEvents",
    "ExtensionLoadDiagnostic",
    "LoadedExtensions",
    "discover_extension_entries",
    "extension_search_paths",
    "load_extensions",
    "load_extensions_async",
]

#: Directory under the config dir / cwd that holds extensions (pi: `extensions`).
EXTENSIONS_DIR_NAME = "extensions"

#: The project config directory (pi's `.pi`).
KAREN_DIR_NAME = ".karen"

#: The factory a module must expose.
FACTORY_NAME = "extension"


@dataclass
class ExtensionLoadDiagnostic:
    """A discovery, import, or factory problem, reported at startup."""

    path: str
    message: str
    severity: str = "error"  # "error" | "warning"


@dataclass
class LoadedExtensions:
    """Everything the loader produced: the runner plus what went wrong."""

    runner: ExtensionRunner
    extensions: List[Extension] = field(default_factory=list)
    diagnostics: List[ExtensionLoadDiagnostic] = field(default_factory=list)

    @property
    def errors(self) -> List[ExtensionLoadDiagnostic]:
        return [item for item in self.diagnostics if item.severity == "error"]

    @property
    def warnings(self) -> List[ExtensionLoadDiagnostic]:
        return [item for item in self.diagnostics if item.severity == "warning"]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def extension_search_paths(cwd: str, agent_dir: Any) -> List[Tuple[str, str]]:
    """The conventional extension directories as ``(path, source)``, in load order."""
    return [
        (str(Path(cwd) / KAREN_DIR_NAME / EXTENSIONS_DIR_NAME), "project"),
        (str(Path(agent_dir) / EXTENSIONS_DIR_NAME), "user"),
    ]


def _resolve_configured(path: str, cwd: str) -> str:
    expanded = Path(os.path.expanduser(path))
    if not expanded.is_absolute():
        expanded = Path(cwd) / expanded
    return str(expanded)


def _entries_in_directory(directory: Path, source: str) -> Tuple[List[Tuple[str, str]], List[ExtensionLoadDiagnostic]]:
    """Resolve one search directory into entry files."""
    entries: List[Tuple[str, str]] = []
    diagnostics: List[ExtensionLoadDiagnostic] = []
    try:
        children = sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError as error:
        diagnostics.append(
            ExtensionLoadDiagnostic(str(directory), f"cannot read extension directory: {error}")
        )
        return entries, diagnostics
    for child in children:
        name = child.name
        if name.startswith(".") or name.startswith("_"):
            continue
        try:
            if child.is_dir():
                index = child / "index.py"
                if index.is_file():
                    entries.append((str(index), source))
                continue
        except OSError as error:
            diagnostics.append(ExtensionLoadDiagnostic(str(child), f"cannot inspect: {error}"))
            continue
        if child.suffix == ".py" and child.is_file():
            entries.append((str(child), source))
    return entries, diagnostics


def discover_extension_entries(
    roots: Sequence[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], List[ExtensionLoadDiagnostic]]:
    """Resolve every search root (and explicit path) to entry files.

    A root that is itself a file — how ``--extension ./hello.py`` and a settings
    path usually arrive — is taken as the entry directly.
    """
    entries: List[Tuple[str, str]] = []
    diagnostics: List[ExtensionLoadDiagnostic] = []
    seen: set[str] = set()
    for raw_path, source in roots:
        path = Path(raw_path)
        if path.is_file():
            resolved = str(path)
            if resolved not in seen:
                seen.add(resolved)
                entries.append((resolved, source))
            continue
        if not path.is_dir():
            # A configured path that is not there is worth saying out loud; a
            # missing conventional directory is not (most projects have none).
            if source in ("settings", "cli"):
                diagnostics.append(
                    ExtensionLoadDiagnostic(raw_path, "extension path does not exist")
                )
            continue
        found, root_diagnostics = _entries_in_directory(path, source)
        diagnostics.extend(root_diagnostics)
        for entry, entry_source in found:
            if entry not in seen:
                seen.add(entry)
                entries.append((entry, entry_source))
    return entries, diagnostics


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _module_name(path: str) -> str:
    digest = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]
    return f"karen_extension_{digest}"


def _import_module(path: str) -> Any:
    """Import one extension file as a fresh module, without touching sys.path."""
    resolved = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location(_module_name(resolved), resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load a Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before executing so a module that imports itself (or is imported
    # by a helper) sees the half-built module instead of loading twice.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:  # cleanup, then let it surface as a diagnostic
        sys.modules.pop(spec.name, None)
        raise
    return module


class ExtensionEvents:
    """The extension-to-extension bus (pi's ``pi.events``).

    An in-process pub/sub keyed by string, so two extensions that never import
    each other can still coordinate: ``on(name, handler)`` returns the
    unsubscribe function, and ``emit`` is awaited (handlers may be ``async``).
    """

    def __init__(self, runner: ExtensionRunner) -> None:
        self._runner = runner

    def on(self, name: str, handler: Callable[[Any], Any]) -> Callable[[], None]:
        return self._runner.on_bus(name, handler)

    async def emit(self, name: str, payload: Any = None) -> None:
        await self._runner.emit_bus(name, payload)


class ExtensionAPI:
    """What an extension registers with (pi's ``ExtensionAPI``).

    Registration is only half of it; the rest are the session controls pi's API
    exposes, which here delegate to a host bound to the live session — they
    raise :class:`RuntimeError` when nothing is bound yet (inside a factory, or
    in a run without a session), exactly like pi's actions.
    """

    def __init__(
        self,
        record: Extension,
        runner: ExtensionRunner,
        host: Any = None,
        settings: Any = None,
    ) -> None:
        self._record = record
        self._runner = runner
        self._host = host
        self._settings = settings
        self.events = ExtensionEvents(runner)

    # -- registration ----------------------------------------------------------

    def on(self, event: str, handler: Callable[[Any, Any], Any]) -> Callable[[], None]:
        """Subscribe to *event*; returns the unsubscribe function."""
        if event not in EVENT_NAMES:
            raise ValueError(
                f"unknown event {event!r}; karen emits {', '.join(sorted(EVENT_NAMES))}"
            )
        handlers = self._record.handlers.setdefault(event, [])
        handlers.append(handler)

        def unsubscribe() -> None:
            if handler in handlers:
                handlers.remove(handler)

        return unsubscribe

    def register_tool(
        self,
        *,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        execute: Callable[..., Any],
        label: Optional[str] = None,
    ) -> None:
        """Add a tool the model can call.

        *execute* takes ``(tool_call_id, args, signal, on_update)`` and returns a
        :class:`~karen_coding_agent.extensions.types.ToolResultDict`, a mapping
        with those fields, a plain string, or raises on failure.
        """
        if not name:
            raise ValueError("a tool needs a name")
        self._record.tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            execute=execute,
            label=label,
        )

    def register_command(
        self,
        name: str,
        handler: Callable[[Optional[str], Any], Any],
        description: str = "",
    ) -> None:
        """Add a ``/`` command. *handler* takes ``(args_text, context)``."""
        if not name or name.startswith("/"):
            raise ValueError("a command name is written without the leading slash")
        self._record.commands[name] = Command(name=name, handler=handler, description=description)

    # -- session controls ------------------------------------------------------

    def _require_host(self) -> Any:
        if self._host is None or getattr(self._host, "session", None) is None:
            raise RuntimeError("no session is bound to this extension runtime")
        return self._host

    def get_settings(self) -> Any:
        """The merged settings the session was started with (never ``None``)."""
        return self._settings

    def get_commands(self) -> Dict[str, Command]:
        """Every registered extension command, this extension's included."""
        return self._runner.commands()

    def notify(self, message: str, level: str = "info") -> None:
        """Show *message* to the user without entering the transcript."""
        if self._host is not None:
            self._host.notify(message, level)

    def append_entry(self, custom_type: str, data: Any = None) -> Awaitable[str]:
        """Persist non-context session data (pi's ``appendEntry``)."""
        return self._require_host().append_entry(custom_type, data)

    def send_message(
        self,
        custom_type: str,
        content: Any,
        display: bool = False,
        details: Any = None,
    ) -> Awaitable[None]:
        """Store a custom message the model will see on its next request."""
        return self._require_host().send_message(custom_type, content, display, details)

    def send_user_message(
        self, text: str, images: Any = None, deliver_as: str = "steer"
    ) -> Awaitable[None]:
        """Queue a user message as if it had been typed (``steer``/``follow_up``)."""
        return self._require_host().send_user_message(text, images, deliver_as)

    def exec(self, command: str, **kwargs: Any) -> Awaitable[Any]:
        """Run a shell command through the session (pi's ``exec``) — recorded in the transcript."""
        return self._require_host().exec_bash(command, **kwargs)

    def set_session_name(self, name: str) -> Awaitable[None]:
        return self._require_host().set_session_name(name)

    def set_label(self, entry_id: str, label: Optional[str]) -> Awaitable[None]:
        return self._require_host().set_label(entry_id, label)

    def get_active_tools(self) -> List[str]:
        """The tool names currently declared to the model."""
        return self._require_host().get_active_tools()

    def set_active_tools(self, names: Sequence[str]) -> None:
        """Replace the active tool set; unknown names are ignored, like pi."""
        self._require_host().set_active_tools(list(names))

    def get_thinking_level(self) -> Optional[str]:
        return self._require_host().get_thinking_level()

    def set_thinking_level(self, level: str) -> Awaitable[None]:
        return self._require_host().set_thinking_level(level)

    def set_model(self, spec: str) -> Awaitable[bool]:
        """Switch models by ``"provider/id"``; False when no such model exists."""
        return self._require_host().set_model(spec)


def _load_one(
    entry: str,
    source: str,
    runner: ExtensionRunner,
    host: Any,
    settings: Any,
    diagnostics: List[ExtensionLoadDiagnostic],
) -> Optional[Extension]:
    """Import and run one extension module; records diagnostics on failure."""
    record = Extension(path=entry, module_path=os.path.abspath(entry), source=source)
    directory = os.path.dirname(os.path.abspath(entry))
    added_to_path = False
    try:
        # A directory extension can import its own helper modules.
        if directory not in sys.path:
            sys.path.insert(0, directory)
            added_to_path = True
        module = _import_module(entry)
        factory = getattr(module, FACTORY_NAME, None)
        if factory is None:
            raise AttributeError(
                f"the module defines no {FACTORY_NAME}() factory "
                f"(add `def {FACTORY_NAME}(pi): ...`)"
            )
        if not callable(factory):
            raise TypeError(f"{FACTORY_NAME} must be callable, not {type(factory).__name__}")
        api = ExtensionAPI(record, runner, host=host, settings=settings)
        result = factory(api)
        if hasattr(result, "__await__"):
            return record, result  # an async factory; the caller awaits it
        return record, None
    except Exception as error:  # noqa: BLE001 - reported, never fatal
        diagnostics.append(
            ExtensionLoadDiagnostic(
                entry,
                f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}".strip(),
            )
        )
        sys.modules.pop(_module_name(entry), None)
        return None
    finally:
        if added_to_path:
            try:
                sys.path.remove(directory)
            except ValueError:
                pass


async def load_extensions_async(
    *,
    cwd: str,
    agent_dir: Any,
    settings_paths: Sequence[str] = (),
    cli_paths: Sequence[str] = (),
    host: Any = None,
    settings: Any = None,
    on_error: Optional[ErrorSink] = None,
) -> LoadedExtensions:
    """Discover and load every extension, awaiting async factories.

    Failures never stop startup: a module that does not import, a factory that
    raises, and a registration that is rejected all land in ``diagnostics`` (and
    in the runner's error list) while the remaining extensions load normally.
    """
    roots = extension_search_paths(cwd, agent_dir)
    roots += [(_resolve_configured(path, cwd), "settings") for path in settings_paths]
    roots += [(_resolve_configured(path, cwd), "cli") for path in cli_paths]
    entries, diagnostics = discover_extension_entries(roots)

    runner = ExtensionRunner([], context_factory=getattr(host, "make_context", None), on_error=on_error)
    runner.host = host
    loaded: List[Extension] = []
    for entry, source in entries:
        outcome = _load_one(entry, source, runner, host, settings, diagnostics)
        if outcome is None:
            continue
        record, pending = outcome
        if pending is not None:
            try:
                await maybe_await(pending)
            except Exception as error:  # noqa: BLE001 - reported, never fatal
                diagnostics.append(
                    ExtensionLoadDiagnostic(
                        entry,
                        f"async {FACTORY_NAME}() failed: {type(error).__name__}: {error}",
                    )
                )
                continue
        loaded.append(record)

    runner.extensions = loaded
    return LoadedExtensions(runner=runner, extensions=loaded, diagnostics=diagnostics)


def load_extensions(**kwargs: Any) -> LoadedExtensions:
    """Synchronous :func:`load_extensions_async`, for callers with no loop.

    It drives its own loop with :func:`asyncio.run`, so it must not be called
    from inside a running event loop (call the async variant there instead).
    """
    import asyncio

    return asyncio.run(load_extensions_async(**kwargs))

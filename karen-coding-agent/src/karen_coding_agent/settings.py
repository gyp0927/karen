"""Settings files — a karen-scoped subset of pi coding-agent's
`core/settings-manager.ts`.

Locations: global `~/.karen/settings.json` plus project
`<cwd>/.karen/settings.json`; the project file deep-merges over the global
one (pi's `deepMergeSettings`: nested objects merge recursively, scalars and
arrays replace, `defaultTools` gets the special `+name`/`-name` modifier
merge). Like pi, files are **not schema-validated**: unknown keys are
ignored, wrong-typed values are dropped, and an unreadable or malformed file
produces a diagnostic and is skipped. Writes (`update_settings`, behind the
CLI's `/settings` command) merge camelCase keys back into one scope's file
atomically; pi's lock file and modified-field bookkeeping are dropped
(karen runs single-process).

Ported subset (camelCase wire keys, like pi): `defaultProvider`,
`defaultModel`, `shellPath`, `shellCommandPrefix`, `sessionDir`,
`compaction` (`enabled`/`reserveTokens`/`keepRecentTokens`), `retry`
(`enabled`/`maxRetries`/`baseDelayMs`/`maxAgentDelayMs`), `prompts`,
`defaultTools`, plus karen additions `images`, `mcpServers` and `tui`
(`tui: false` keeps `karen` in the plain REPL instead of the alt-screen TUI;
read at startup, like the model and shell settings). The rest of pi's Settings
is extensions/analytics scope and intentionally not ported — including
`retry.provider` (the provider-adapter retry knobs; karen-ai adapters take
those per request).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from karen_ai import DEFAULT_MAX_AGENT_RETRY_DELAY_MS, RetryPolicy
from karen_agent.compaction import CompactionSettings

#: Global settings file (pi: `~/.pi/agent/settings.json`).
DEFAULT_SETTINGS_PATH = Path.home() / ".karen" / "settings.json"
#: Project settings file, relative to the session cwd (pi: `<cwd>/.pi/settings.json`).
PROJECT_SETTINGS_RELATIVE = Path(".karen") / "settings.json"


@dataclass
class SettingsDiagnostic:
    """One settings file problem, surfaced as a startup warning."""

    scope: str  # "global" | "project"
    path: str
    message: str


@dataclass
class Settings:
    """The merged settings subset karen consumes (all optional)."""

    default_provider: Optional[str] = None
    default_model: Optional[str] = None
    shell_path: Optional[str] = None
    shell_command_prefix: Optional[str] = None
    session_dir: Optional[str] = None
    compaction: Optional[Dict[str, Any]] = None  # raw camelCase dict
    retry: Optional[Dict[str, Any]] = None  # raw camelCase dict
    images: Optional[Dict[str, Any]] = None  # raw camelCase dict
    prompts: Optional[List[str]] = None
    default_tools: Optional[List[str]] = None
    mcp_servers: Optional[Dict[str, Any]] = None  # camelCase `mcpServers` → list of server dicts
    #: `tui: false` keeps `karen` in the plain REPL; unset means "auto" (the
    #: TUI whenever both ends are a terminal). An explicit `--tui`/`--repl`
    #: outranks it.
    tui: Optional[bool] = None


def _mcp_servers_from_wire(value: Any) -> Optional[Dict[str, Any]]:
    """`mcpServers` is a `{name: {command?, args?, url?, headers?}}` dict.
    Unknown keys are kept (the manager decides what to do with each server);
    wrong-typed values are dropped."""
    if not isinstance(value, dict):
        return None
    servers: Dict[str, Any] = {}
    for name, spec in value.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            continue
        servers[name] = spec
    return servers or None


@dataclass
class LoadedSettings:
    settings: Settings
    diagnostics: List[SettingsDiagnostic] = field(default_factory=list)


# ---------------------------------------------------------------------------
# merge semantics (pi's deepMergeObjects / deepMergeSettings)
# ---------------------------------------------------------------------------


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, override_value in overrides.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            result[key] = _deep_merge(base_value, override_value)
        else:
            result[key] = override_value
    return result


def _is_tool_modifier(entry: Any) -> bool:
    return isinstance(entry, str) and (entry.startswith("+") or entry.startswith("-"))


def merge_default_tools(base: Optional[List[str]], overrides: Optional[List[str]]) -> Optional[List[str]]:
    """pi's mergeDefaultTools: a list of only `+name`/`-name` entries modifies
    the inherited selection; anything else replaces it."""
    if overrides is None:
        return base
    if not isinstance(base, list) or not isinstance(overrides, list) or not all(
        _is_tool_modifier(entry) for entry in overrides
    ):
        return overrides
    return [*base, *overrides]


def resolve_default_tool_names(entries: List[str], default_names: List[str]) -> List[str]:
    """pi's resolveDefaultTools: plain names replace the defaults, then `+name`
    adds and `-name` removes, in list order."""
    plain = [entry for entry in entries if not _is_tool_modifier(entry)]
    tools = list(plain) if plain or not entries else list(default_names)
    for entry in entries:
        if not _is_tool_modifier(entry):
            continue
        name = entry[1:]
        if entry.startswith("+") and name and name not in tools:
            tools.append(name)
        elif entry.startswith("-") and name in tools:
            tools.remove(name)
    return tools


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _read_settings_file(path: Path, scope: str, diagnostics: List[SettingsDiagnostic]) -> Optional[Dict[str, Any]]:
    try:
        content = path.read_text(encoding="utf-8-sig")  # pi strips a leading BOM
    except FileNotFoundError:
        return None
    except OSError as error:
        diagnostics.append(SettingsDiagnostic(scope, str(path), str(error)))
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        diagnostics.append(SettingsDiagnostic(scope, str(path), f"Invalid JSON: {error}"))
        return None
    if not isinstance(value, dict):
        diagnostics.append(SettingsDiagnostic(scope, str(path), "Settings file must contain a JSON object"))
        return None
    return value


def _as_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _as_str_list(value: Any) -> Optional[List[str]]:
    if not isinstance(value, list):
        return None
    return [entry for entry in value if isinstance(entry, str)]


def _as_dict(value: Any) -> Optional[Dict[str, Any]]:
    return value if isinstance(value, dict) else None


def _as_bool(value: Any) -> Optional[bool]:
    """A real JSON boolean, or None. `1`/`"true"` are wrong-typed and dropped,
    like every other setting (a truthy string must not silently win)."""
    return value if isinstance(value, bool) else None


def _expand_user(value: Optional[str]) -> Optional[str]:
    """pi expands a leading `~` in shellPath; do the same for path settings."""
    return os.path.expanduser(value) if value else value


def _settings_from_wire(merged: Dict[str, Any]) -> Settings:
    return Settings(
        default_provider=_as_str(merged.get("defaultProvider")),
        default_model=_as_str(merged.get("defaultModel")),
        shell_path=_expand_user(_as_str(merged.get("shellPath"))),
        shell_command_prefix=_as_str(merged.get("shellCommandPrefix")),
        session_dir=_expand_user(_as_str(merged.get("sessionDir"))),
        compaction=_as_dict(merged.get("compaction")),
        retry=_as_dict(merged.get("retry")),
        images=_as_dict(merged.get("images")),
        prompts=[_expand_user(entry) for entry in _as_str_list(merged.get("prompts")) or []] or None,
        default_tools=_as_str_list(merged.get("defaultTools")),
        mcp_servers=_mcp_servers_from_wire(merged.get("mcpServers")),
        tui=_as_bool(merged.get("tui")),
    )


def load_settings(
    cwd: str,
    *,
    global_path: Optional[str] = None,
    project_path: Optional[str] = None,
) -> LoadedSettings:
    """Load and merge the global and project settings files for `cwd`.

    The global file defaults to `~/.karen/settings.json` (overridable with the
    `KAREN_SETTINGS_PATH` environment variable — a karen addition, handy for
    tests and sandboxed runs).
    """
    diagnostics: List[SettingsDiagnostic] = []
    global_override = global_path or os.environ.get("KAREN_SETTINGS_PATH")
    layers = [
        _read_settings_file(
            Path(global_override) if global_override else DEFAULT_SETTINGS_PATH, "global", diagnostics
        ),
        _read_settings_file(
            Path(project_path) if project_path else Path(cwd) / PROJECT_SETTINGS_RELATIVE,
            "project",
            diagnostics,
        ),
    ]
    merged: Dict[str, Any] = {}
    for layer in layers:
        if layer is None:
            continue
        default_tools = merge_default_tools(merged.get("defaultTools"), layer.get("defaultTools"))
        merged = _deep_merge(merged, layer)
        if default_tools is not None:
            merged["defaultTools"] = default_tools
    return LoadedSettings(settings=_settings_from_wire(merged), diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# writes (pi's `/settings` persistence, simplified)
# ---------------------------------------------------------------------------


def update_settings(
    updates: Dict[str, Any],
    *,
    scope: str = "global",
    cwd: Optional[str] = None,
    global_path: Optional[str] = None,
    project_path: Optional[str] = None,
) -> Path:
    """Merge `updates` into one settings file and write it back atomically.

    `updates` uses the camelCase wire keys (like the files themselves). Nested
    dict values deep-merge over the file's existing object for that key (pi's
    nested-field persistence); scalars and arrays replace. Unknown keys are
    kept (pi does not schema-validate writes either). The directory is created
    on demand; the file is written via a temp file + `os.replace` so a crash
    mid-write can't corrupt it. Returns the path written.

    pi wraps this in a file lock and persists only fields marked modified;
    karen runs single-process, so a straight read-modify-write is enough.

    A file that exists but cannot be read or parsed is *not* overwritten —
    pi's `save()` returns early when that scope had a load error, so a typo in
    the file never costs the user their settings. `ValueError` is raised
    instead (pi stays quiet; a CLI should say what happened).
    """
    if scope not in ("global", "project"):
        raise ValueError(f"scope must be 'global' or 'project', got {scope!r}")
    if scope == "global":
        override = global_path or os.environ.get("KAREN_SETTINGS_PATH")
        path = Path(override) if override else DEFAULT_SETTINGS_PATH
    else:
        if cwd is None and project_path is None:
            raise ValueError("project scope requires cwd or project_path")
        path = Path(project_path) if project_path else Path(cwd) / PROJECT_SETTINGS_RELATIVE

    diagnostics: List[SettingsDiagnostic] = []
    current = _read_settings_file(path, scope, diagnostics)
    if diagnostics:
        raise ValueError(f"refusing to overwrite {diagnostics[0].path}: {diagnostics[0].message}")
    merged = _deep_merge(current or {}, updates)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp_path, path)
    return path


def compaction_settings_from_wire(value: Dict[str, Any]) -> CompactionSettings:
    """Tolerantly map a settings `compaction` dict onto CompactionSettings."""
    settings = CompactionSettings()
    enabled = value.get("enabled")
    if isinstance(enabled, bool):
        settings.enabled = enabled
    reserve_tokens = value.get("reserveTokens")
    if isinstance(reserve_tokens, (int, float)) and not isinstance(reserve_tokens, bool):
        settings.reserve_tokens = int(reserve_tokens)
    keep_recent_tokens = value.get("keepRecentTokens")
    if isinstance(keep_recent_tokens, (int, float)) and not isinstance(keep_recent_tokens, bool):
        settings.keep_recent_tokens = int(keep_recent_tokens)
    return settings


#: pi's coding-agent retry defaults (`SettingsManager`: 3 retries, 2s base).
DEFAULT_RETRY_POLICY = RetryPolicy(
    enabled=True,
    max_retries=3,
    base_delay_ms=2000,
    max_agent_delay_ms=DEFAULT_MAX_AGENT_RETRY_DELAY_MS,
)


def mcp_server_configs_from_settings(settings: "Settings") -> List[Any]:
    """Turn a `Settings.mcp_servers` dict into `McpServerConfig` objects.

    Each entry in the settings file's `mcpServers` maps a server name to a
    spec (`command`/`args` for stdio, `url`/`headers` for HTTP). The import of
    `McpServerConfig` is lazy so this function is usable before `karen_mcp`
    is installed in a minimal environment.
    """
    if not settings.mcp_servers:
        return []
    from karen_coding_agent.mcp import McpServerConfig

    configs: List[McpServerConfig] = []
    for name, spec in settings.mcp_servers.items():
        if not isinstance(spec, dict):
            continue
        config = McpServerConfig(name=name)
        if isinstance(spec.get("command"), str):
            config.command = spec["command"]
            args = spec.get("args")
            config.args = [entry for entry in args if isinstance(entry, str)] if isinstance(args, list) else None
        if isinstance(spec.get("url"), str):
            config.url = spec["url"]
            headers = spec.get("headers")
            config.headers = {k: v for k, v in headers.items() if isinstance(k, str) and isinstance(v, str)} if isinstance(headers, dict) else None
        configs.append(config)
    return configs


def _as_count(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value)


def image_auto_resize(value: Optional[Dict[str, Any]]) -> bool:
    """pi's `getImageAutoResize`: `settings.images?.autoResize ?? true`."""
    if not isinstance(value, dict):
        return True
    enabled = value.get("autoResize")
    return enabled if isinstance(enabled, bool) else True


def image_block_images(value: Optional[Dict[str, Any]]) -> bool:
    """pi's `images.blockImages` (default false): strip images in convert_to_llm."""
    if not isinstance(value, dict):
        return False
    blocked = value.get("blockImages")
    return blocked if isinstance(blocked, bool) else False


def retry_policy_from_wire(value: Optional[Dict[str, Any]]) -> Optional[RetryPolicy]:
    """Map a settings `retry` dict onto a RetryPolicy, falling back per field
    to pi's coding-agent defaults (enabled, 3 retries, 2s base, 60s cap)."""
    if value is None:
        return None
    enabled = value.get("enabled")
    max_retries = _as_count(value.get("maxRetries"))
    base_delay_ms = _as_count(value.get("baseDelayMs"))
    max_agent_delay_ms = _as_count(value.get("maxAgentDelayMs"))
    return RetryPolicy(
        enabled=enabled if isinstance(enabled, bool) else DEFAULT_RETRY_POLICY.enabled,
        max_retries=max_retries if max_retries is not None else DEFAULT_RETRY_POLICY.max_retries,
        base_delay_ms=base_delay_ms if base_delay_ms is not None else DEFAULT_RETRY_POLICY.base_delay_ms,
        max_agent_delay_ms=(
            max_agent_delay_ms
            if max_agent_delay_ms is not None
            else DEFAULT_RETRY_POLICY.max_agent_delay_ms
        ),
    )


__all__ = [
    "DEFAULT_RETRY_POLICY",
    "DEFAULT_SETTINGS_PATH",
    "PROJECT_SETTINGS_RELATIVE",
    "LoadedSettings",
    "Settings",
    "SettingsDiagnostic",
    "compaction_settings_from_wire",
    "image_auto_resize",
    "image_block_images",
    "load_settings",
    "merge_default_tools",
    "mcp_server_configs_from_settings",
    "resolve_default_tool_names",
    "retry_policy_from_wire",
    "update_settings",
]

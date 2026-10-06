"""Settings files — a karen-scoped subset of pi coding-agent's
`core/settings-manager.ts`.

Locations: global `~/.karen/settings.json` plus project
`<cwd>/.karen/settings.json`; the project file deep-merges over the global
one (pi's `deepMergeSettings`: nested objects merge recursively, scalars and
arrays replace, `defaultTools` gets the special `+name`/`-name` modifier
merge). Like pi, files are **not schema-validated**: unknown keys are
ignored, wrong-typed values are dropped, and an unreadable or malformed file
produces a diagnostic and is skipped. Writes (pi's `/settings` commands) are
not ported.

Ported subset (camelCase wire keys, like pi): `defaultProvider`,
`defaultModel`, `shellPath`, `shellCommandPrefix`, `sessionDir`,
`compaction` (`enabled`/`reserveTokens`/`keepRecentTokens`), `prompts`,
`defaultTools`. The rest of pi's Settings is TUI/extensions/analytics scope
and intentionally not ported.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    prompts: Optional[List[str]] = None
    default_tools: Optional[List[str]] = None


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
        prompts=[_expand_user(entry) for entry in _as_str_list(merged.get("prompts")) or []] or None,
        default_tools=_as_str_list(merged.get("defaultTools")),
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


__all__ = [
    "DEFAULT_SETTINGS_PATH",
    "PROJECT_SETTINGS_RELATIVE",
    "LoadedSettings",
    "Settings",
    "SettingsDiagnostic",
    "compaction_settings_from_wire",
    "load_settings",
    "merge_default_tools",
    "resolve_default_tool_names",
]

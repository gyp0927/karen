"""Tests for the settings files (pi's settings-manager subset) and their
wiring into the CLI: defaults, tool selection, compaction, sessions dir."""

import json
import os
import sys

import pytest

from karen_ai.providers import faux_assistant_message
from karen_agent.compaction import CompactionSettings
from karen_agent.tools.local_shell import resolve_shell_config
from karen_coding_agent import cli as karen_cli
from karen_coding_agent.settings import (
    compaction_settings_from_wire,
    load_settings,
    merge_default_tools,
    resolve_default_tool_names,
    retry_policy_from_wire,
)
from karen_coding_agent.tools import create_default_tools
from test_cli import _faux_factory

try:
    resolve_shell_config()
    HAS_BASH = True
except Exception:
    HAS_BASH = False


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")


@pytest.fixture
def isolated_global(tmp_path, monkeypatch):
    """Point the global settings file at a nonexistent path."""
    monkeypatch.setenv("KAREN_SETTINGS_PATH", str(tmp_path / "no-global-settings.json"))


# ---------------------------------------------------------------------------
# loading + merging
# ---------------------------------------------------------------------------


def test_missing_files_yield_empty_settings(tmp_path, isolated_global):
    loaded = load_settings(str(tmp_path))
    assert loaded.settings.default_model is None
    assert loaded.diagnostics == []


def test_project_deep_merges_over_global(tmp_path):
    global_path = tmp_path / "global.json"
    _write(global_path, {
        "defaultModel": "a",
        "defaultProvider": "p1",
        "compaction": {"reserveTokens": 1000, "keepRecentTokens": 500},
        "unknownFutureKey": {"nested": 1},
    })
    _write(tmp_path / ".karen" / "settings.json", {
        "defaultModel": "b",
        "compaction": {"keepRecentTokens": 800},
    })
    loaded = load_settings(str(tmp_path), global_path=str(global_path))
    assert loaded.settings.default_model == "b"  # project wins on scalars
    assert loaded.settings.default_provider == "p1"  # inherited
    # nested objects merge recursively
    assert loaded.settings.compaction == {"reserveTokens": 1000, "keepRecentTokens": 800}
    assert loaded.diagnostics == []


def test_malformed_file_is_diagnosed_and_skipped(tmp_path):
    global_path = tmp_path / "global.json"
    _write(global_path, "{ not json")
    _write(tmp_path / ".karen" / "settings.json", {"defaultModel": "b"})
    loaded = load_settings(str(tmp_path), global_path=str(global_path))
    assert loaded.settings.default_model == "b"  # the healthy file still applies
    assert len(loaded.diagnostics) == 1
    diagnostic = loaded.diagnostics[0]
    assert diagnostic.scope == "global"
    assert diagnostic.path == str(global_path)
    assert "Invalid JSON" in diagnostic.message


def test_non_object_file_is_diagnosed(tmp_path):
    global_path = tmp_path / "global.json"
    _write(global_path, '["not", "an", "object"]')
    loaded = load_settings(str(tmp_path), global_path=str(global_path))
    assert "JSON object" in loaded.diagnostics[0].message


def test_bom_is_tolerated(tmp_path):
    global_path = tmp_path / "global.json"
    global_path.write_bytes(b'\xef\xbb\xbf{"defaultModel": "bom-model"}')
    loaded = load_settings(str(tmp_path), global_path=str(global_path))
    assert loaded.settings.default_model == "bom-model"
    assert loaded.diagnostics == []


def test_wrong_typed_values_are_dropped(tmp_path):
    global_path = tmp_path / "global.json"
    _write(global_path, {"defaultModel": 42, "prompts": "not-a-list", "shellPath": True})
    loaded = load_settings(str(tmp_path), global_path=str(global_path))
    assert loaded.settings.default_model is None
    assert loaded.settings.prompts is None
    assert loaded.settings.shell_path is None
    assert loaded.diagnostics == []


def test_tilde_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # ntpath.expanduser reads USERPROFILE
    global_path = tmp_path / "global.json"
    _write(global_path, {"shellPath": "~/bin/bash", "sessionDir": "~/sessions"})
    loaded = load_settings(str(tmp_path), global_path=str(global_path))
    assert os.path.normpath(loaded.settings.shell_path) == str(tmp_path / "bin" / "bash")
    assert os.path.normpath(loaded.settings.session_dir) == str(tmp_path / "sessions")


# ---------------------------------------------------------------------------
# defaultTools semantics (pi's mergeDefaultTools / resolveDefaultTools)
# ---------------------------------------------------------------------------


def test_default_tools_modifier_lists_merge_across_layers(tmp_path):
    global_path = tmp_path / "global.json"
    _write(global_path, {"defaultTools": ["read", "bash"]})
    _write(tmp_path / ".karen" / "settings.json", {"defaultTools": ["-bash", "+ls"]})
    loaded = load_settings(str(tmp_path), global_path=str(global_path))
    assert loaded.settings.default_tools == ["read", "bash", "-bash", "+ls"]
    assert resolve_default_tool_names(loaded.settings.default_tools, []) == ["read", "ls"]


def test_default_tools_plain_list_replaces(tmp_path):
    global_path = tmp_path / "global.json"
    _write(global_path, {"defaultTools": ["read"]})
    _write(tmp_path / ".karen" / "settings.json", {"defaultTools": ["write", "edit"]})
    loaded = load_settings(str(tmp_path), global_path=str(global_path))
    assert loaded.settings.default_tools == ["write", "edit"]


def test_merge_default_tools_malformed_overrides_replace():
    assert merge_default_tools(["read"], "junk") == "junk"
    assert merge_default_tools(None, ["+ls"]) == ["+ls"]


def test_resolve_default_tool_names_modifier_only_starts_from_defaults():
    defaults = ["read", "bash", "grep"]
    assert resolve_default_tool_names(["-bash"], defaults) == ["read", "grep"]
    assert resolve_default_tool_names(["+ls", "-grep"], defaults) == ["read", "bash", "ls"]
    assert resolve_default_tool_names([], defaults) == []
    assert resolve_default_tool_names(["read"], defaults) == ["read"]


# ---------------------------------------------------------------------------
# compaction mapping
# ---------------------------------------------------------------------------


def test_compaction_settings_from_wire_partial_and_tolerant():
    settings = compaction_settings_from_wire({"reserveTokens": 1000, "enabled": False, "keepRecentTokens": "x"})
    assert settings.reserve_tokens == 1000
    assert settings.enabled is False
    assert settings.keep_recent_tokens == CompactionSettings().keep_recent_tokens


# ---------------------------------------------------------------------------
# retry mapping
# ---------------------------------------------------------------------------


def test_retry_policy_from_wire_absent_and_partial():
    assert retry_policy_from_wire(None) is None
    empty = retry_policy_from_wire({})
    assert (empty.enabled, empty.max_retries, empty.base_delay_ms, empty.max_agent_delay_ms) == (
        True,
        3,
        2000,
        60000,
    )
    partial = retry_policy_from_wire({"baseDelayMs": 50})
    assert (partial.enabled, partial.max_retries, partial.base_delay_ms, partial.max_agent_delay_ms) == (
        True,
        3,
        50,
        60000,
    )


def test_retry_policy_from_wire_reads_every_field():
    policy = retry_policy_from_wire(
        {"enabled": False, "maxRetries": 5, "baseDelayMs": 10, "maxAgentDelayMs": 20}
    )
    assert (policy.enabled, policy.max_retries, policy.base_delay_ms, policy.max_agent_delay_ms) == (
        False,
        5,
        10,
        20,
    )


def test_retry_policy_from_wire_drops_wrong_types():
    policy = retry_policy_from_wire(
        {"enabled": "yes", "maxRetries": 2.5, "baseDelayMs": True, "maxAgentDelayMs": "60000"}
    )
    assert (policy.enabled, policy.max_retries, policy.base_delay_ms, policy.max_agent_delay_ms) == (
        True,
        3,
        2000,
        60000,
    )


def test_retry_settings_merge_and_reach_the_session(tmp_path, isolated_global):
    _write(tmp_path / ".karen" / "settings.json", {"retry": {"maxRetries": 7}})
    loaded = load_settings(str(tmp_path))
    assert loaded.settings.retry == {"maxRetries": 7}
    assert retry_policy_from_wire(loaded.settings.retry).max_retries == 7


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_uses_settings_default_provider_and_model(tmp_path, monkeypatch, capsys, isolated_global):
    monkeypatch.setattr(karen_cli, "build_models", _faux_factory([faux_assistant_message("pong")]))
    monkeypatch.setenv("KAREN_SESSIONS_ROOT", str(tmp_path / "sessions"))
    _write(tmp_path / ".karen" / "settings.json", {"defaultProvider": "faux", "defaultModel": "faux-1"})
    monkeypatch.delenv("KAREN_PROVIDER", raising=False)
    monkeypatch.delenv("KAREN_MODEL", raising=False)

    exit_code = karen_cli.main(["-p", "ping", "--new", "--cwd", str(tmp_path)])

    assert exit_code == 0
    assert capsys.readouterr().out == "pong\n"


async def test_cli_settings_default_tools_selects_subset(tmp_path, monkeypatch, isolated_global):
    monkeypatch.setattr(karen_cli, "build_models", _faux_factory([faux_assistant_message("pong")]))
    _write(tmp_path / ".karen" / "settings.json", {"defaultTools": ["read", "grep", "-grep", "+ls"]})
    cli = karen_cli.KarenCli(cwd=str(tmp_path), model_id="faux-1", fresh=True, provider="faux")
    await cli._open_session(fresh=True)
    try:
        assert [tool.name for tool in cli.session.tools] == ["read", "ls"]
    finally:
        await cli.session.close()


async def test_cli_settings_session_dir_is_honored(tmp_path, monkeypatch, isolated_global):
    monkeypatch.setattr(karen_cli, "build_models", _faux_factory([faux_assistant_message("pong")]))
    monkeypatch.delenv("KAREN_SESSIONS_ROOT", raising=False)
    custom_dir = tmp_path / "custom-sessions"
    _write(tmp_path / ".karen" / "settings.json", {"sessionDir": str(custom_dir)})
    cli = karen_cli.KarenCli(cwd=str(tmp_path), model_id="faux-1", fresh=True, provider="faux")
    await cli._open_session(fresh=True)
    try:
        assert cli.session.sessions_root == str(custom_dir)
        assert list(custom_dir.rglob("*.jsonl")), "session file should live under the settings sessionDir"
    finally:
        await cli.session.close()


async def test_cli_settings_compaction_reaches_session(tmp_path, monkeypatch, isolated_global):
    monkeypatch.setattr(karen_cli, "build_models", _faux_factory([faux_assistant_message("pong")]))
    _write(tmp_path / ".karen" / "settings.json", {"compaction": {"reserveTokens": 1234}})
    cli = karen_cli.KarenCli(cwd=str(tmp_path), model_id="faux-1", fresh=True, provider="faux")
    await cli._open_session(fresh=True)
    try:
        assert cli.session.settings.reserve_tokens == 1234
    finally:
        await cli.session.close()


@pytest.mark.skipif(not HAS_BASH, reason="no bash shell available on this machine")
async def test_shell_command_prefix_reaches_bash_tool(tmp_path):
    tools = create_default_tools(str(tmp_path), shell_command_prefix="echo from-prefix")
    bash = next(tool for tool in tools if tool.name == "bash")
    result = await bash.execute("t1", {"command": "echo from-command"}, None, None)
    assert result.content[0].text == "from-prefix\nfrom-command\n"

"""Contract tests for the Neo4j memory post_setup wizard."""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import patch

from neo4j_memory import Neo4jMemoryProvider
from neo4j_memory import cli as cli_mod


def _inputs(*values: str) -> io.StringIO:
    return io.StringIO("\n".join(values) + "\n")


def test_provider_exposes_post_setup_and_config_schema(tmp_path):
    provider = Neo4jMemoryProvider()

    schema = provider.get_config_schema()
    keys = {entry["key"] for entry in schema}
    assert {"uri", "user", "password", "database", "llm", "embedding"}.issubset(keys)
    assert (
        next(entry for entry in schema if entry["key"] == "password")["secret"] is True
    )

    with patch("neo4j_memory.cli.cmd_setup", return_value={"ok": True}) as setup:
        result = provider.post_setup(str(tmp_path), {"seed": "value"})

    setup.assert_called_once()
    assert result == {"ok": True}


def test_cmd_setup_rejects_invalid_neo4j_uri_before_smoke_test(tmp_path):
    captured = io.StringIO()
    inputs = _inputs(
        "http://not-a-bolt-url",
        "bolt://localhost:7687",
        "neo4j",
        "pass",
        "neo4j",
        "anthropic/claude-3-5-sonnet-latest",
        "openai/text-embedding-3-small",
    )

    with (
        patch("neo4j_memory.cli._get_hermes_home", return_value=tmp_path),
        patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
        patch("neo4j_memory.cli._detect_llm_keys", return_value={}),
        patch("neo4j_memory.cli._smoke_test", return_value=(True, "")) as smoke,
        patch("sys.stdin", inputs),
        patch("sys.stdout", captured),
    ):
        result = cli_mod.cmd_setup(SimpleNamespace())

    assert result is not None
    assert smoke.call_count == 1
    assert result["uri"] == "bolt://localhost:7687"
    assert "Invalid Neo4j URI" in captured.getvalue()


def test_cmd_setup_redacts_password_from_smoke_test_errors(tmp_path):
    captured = io.StringIO()
    inputs = _inputs(
        "bolt://localhost:7687",
        "neo4j",
        "ultra-secret-password",
        "neo4j",
        "anthropic/claude-3-5-sonnet-latest",
        "openai/text-embedding-3-small",
        "n",
    )

    with (
        patch("neo4j_memory.cli._get_hermes_home", return_value=tmp_path),
        patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
        patch("neo4j_memory.cli._detect_llm_keys", return_value={}),
        patch(
            "neo4j_memory.cli._smoke_test",
            return_value=(False, "auth failed for ultra-secret-password"),
        ),
        patch("sys.stdin", inputs),
        patch("sys.stdout", captured),
    ):
        result = cli_mod.cmd_setup(SimpleNamespace())

    assert result is None
    output = captured.getvalue()
    assert "ultra-secret-password" not in output
    assert "[redacted]" in output


def test_cmd_setup_sets_memory_provider_in_hermes_config_yaml(tmp_path):
    captured = io.StringIO()
    inputs = _inputs(
        "bolt://localhost:7687",
        "neo4j",
        "pass",
        "neo4j",
        "anthropic/claude-3-5-sonnet-latest",
        "openai/text-embedding-3-small",
    )

    with (
        patch("neo4j_memory.cli._get_hermes_home", return_value=tmp_path),
        patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
        patch("neo4j_memory.cli._detect_llm_keys", return_value={}),
        patch("neo4j_memory.cli._smoke_test", return_value=(True, "")),
        patch("sys.stdin", inputs),
        patch("sys.stdout", captured),
    ):
        result = cli_mod.cmd_setup(SimpleNamespace())

    assert result is not None
    config_yaml = tmp_path / "config.yaml"
    assert config_yaml.exists()
    assert "provider: neo4j" in config_yaml.read_text()

"""Unit tests for neo4j_memory/cli.py::cmd_setup()."""

from __future__ import annotations

import io
import json
import os
from types import SimpleNamespace
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(*lines: str) -> io.StringIO:
    """Build a StringIO that feeds successive readline() calls."""
    return io.StringIO("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestCmdSetupHappyPath:
    def test_returns_config_dict_and_writes_file(self, tmp_path):
        """cmd_setup returns the written config dict on success."""
        config_file = tmp_path / "neo4j-memory.json"

        inputs = _make_inputs(
            "bolt://localhost:7687",  # URI
            "neo4j",  # username
            "s3cr3t",  # password
            "neo4j",  # database
            "anthropic/claude-3-5-sonnet-latest",  # LLM
            "openai/text-embedding-3-small",  # embedding
        )

        with (
            patch("neo4j_memory.cli._get_hermes_home", return_value=tmp_path),
            patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
            patch("neo4j_memory.cli._smoke_test", return_value=(True, "")),
            patch("sys.stdin", inputs),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            from neo4j_memory.cli import cmd_setup

            result = cmd_setup(SimpleNamespace())

        assert result is not None
        assert result["uri"] == "bolt://localhost:7687"
        assert result["user"] == "neo4j"
        assert result["password"] == "s3cr3t"
        assert result["database"] == "neo4j"
        assert result["llm"] == "anthropic/claude-3-5-sonnet-latest"
        assert result["embedding"] == "openai/text-embedding-3-small"

        # Verify file was actually written
        assert config_file.exists()
        on_disk = json.loads(config_file.read_text())
        assert on_disk["uri"] == "bolt://localhost:7687"
        assert on_disk["password"] == "s3cr3t"

    def test_returns_none_when_driver_not_installed_and_user_declines(self, tmp_path):
        """Returns None when driver unavailable and user declines install."""
        with (
            patch("neo4j_memory.cli._get_hermes_home", return_value=tmp_path),
            patch("neo4j_memory.cli._ensure_driver_installed", return_value=False),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            from neo4j_memory.cli import cmd_setup

            result = cmd_setup(SimpleNamespace())

        assert result is None


# ---------------------------------------------------------------------------
# Connection failure → retry → success
# ---------------------------------------------------------------------------


class TestConnectionRetry:
    def test_reprompts_on_failure_then_succeeds(self, tmp_path):
        """On first connection failure the wizard loops back; success on retry."""
        # First attempt: bad password that fails
        # Second attempt: correct password that passes
        inputs = _make_inputs(
            # --- First attempt ---
            "bolt://localhost:7687",  # URI
            "neo4j",  # username
            "wrong-pass",  # bad password
            "neo4j",  # database
            "anthropic/claude-3-5-sonnet-latest",
            "openai/text-embedding-3-small",
            "y",  # retry?
            # --- Second attempt ---
            "bolt://localhost:7687",
            "neo4j",
            "correct-pass",
            "neo4j",
            "anthropic/claude-3-5-sonnet-latest",
            "openai/text-embedding-3-small",
        )

        smoke_results = [(False, "Auth failed"), (True, "")]

        with (
            patch("neo4j_memory.cli._get_hermes_home", return_value=tmp_path),
            patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
            patch("neo4j_memory.cli._smoke_test", side_effect=smoke_results),
            patch("sys.stdin", inputs),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            from neo4j_memory.cli import cmd_setup

            result = cmd_setup(SimpleNamespace())

        assert result is not None
        assert result["password"] == "correct-pass"

    def test_aborts_when_user_declines_retry(self, tmp_path):
        """Returns None when the user declines to retry after a failed connection."""
        inputs = _make_inputs(
            "bolt://bad-host:7687",
            "neo4j",
            "pass",
            "neo4j",
            "anthropic/claude-3-5-sonnet-latest",
            "openai/text-embedding-3-small",
            "n",  # decline retry
        )

        with (
            patch("neo4j_memory.cli._get_hermes_home", return_value=tmp_path),
            patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
            patch(
                "neo4j_memory.cli._smoke_test",
                return_value=(False, "Connection refused"),
            ),
            patch("sys.stdin", inputs),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            from neo4j_memory.cli import cmd_setup

            result = cmd_setup(SimpleNamespace())

        assert result is None
        # No config file should have been written
        assert not (tmp_path / "neo4j-memory.json").exists()


# ---------------------------------------------------------------------------
# LLM key detection
# ---------------------------------------------------------------------------


class TestLLMKeyWarning:
    def test_warning_printed_when_no_llm_key_present(self, tmp_path):
        """When no LLM API keys are found, a warning is emitted."""
        # Empty .env and no relevant env vars
        env_file = tmp_path / ".env"
        env_file.write_text("")

        inputs = _make_inputs(
            "bolt://localhost:7687",
            "neo4j",
            "pass",
            "neo4j",
            "some/model",
            "openai/text-embedding-3-small",
        )

        captured = io.StringIO()

        with (
            patch("neo4j_memory.cli._get_hermes_home", return_value=tmp_path),
            patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
            patch("neo4j_memory.cli._smoke_test", return_value=(True, "")),
            patch("sys.stdin", inputs),
            patch("sys.stdout", captured),
            patch.dict(os.environ, {}, clear=False),
        ):
            # Remove any LLM keys that might be in actual environment
            clean_env = {
                k: v
                for k, v in os.environ.items()
                if k
                not in {
                    "ANTHROPIC_API_KEY",
                    "OPENAI_API_KEY",
                    "GROQ_API_KEY",
                    "COHERE_API_KEY",
                    "TOGETHER_API_KEY",
                    "MISTRAL_API_KEY",
                    "LM_API_KEY",
                }
            }
            with patch.dict(os.environ, clean_env, clear=True):
                from neo4j_memory import cli as cli_mod  # noqa: F401

                # Reset the module's detection function to use cleaned env
                with patch("neo4j_memory.cli._detect_llm_keys", return_value={}):
                    from neo4j_memory.cli import cmd_setup

                    cmd_setup(SimpleNamespace())

        output = captured.getvalue()
        assert "WARNING" in output or "No LLM API keys" in output

    def test_detected_keys_shown_in_output(self, tmp_path):
        """When LLM keys are found, detected providers appear in wizard output."""
        inputs = _make_inputs(
            "bolt://localhost:7687",
            "neo4j",
            "pass",
            "neo4j",
            "anthropic/claude-3-5-sonnet-latest",
            "openai/text-embedding-3-small",
        )

        captured = io.StringIO()

        with (
            patch("neo4j_memory.cli._get_hermes_home", return_value=tmp_path),
            patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
            patch("neo4j_memory.cli._smoke_test", return_value=(True, "")),
            patch(
                "neo4j_memory.cli._detect_llm_keys",
                return_value={"anthropic": "ANTHROPIC_API_KEY"},
            ),
            patch("sys.stdin", inputs),
            patch("sys.stdout", captured),
        ):
            from neo4j_memory.cli import cmd_setup

            cmd_setup(SimpleNamespace())

        output = captured.getvalue()
        assert "anthropic" in output.lower() or "ANTHROPIC_API_KEY" in output


# ---------------------------------------------------------------------------
# Defaults and edge cases
# ---------------------------------------------------------------------------


class TestInputEdgeCases:
    def test_accepts_empty_input_uses_defaults(self, tmp_path):
        """Empty input on every prompt should use defaults."""
        inputs = _make_inputs(
            "",  # URI -> default bolt://localhost:7687
            "",  # username -> default neo4j
            "s3cr3t",  # password has no default — must be supplied
            "",  # database -> default neo4j
            "",  # LLM -> default
            "",  # embedding -> default
        )

        with (
            patch("neo4j_memory.cli._get_hermes_home", return_value=tmp_path),
            patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
            patch("neo4j_memory.cli._smoke_test", return_value=(True, "")),
            patch("neo4j_memory.cli._detect_llm_keys", return_value={}),
            patch("sys.stdin", inputs),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            from neo4j_memory.cli import cmd_setup

            result = cmd_setup(SimpleNamespace())

        assert result is not None
        assert result["uri"] == "bolt://localhost:7687"
        assert result["user"] == "neo4j"
        assert result["database"] == "neo4j"

    def test_existing_config_used_as_defaults(self, tmp_path):
        """Pre-existing config values appear as defaults."""
        existing = {
            "uri": "bolt://db.internal:7687",
            "user": "hermes_user",
            "password": "existing-pass",
            "database": "hermes_db",
            "llm": "openai/gpt-4o",
            "embedding": "openai/text-embedding-3-small",
        }
        (tmp_path / "neo4j-memory.json").write_text(json.dumps(existing))

        # Accept all defaults (empty input)
        inputs = _make_inputs("", "", "", "", "", "")

        with (
            patch("neo4j_memory.cli._get_hermes_home", return_value=tmp_path),
            patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
            patch("neo4j_memory.cli._smoke_test", return_value=(True, "")),
            patch("neo4j_memory.cli._detect_llm_keys", return_value={}),
            patch("sys.stdin", inputs),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            from neo4j_memory.cli import cmd_setup

            result = cmd_setup(SimpleNamespace())

        # Empty password input with existing password should keep existing
        assert result is not None
        assert result["password"] == "existing-pass"
        assert result["uri"] == "bolt://db.internal:7687"

    def test_config_not_written_on_abort(self, tmp_path):
        """When the user aborts after a smoke-test failure, no file is written."""
        inputs = _make_inputs(
            "bolt://nowhere:7687",
            "neo4j",
            "any-pass",
            "neo4j",
            "some/model",
            "some/embedding",
            "n",  # don't retry
        )

        with (
            patch("neo4j_memory.cli._get_hermes_home", return_value=tmp_path),
            patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
            patch("neo4j_memory.cli._smoke_test", return_value=(False, "Refused")),
            patch("sys.stdin", inputs),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            from neo4j_memory.cli import cmd_setup

            result = cmd_setup(SimpleNamespace())

        assert result is None
        assert not (tmp_path / "neo4j-memory.json").exists()


# ---------------------------------------------------------------------------
# get_config_schema
# ---------------------------------------------------------------------------


class TestGetConfigSchema:
    def test_schema_has_required_keys(self):
        from neo4j_memory.cli import get_config_schema

        schema = get_config_schema()
        keys = {entry["key"] for entry in schema}
        assert "uri" in keys
        assert "user" in keys
        assert "password" in keys
        assert "database" in keys

    def test_password_is_secret(self):
        from neo4j_memory.cli import get_config_schema

        schema = get_config_schema()
        pw = next(e for e in schema if e["key"] == "password")
        assert pw.get("secret") is True

    def test_env_var_annotations_present(self):
        from neo4j_memory.cli import get_config_schema

        schema = get_config_schema()
        for entry in schema:
            if entry["key"] in {"uri", "user", "password", "database"}:
                assert "env_var" in entry, f"Missing env_var on {entry['key']}"

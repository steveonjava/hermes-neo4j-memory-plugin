"""Integration tests for the Neo4j memory setup wizard (cli.cmd_setup).

Tests run the wizard end-to-end by mocking prompts, env, and Hermes keys.
They are skipped automatically in unit-test-only runs.

Run with:
    docker compose -f docker-compose.test.yml up -d
    NEO4J_TEST_URI=bolt://localhost:7687 \\
    NEO4J_TEST_PASSWORD=testpassword \\
    pytest tests/integration/test_setup_wizard.py --run-integration -v

Or set NEO4J_TEST_URI to enable without the flag.

No real user secrets are required — all tests use the ephemeral
docker-compose.test.yml Neo4j instance (neo4j/testpassword).
"""

from __future__ import annotations

import io
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

NEO4J_TEST_URI = os.environ.get("NEO4J_TEST_URI", "")


def pytest_addoption(parser):
    """Register --run-integration flag (safe to call multiple times)."""
    try:
        parser.addoption(
            "--run-integration",
            action="store_true",
            default=False,
            help="Run integration tests (requires a live Neo4j instance)",
        )
    except ValueError:
        pass  # Already registered by another conftest / test file


def _integration_enabled(config) -> bool:
    return bool(
        NEO4J_TEST_URI
        or (
            hasattr(config, "getoption")
            and config.getoption("--run-integration", default=False)
        )
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def neo4j_test_uri():
    """Return the Neo4j test URI from environment or docker-compose default."""
    return NEO4J_TEST_URI or "bolt://localhost:7687"


@pytest.fixture
def clean_config(tmp_path):
    """Yield a fresh hermes_home for each test, cleaned up afterwards."""
    yield tmp_path
    cfg_file = tmp_path / "neo4j-memory.json"
    if cfg_file.exists():
        cfg_file.unlink()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wizard_inputs(
    uri,
    user,
    password,
    database="neo4j",
    llm="anthropic/claude-3-5-sonnet-latest",
    embedding="openai/text-embedding-3-small",
):
    """Build a newline-delimited stdin string for cmd_setup prompts."""
    return "\n".join([uri, user, password, database, llm, embedding, ""]) + "\n"


def _run_wizard(clean_config, uri, user, password, smoke_ok=True, smoke_err=""):
    """Run cmd_setup with patched stdin, stdout, hermes_home, and smoke test."""
    inputs = _wizard_inputs(uri, user, password)

    # Patch _smoke_test so no real network call is made unless overridden.
    smoke_return = (smoke_ok, smoke_err)

    with (
        patch("neo4j_memory.cli._get_hermes_home", return_value=clean_config),
        patch("neo4j_memory.cli._smoke_test", return_value=smoke_return),
        patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
        patch("sys.stdin", io.StringIO(inputs)),
        patch("sys.stdout", new_callable=io.StringIO),
    ):
        from neo4j_memory.cli import cmd_setup

        return cmd_setup(SimpleNamespace())


# ---------------------------------------------------------------------------
# 1. Happy-path — no Neo4j required (smoke test mocked)
# ---------------------------------------------------------------------------


def test_wizard_writes_config_file(clean_config):
    """Wizard must write neo4j-memory.json to hermes_home on success."""
    result = _run_wizard(clean_config, "bolt://localhost:7687", "neo4j", "secret")
    assert result is not None, "cmd_setup returned None — wizard aborted unexpectedly"
    config_file = clean_config / "neo4j-memory.json"
    assert config_file.exists(), "Config file was not written to hermes_home"


def test_wizard_config_contains_correct_uri(clean_config):
    """Written config must contain the URI supplied to the wizard."""
    test_uri = "bolt://testhost:7687"
    _run_wizard(clean_config, test_uri, "neo4j", "pass")
    on_disk = json.loads((clean_config / "neo4j-memory.json").read_text())
    assert on_disk["uri"] == test_uri


def test_wizard_config_contains_user(clean_config):
    """Written config must contain the username supplied to the wizard."""
    _run_wizard(clean_config, "bolt://localhost:7687", "alice", "pass")
    on_disk = json.loads((clean_config / "neo4j-memory.json").read_text())
    assert on_disk["user"] == "alice"


def test_wizard_config_persists_password(clean_config):
    """Written config must persist the password (no masking in JSON file)."""
    _run_wizard(clean_config, "bolt://localhost:7687", "neo4j", "supersecret")
    on_disk = json.loads((clean_config / "neo4j-memory.json").read_text())
    assert on_disk["password"] == "supersecret"


def test_wizard_config_default_database(clean_config):
    """Accepting the default database prompt must write 'neo4j'."""
    _run_wizard(clean_config, "bolt://localhost:7687", "neo4j", "pass")
    on_disk = json.loads((clean_config / "neo4j-memory.json").read_text())
    assert on_disk["database"] == "neo4j"


def test_wizard_config_contains_llm(clean_config):
    """Written config must contain the llm field."""
    _run_wizard(clean_config, "bolt://localhost:7687", "neo4j", "pass")
    on_disk = json.loads((clean_config / "neo4j-memory.json").read_text())
    assert "llm" in on_disk
    assert on_disk["llm"]  # must be non-empty


def test_wizard_config_contains_embedding(clean_config):
    """Written config must contain the embedding field."""
    _run_wizard(clean_config, "bolt://localhost:7687", "neo4j", "pass")
    on_disk = json.loads((clean_config / "neo4j-memory.json").read_text())
    assert "embedding" in on_disk
    assert on_disk["embedding"]


def test_wizard_returns_dict_on_success(clean_config):
    """cmd_setup must return the config dict (not None) on success."""
    result = _run_wizard(clean_config, "bolt://localhost:7687", "neo4j", "pass")
    assert isinstance(result, dict)
    assert result["uri"] == "bolt://localhost:7687"


# ---------------------------------------------------------------------------
# 2. Secrets safety — no real credentials needed
# ---------------------------------------------------------------------------


def test_wizard_does_not_log_password_to_stdout(clean_config):
    """The wizard must not echo the password to stdout."""
    captured = io.StringIO()
    inputs = _wizard_inputs("bolt://localhost:7687", "neo4j", "top_secret_password")
    with (
        patch("neo4j_memory.cli._get_hermes_home", return_value=clean_config),
        patch("neo4j_memory.cli._smoke_test", return_value=(True, "")),
        patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
        patch("sys.stdin", io.StringIO(inputs)),
        patch("sys.stdout", captured),
    ):
        from neo4j_memory.cli import cmd_setup

        cmd_setup(SimpleNamespace())

    assert "top_secret_password" not in captured.getvalue()


def test_wizard_config_json_is_valid(clean_config):
    """The written config file must be valid JSON."""
    _run_wizard(clean_config, "bolt://localhost:7687", "neo4j", "pass")
    raw = (clean_config / "neo4j-memory.json").read_text(encoding="utf-8")
    data = json.loads(raw)  # would raise JSONDecodeError if invalid
    assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# 3. Failure / retry behaviour — no Neo4j needed
# ---------------------------------------------------------------------------


def test_wizard_aborts_on_smoke_failure_and_user_decline(clean_config):
    """Wizard must return None and not write config when user declines retry."""
    # Smoke test always fails; user says "n" to retry
    fail_inputs = (
        "\n".join(
            [
                "bolt://localhost:7687",
                "neo4j",
                "badpass",
                "neo4j",
                "anthropic/claude-3-5-sonnet-latest",
                "openai/text-embedding-3-small",
                "n",  # decline retry
                "",
            ]
        )
        + "\n"
    )

    with (
        patch("neo4j_memory.cli._get_hermes_home", return_value=clean_config),
        patch(
            "neo4j_memory.cli._smoke_test", return_value=(False, "Connection refused")
        ),
        patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
        patch("sys.stdin", io.StringIO(fail_inputs)),
        patch("sys.stdout", new_callable=io.StringIO),
    ):
        from neo4j_memory.cli import cmd_setup

        result = cmd_setup(SimpleNamespace())

    assert result is None, "Expected None when user declines retry after smoke failure"
    assert not (clean_config / "neo4j-memory.json").exists(), (
        "Config file must not be written when wizard aborts"
    )


def test_wizard_retries_on_smoke_failure_then_succeeds(clean_config):
    """Wizard must loop back and succeed on second attempt after one failure."""
    # First attempt fails; user accepts retry; second attempt succeeds.
    # The retry path re-prompts all fields — supply them again.
    retry_inputs = (
        "\n".join(
            [
                # First attempt (fails)
                "bolt://badhost:7687",
                "neo4j",
                "badpass",
                "neo4j",
                "anthropic/claude-3-5-sonnet-latest",
                "openai/text-embedding-3-small",
                "y",  # accept retry
                # Second attempt (succeeds)
                "bolt://localhost:7687",
                "neo4j",
                "goodpass",
                "neo4j",
                "anthropic/claude-3-5-sonnet-latest",
                "openai/text-embedding-3-small",
                "",
            ]
        )
        + "\n"
    )

    call_count = {"n": 0}

    def smoke_side_effect(cfg):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (False, "timeout")
        return (True, "")

    with (
        patch("neo4j_memory.cli._get_hermes_home", return_value=clean_config),
        patch("neo4j_memory.cli._smoke_test", side_effect=smoke_side_effect),
        patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
        patch("sys.stdin", io.StringIO(retry_inputs)),
        patch("sys.stdout", new_callable=io.StringIO),
    ):
        from neo4j_memory.cli import cmd_setup

        result = cmd_setup(SimpleNamespace())

    assert result is not None, "Wizard should succeed on second attempt"
    assert call_count["n"] == 2, "Expected exactly 2 smoke-test calls (fail then pass)"
    on_disk = json.loads((clean_config / "neo4j-memory.json").read_text())
    assert on_disk["uri"] == "bolt://localhost:7687"


def test_wizard_aborts_when_driver_unavailable(clean_config):
    """If _ensure_driver_installed returns False, cmd_setup must return None."""
    with (
        patch("neo4j_memory.cli._get_hermes_home", return_value=clean_config),
        patch("neo4j_memory.cli._ensure_driver_installed", return_value=False),
        patch("sys.stdin", io.StringIO("")),
        patch("sys.stdout", new_callable=io.StringIO),
    ):
        from neo4j_memory.cli import cmd_setup

        result = cmd_setup(SimpleNamespace())

    assert result is None


# ---------------------------------------------------------------------------
# 4. API key detection — no real keys needed
# ---------------------------------------------------------------------------


def test_detect_llm_keys_with_anthropic(tmp_path, monkeypatch):
    """_detect_llm_keys must detect ANTHROPIC_API_KEY from environment."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("neo4j_memory.cli._get_hermes_home", return_value=tmp_path):
        from neo4j_memory.cli import _detect_llm_keys

        detected = _detect_llm_keys()

    assert "anthropic" in detected


def test_detect_llm_keys_with_openai(tmp_path, monkeypatch):
    """_detect_llm_keys must detect OPENAI_API_KEY from environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("neo4j_memory.cli._get_hermes_home", return_value=tmp_path):
        from neo4j_memory.cli import _detect_llm_keys

        detected = _detect_llm_keys()

    assert "openai" in detected


def test_detect_llm_keys_ignores_placeholder(tmp_path, monkeypatch):
    """_detect_llm_keys must ignore placeholder values like 'CHANGEME'."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "CHANGEME")

    with patch("neo4j_memory.cli._get_hermes_home", return_value=tmp_path):
        from neo4j_memory.cli import _detect_llm_keys

        detected = _detect_llm_keys()

    assert "anthropic" not in detected


def test_suggest_llm_prefers_anthropic(tmp_path):
    """_suggest_llm_provider must prefer Anthropic when that key is detected."""
    from neo4j_memory.cli import _suggest_llm_provider

    result = _suggest_llm_provider({"anthropic": "ANTHROPIC_API_KEY"})
    assert result is not None
    assert "anthropic" in result.lower() or "claude" in result.lower()


def test_suggest_llm_falls_back_to_openai(tmp_path):
    """_suggest_llm_provider must suggest OpenAI when only that key is detected."""
    from neo4j_memory.cli import _suggest_llm_provider

    result = _suggest_llm_provider({"openai": "OPENAI_API_KEY"})
    assert result is not None
    assert "openai" in result.lower() or "gpt" in result.lower()


def test_suggest_embedding_prefers_openai(tmp_path):
    """_suggest_embedding_provider must prefer OpenAI for embeddings."""
    from neo4j_memory.cli import _suggest_embedding_provider

    result = _suggest_embedding_provider({"openai": "OPENAI_API_KEY"})
    assert "openai" in result.lower() or "embedding" in result.lower()


def test_suggest_embedding_fallback_no_keys(tmp_path):
    """_suggest_embedding_provider must return a default even with no keys detected."""
    from neo4j_memory.cli import _suggest_embedding_provider

    result = _suggest_embedding_provider({})
    assert isinstance(result, str) and result  # non-empty string


# ---------------------------------------------------------------------------
# 5. Live integration — requires docker-compose.test.yml
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_wizard_live_connection(clean_config, neo4j_test_uri, request):
    """Wizard connects to real Neo4j, smoke-tests successfully, writes config."""
    if not _integration_enabled(request.config):
        pytest.skip(
            "Integration tests disabled — pass --run-integration or set NEO4J_TEST_URI"
        )

    user = os.environ.get("NEO4J_TEST_USER", "neo4j")
    password = os.environ.get("NEO4J_TEST_PASSWORD", "testpassword")

    inputs = _wizard_inputs(neo4j_test_uri, user, password)

    with (
        patch("neo4j_memory.cli._get_hermes_home", return_value=clean_config),
        patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
        patch("sys.stdin", io.StringIO(inputs)),
        patch("sys.stdout", new_callable=io.StringIO),
    ):
        # Use the real _smoke_test — exercises actual Neo4j connectivity.
        from neo4j_memory.cli import cmd_setup

        result = cmd_setup(SimpleNamespace())

    assert result is not None, (
        f"cmd_setup returned None — wizard failed. "
        f"Is Neo4j reachable at {neo4j_test_uri}?"
    )
    config_file = clean_config / "neo4j-memory.json"
    assert config_file.exists()
    on_disk = json.loads(config_file.read_text())
    assert on_disk["uri"] == neo4j_test_uri
    assert on_disk["user"] == user
    assert on_disk["password"] == password


@pytest.mark.integration
def test_live_wizard_provider_is_available_post_setup(
    clean_config, neo4j_test_uri, request
):
    """After a live wizard run, Neo4jMemoryProvider.is_available() must be True."""
    if not _integration_enabled(request.config):
        pytest.skip(
            "Integration tests disabled — pass --run-integration or set NEO4J_TEST_URI"
        )

    user = os.environ.get("NEO4J_TEST_USER", "neo4j")
    password = os.environ.get("NEO4J_TEST_PASSWORD", "testpassword")
    inputs = _wizard_inputs(neo4j_test_uri, user, password)

    with (
        patch("neo4j_memory.cli._get_hermes_home", return_value=clean_config),
        patch("neo4j_memory.cli._ensure_driver_installed", return_value=True),
        patch("sys.stdin", io.StringIO(inputs)),
        patch("sys.stdout", new_callable=io.StringIO),
    ):
        from neo4j_memory.cli import cmd_setup

        result = cmd_setup(SimpleNamespace())

    assert result is not None

    try:
        from neo4j_memory import Neo4jMemoryProvider
    except ImportError:
        pytest.skip("Neo4jMemoryProvider not importable")

    provider = Neo4jMemoryProvider()
    provider.initialize("live-test-session", hermes_home=str(clean_config))
    try:
        assert provider.is_available() is True
    finally:
        provider.shutdown()

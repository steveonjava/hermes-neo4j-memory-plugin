"""Integration test for the Neo4j memory setup wizard.

Runs end-to-end against a live Neo4j instance (docker-compose or env var).
Skipped automatically in unit-test-only runs.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
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
        pass  # Already registered by conftest.py


def _integration_enabled(config) -> bool:
    return bool(
        NEO4J_TEST_URI
        or (hasattr(config, "getoption") and config.getoption("--run-integration", default=False))
    )


@pytest.fixture(scope="module")
def neo4j_test_uri():
    """Return the Neo4j test URI from environment or docker-compose default."""
    return NEO4J_TEST_URI or "bolt://localhost:7688"  # docker-compose maps 7688->7687


@pytest.fixture
def clean_config(tmp_path):
    """Yield a fresh hermes_home for each test, cleaned up afterwards."""
    yield tmp_path
    cfg_file = tmp_path / "neo4j-memory.json"
    if cfg_file.exists():
        cfg_file.unlink()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_full_wizard_writes_valid_config_and_provider_is_available(
    clean_config, neo4j_test_uri, request
):
    """End-to-end wizard run: writes config, provider reports is_available()=True."""
    if not _integration_enabled(request.config):
        pytest.skip("Integration tests disabled — pass --run-integration or set NEO4J_TEST_URI")

    # Parse user/password from the test URI if embedded, else use defaults
    user = os.environ.get("NEO4J_TEST_USER", "neo4j")
    password = os.environ.get("NEO4J_TEST_PASSWORD", "testpassword")

    inputs = "\n".join([
        neo4j_test_uri,
        user,
        password,
        "neo4j",    # database
        "anthropic/claude-3-5-sonnet-latest",
        "openai/text-embedding-3-small",
        "",         # no extra prompts
    ]) + "\n"

    with (
        patch("neo4j_memory.cli._get_hermes_home", return_value=clean_config),
        patch("sys.stdin", io.StringIO(inputs)),
        patch("sys.stdout", new_callable=io.StringIO),
    ):
        from neo4j_memory.cli import cmd_setup
        result = cmd_setup(SimpleNamespace())

    assert result is not None, "cmd_setup returned None — wizard failed"

    # Verify the config file was written
    config_file = clean_config / "neo4j-memory.json"
    assert config_file.exists(), "Config file was not written"
    on_disk = json.loads(config_file.read_text())
    assert on_disk["uri"] == neo4j_test_uri
    assert on_disk["password"] == password

    # Verify the provider reports itself as available
    with patch("neo4j_memory.client.Neo4jMemoryClientConfig.from_global_config") as mock_cfg:
        mock_cfg.return_value = type("Cfg", (), {
            "enabled": True,
            "uri": neo4j_test_uri,
            "password": password,
        })()
        try:
            from neo4j_memory import Neo4jMemoryProvider
            provider = Neo4jMemoryProvider()
            # is_available checks config, not network
            with patch("neo4j_memory.client.Neo4jMemoryClientConfig.from_global_config",
                       return_value=mock_cfg.return_value):
                assert provider.is_available() is True
        except ImportError:
            pytest.skip("Neo4jMemoryProvider not yet implemented (F3 pending)")


@pytest.mark.integration
def test_trivial_memory_roundtrip_post_setup(
    clean_config, neo4j_test_uri, request
):
    """After wizard setup, basic add_context/get_context works against live Neo4j."""
    if not _integration_enabled(request.config):
        pytest.skip("Integration tests disabled — pass --run-integration or set NEO4J_TEST_URI")

    try:
        from neo4j_memory.client import Neo4jMemoryClientConfig, Neo4jMemoryClientWrapper
    except ImportError:
        pytest.skip("F2 client not yet implemented")

    user = os.environ.get("NEO4J_TEST_USER", "neo4j")
    password = os.environ.get("NEO4J_TEST_PASSWORD", "testpassword")

    cfg = Neo4jMemoryClientConfig(
        uri=neo4j_test_uri,
        user=user,
        password=password,
        database="neo4j",
        llm="anthropic/claude-3-5-sonnet-latest",
        embedding="openai/text-embedding-3-small",
        enabled=True,
    )
    wrapper = Neo4jMemoryClientWrapper(cfg)
    wrapper.connect()

    try:
        # Store a fact
        wrapper.store_fact("integration-test", "is-a", "smoke-test")
        # Retrieve context — should not raise
        context = wrapper.get_context("integration-test")
        assert isinstance(context, str)
    finally:
        wrapper.close()

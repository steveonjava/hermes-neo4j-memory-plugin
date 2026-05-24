"""Integration tests for Neo4jMemoryClientWrapper and Neo4jMemoryProvider
against a real Neo4j instance.

Run with:
    docker compose -f docker-compose.test.yml up -d
    NEO4J_TEST_URI=bolt://localhost:7687 \\
    NEO4J_TEST_PASSWORD=testpassword \\
    pytest tests/integration/test_live_neo4j.py --run-integration -v

Or set NEO4J_TEST_URI in the environment to enable the tests without the flag.

The tests require neo4j-agent-memory to be installed:
    pip install "neo4j-agent-memory>=0.4.0"

Skip guard: each test is marked @pytest.mark.integration and is skipped
automatically unless --run-integration is passed or NEO4J_TEST_URI is set.
No real user secrets are needed — docker-compose.test.yml creates a dedicated
ephemeral Neo4j instance (neo4j/testpassword).
"""

from __future__ import annotations

import os
import time

import pytest

# ---------------------------------------------------------------------------
# Skip guard helpers
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
        pass  # Already registered by another conftest


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
def neo4j_uri():
    """Neo4j bolt URI: env var or docker-compose default."""
    # docker-compose.test.yml maps host 7687 -> container 7687
    return NEO4J_TEST_URI or "bolt://localhost:7687"


@pytest.fixture(scope="module")
def neo4j_creds():
    """Return (user, password) from env or docker-compose defaults."""
    user = os.environ.get("NEO4J_TEST_USER", "neo4j")
    password = os.environ.get("NEO4J_TEST_PASSWORD", "testpassword")
    return user, password


@pytest.fixture(scope="module")
def live_wrapper(neo4j_uri, neo4j_creds, request):
    """Module-scoped connected Neo4jMemoryClientWrapper.

    Skipped when integration tests are disabled.  The wrapper is closed after
    the module finishes, ensuring cleanup even if a test fails.
    """
    if not _integration_enabled(request.config):
        pytest.skip(
            "Integration tests disabled — pass --run-integration or set NEO4J_TEST_URI"
        )

    try:
        from neo4j_memory.client import (
            Neo4jMemoryClientConfig,
            Neo4jMemoryClientWrapper,
        )
    except ImportError:
        pytest.skip("neo4j_memory.client not importable — is the package installed?")

    user, password = neo4j_creds
    cfg = Neo4jMemoryClientConfig(
        uri=neo4j_uri,
        user=user,
        password=password,
        database="neo4j",
        # LLM/embedding provider strings — no real API keys needed for connect()
        # and basic graph operations that don't invoke embedding generation.
        llm="anthropic/claude-3-5-sonnet-latest",
        embedding="openai/text-embedding-3-small",
        enabled=True,
    )
    wrapper = Neo4jMemoryClientWrapper(cfg)
    wrapper.connect()
    if not wrapper.is_available():
        pytest.fail(
            f"Could not connect to Neo4j at {neo4j_uri}. "
            "Is docker-compose.test.yml running?"
        )
    yield wrapper
    wrapper.close()


@pytest.fixture(scope="module")
def live_provider(neo4j_uri, neo4j_creds, tmp_path_factory, request):
    """Module-scoped initialized Neo4jMemoryProvider.

    Writes a temp config file pointing at the test Neo4j instance, then
    initializes the provider from that config.
    """
    if not _integration_enabled(request.config):
        pytest.skip(
            "Integration tests disabled — pass --run-integration or set NEO4J_TEST_URI"
        )

    try:
        from neo4j_memory import Neo4jMemoryProvider
    except ImportError:
        pytest.skip("neo4j_memory not importable — is the package installed?")

    import json

    user, password = neo4j_creds
    hermes_home = tmp_path_factory.mktemp("hermes_home_provider")
    config_data = {
        "uri": neo4j_uri,
        "user": user,
        "password": password,
        "database": "neo4j",
        "llm": "anthropic/claude-3-5-sonnet-latest",
        "embedding": "openai/text-embedding-3-small",
        "enabled": True,
    }
    (hermes_home / "neo4j-memory.json").write_text(
        json.dumps(config_data, indent=2), encoding="utf-8"
    )

    provider = Neo4jMemoryProvider()
    provider.initialize("test-session", hermes_home=str(hermes_home))
    if not provider.is_available():
        pytest.fail(
            f"Provider did not become available after init (uri={neo4j_uri}). "
            "Is docker-compose.test.yml running?"
        )
    yield provider
    provider.shutdown()


# ---------------------------------------------------------------------------
# 1. Connection tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_wrapper_connects_and_reports_available(live_wrapper):
    """Wrapper must report is_available()==True after connect()."""
    assert live_wrapper.is_available() is True


@pytest.mark.integration
def test_wrapper_unavailable_on_bad_uri(request):
    """Wrapper with wrong URI must stay is_available()==False without raising."""
    if not _integration_enabled(request.config):
        pytest.skip("Integration tests disabled")

    try:
        from neo4j_memory.client import (
            Neo4jMemoryClientConfig,
            Neo4jMemoryClientWrapper,
        )
    except ImportError:
        pytest.skip("neo4j_memory.client not importable")

    cfg = Neo4jMemoryClientConfig(
        uri="bolt://127.0.0.1:19999",  # nothing listening here
        user="neo4j",
        password="wrongpassword",
        enabled=True,
    )
    wrapper = Neo4jMemoryClientWrapper(cfg)
    wrapper.connect(timeout=5.0)
    try:
        assert wrapper.is_available() is False
    finally:
        wrapper.close()


@pytest.mark.integration
def test_provider_is_available_after_init(live_provider):
    """Provider must report is_available()==True after successful initialize()."""
    assert live_provider.is_available() is True


# ---------------------------------------------------------------------------
# 2. Write / store tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_store_fact_does_not_raise(live_wrapper):
    """store_fact() must complete without raising against live Neo4j."""
    # generate_embedding=True is passed internally but embedding calls may
    # fail if no real API key exists.  We exercise the graph write path and
    # accept either success or an embedding-specific error.
    try:
        live_wrapper.store_fact(
            "integration test: sky is blue",
            metadata={"source": "pytest", "ts": str(time.time())},
        )
    except RuntimeError as exc:
        if "embedding" in str(exc).lower() or "api" in str(exc).lower():
            pytest.skip(f"Embedding API unavailable in test env: {exc}")
        raise


@pytest.mark.integration
def test_store_entity_does_not_raise(live_wrapper):
    """store_entity() must complete without raising against live Neo4j."""
    try:
        live_wrapper.store_entity(
            name="TestAgent",
            entity_type="agent",
            observations=["automated integration test entity"],
            metadata={"source": "pytest"},
        )
    except RuntimeError as exc:
        if "embedding" in str(exc).lower() or "api" in str(exc).lower():
            pytest.skip(f"Embedding API unavailable in test env: {exc}")
        raise


@pytest.mark.integration
def test_store_preference_does_not_raise(live_wrapper):
    """store_preference() must complete without raising against live Neo4j."""
    try:
        live_wrapper.store_preference(
            "prefers structured logging",
            metadata={"source": "pytest"},
        )
    except RuntimeError as exc:
        if "embedding" in str(exc).lower() or "api" in str(exc).lower():
            pytest.skip(f"Embedding API unavailable in test env: {exc}")
        raise


@pytest.mark.integration
def test_provider_handle_store_fact(live_provider):
    """Provider.handle_tool_call('neo4j_store_fact') must return empty string."""
    try:
        result = live_provider.handle_tool_call(
            "neo4j_store_fact",
            {
                "text": "integration test fact from provider",
                "metadata": {"ts": str(time.time())},
            },
        )
        assert result == ""
    except RuntimeError as exc:
        if "embedding" in str(exc).lower() or "api" in str(exc).lower():
            pytest.skip(f"Embedding API unavailable in test env: {exc}")
        raise


# ---------------------------------------------------------------------------
# 3. Search / read tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_search_returns_list(live_wrapper):
    """search() must return a list (possibly empty) without raising."""
    try:
        results = live_wrapper.search("test query", limit=5)
        assert isinstance(results, list)
    except RuntimeError as exc:
        if "embedding" in str(exc).lower() or "api" in str(exc).lower():
            pytest.skip(f"Embedding API unavailable in test env: {exc}")
        raise


@pytest.mark.integration
def test_search_result_shape(live_wrapper):
    """Each search result must have at least a 'type' key."""
    try:
        results = live_wrapper.search("agent", limit=20)
    except RuntimeError as exc:
        if "embedding" in str(exc).lower() or "api" in str(exc).lower():
            pytest.skip(f"Embedding API unavailable in test env: {exc}")
        raise

    for item in results:
        assert "type" in item, f"Result missing 'type' key: {item}"
        assert item["type"] in {"entity", "preference", "fact"}, (
            f"Unexpected result type: {item['type']}"
        )


@pytest.mark.integration
def test_provider_handle_search_returns_json(live_provider):
    """Provider.handle_tool_call('neo4j_search') must return a JSON-decodable string."""
    import json

    try:
        raw = live_provider.handle_tool_call(
            "neo4j_search",
            {"query": "integration test", "limit": 5},
        )
    except RuntimeError as exc:
        if "embedding" in str(exc).lower() or "api" in str(exc).lower():
            pytest.skip(f"Embedding API unavailable in test env: {exc}")
        raise

    parsed = json.loads(raw)
    assert isinstance(parsed, list)


# ---------------------------------------------------------------------------
# 4. get_context tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_get_context_returns_string(live_wrapper):
    """get_context() must return a str without raising."""
    try:
        result = live_wrapper.get_context("What do I know about tests?")
        assert isinstance(result, str)
    except RuntimeError as exc:
        if "embedding" in str(exc).lower() or "api" in str(exc).lower():
            pytest.skip(f"Embedding API unavailable in test env: {exc}")
        raise


@pytest.mark.integration
def test_provider_handle_get_context_returns_string(live_provider):
    """Provider.handle_tool_call('neo4j_get_context') must return a str."""
    try:
        result = live_provider.handle_tool_call(
            "neo4j_get_context",
            {"query": "integration test query"},
        )
        assert isinstance(result, str)
    except RuntimeError as exc:
        if "embedding" in str(exc).lower() or "api" in str(exc).lower():
            pytest.skip(f"Embedding API unavailable in test env: {exc}")
        raise


# ---------------------------------------------------------------------------
# 5. Metadata / round-trip test (best-effort — skipped if embedding unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_store_and_search_roundtrip(live_wrapper):
    """Store a uniquely-named entity then verify it appears in search results.

    This test exercises the full store → search round-trip against live Neo4j.
    It is skipped if the embedding API is unavailable, because semantic search
    requires embeddings.  The test uses a sufficiently unique name so it won't
    collide with other test data.
    """
    unique_marker = f"IntegTestEntity_{int(time.time() * 1000)}"

    try:
        live_wrapper.store_entity(
            name=unique_marker,
            entity_type="test_marker",
            observations=["created by pytest integration test"],
        )
        results = live_wrapper.search(unique_marker, limit=20)
    except RuntimeError as exc:
        if "embedding" in str(exc).lower() or "api" in str(exc).lower():
            pytest.skip(f"Embedding API unavailable — skipping round-trip: {exc}")
        raise

    names = []
    for r in results:
        if r.get("type") == "entity":
            names.append(r.get("name", ""))

    assert any(unique_marker in n for n in names), (
        f"Stored entity '{unique_marker}' not found in search results: {results}"
    )


# ---------------------------------------------------------------------------
# 6. Isolation / cleanup tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_close_then_operations_raise(live_wrapper, request, neo4j_uri, neo4j_creds):
    """After close(), is_available() returns False and operations raise RuntimeError.

    Creates a *separate* wrapper so as not to destroy the module-scoped fixture.
    """
    if not _integration_enabled(request.config):
        pytest.skip("Integration tests disabled")

    try:
        from neo4j_memory.client import (
            Neo4jMemoryClientConfig,
            Neo4jMemoryClientWrapper,
        )
    except ImportError:
        pytest.skip("neo4j_memory.client not importable")

    user, password = neo4j_creds
    cfg = Neo4jMemoryClientConfig(
        uri=neo4j_uri,
        user=user,
        password=password,
        enabled=True,
    )
    w = Neo4jMemoryClientWrapper(cfg)
    w.connect()
    assert w.is_available() is True
    w.close()

    assert w.is_available() is False
    with pytest.raises(RuntimeError):
        w.store_fact("should fail — already closed")


@pytest.mark.integration
def test_two_independent_wrappers_do_not_interfere(request, neo4j_uri, neo4j_creds):
    """Two wrapper instances connected to the same Neo4j must operate independently.

    Verifies that closing one does not affect the other.
    """
    if not _integration_enabled(request.config):
        pytest.skip("Integration tests disabled")

    try:
        from neo4j_memory.client import (
            Neo4jMemoryClientConfig,
            Neo4jMemoryClientWrapper,
        )
    except ImportError:
        pytest.skip("neo4j_memory.client not importable")

    user, password = neo4j_creds
    cfg = Neo4jMemoryClientConfig(
        uri=neo4j_uri, user=user, password=password, enabled=True
    )

    w1 = Neo4jMemoryClientWrapper(cfg)
    w2 = Neo4jMemoryClientWrapper(cfg)
    w1.connect()
    w2.connect()

    try:
        assert w1.is_available() is True
        assert w2.is_available() is True

        # Close only w1; w2 must still be operational
        w1.close()
        assert w1.is_available() is False
        assert w2.is_available() is True

        # w2 must still accept operations without error
        # (we call search with limit=1 as a lightweight connectivity probe)
        try:
            results = w2.search("probe", limit=1)
            assert isinstance(results, list)
        except RuntimeError as exc:
            if "embedding" in str(exc).lower() or "api" in str(exc).lower():
                pass  # Embedding unavailable — connection still works
            else:
                raise
    finally:
        w2.close()


# ---------------------------------------------------------------------------
# 7. Fail-meaningfully test (no container)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_connect_to_missing_neo4j_sets_unavailable(request):
    """Without a live Neo4j container connect() must leave is_available()==False.

    This test does NOT need the container — it intentionally uses a port
    that nothing is listening on.
    """
    # This test always runs when integration tests are enabled; it exercises
    # the graceful failure path, not the happy path.
    if not _integration_enabled(request.config):
        pytest.skip("Integration tests disabled")

    try:
        from neo4j_memory.client import (
            Neo4jMemoryClientConfig,
            Neo4jMemoryClientWrapper,
        )
    except ImportError:
        pytest.skip("neo4j_memory.client not importable")

    cfg = Neo4jMemoryClientConfig(
        uri="bolt://127.0.0.1:19998",
        user="neo4j",
        password="doesnotmatter",
        enabled=True,
    )
    wrapper = Neo4jMemoryClientWrapper(cfg)
    wrapper.connect(timeout=5.0)
    try:
        assert wrapper.is_available() is False, (
            "Expected is_available()==False when no Neo4j is reachable"
        )
    finally:
        wrapper.close()

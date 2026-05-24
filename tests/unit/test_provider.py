"""Unit tests for Neo4jMemoryProvider."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from neo4j_memory import Neo4jMemoryProvider


@pytest.fixture
def mock_wrapper_cls():
    """Patch Neo4jMemoryClientWrapper to avoid real connection."""
    with patch("neo4j_memory.Neo4jMemoryClientWrapper") as MockCls:
        instance = MagicMock()
        instance.is_available.return_value = True
        MockCls.return_value = instance
        yield MockCls, instance


@pytest.fixture
def initialized_provider(mock_wrapper_cls, tmp_path):
    """A provider whose wrapper has been patched and initialized."""
    _, wrapper = mock_wrapper_cls
    provider = Neo4jMemoryProvider()
    provider.initialize("sess1", hermes_home=str(tmp_path))
    return provider, wrapper


def test_get_tool_schemas_returns_5_tools():
    provider = Neo4jMemoryProvider()
    schemas = provider.get_tool_schemas()
    assert isinstance(schemas, list)
    assert len(schemas) == 5
    names = {s["name"] for s in schemas}
    assert names == {
        "neo4j_store_fact",
        "neo4j_store_entity",
        "neo4j_store_preference",
        "neo4j_search",
        "neo4j_get_context",
    }
    for s in schemas:
        assert "name" in s
        assert "description" in s
        assert "parameters" in s


def test_store_fact_routes_to_wrapper(initialized_provider):
    provider, wrapper = initialized_provider
    result = provider.handle_tool_call("neo4j_store_fact", {"text": "foo"})
    wrapper.store_fact.assert_called_once_with(text="foo", metadata=None)
    assert result == ""


def test_store_entity_routes_to_wrapper(initialized_provider):
    provider, wrapper = initialized_provider
    args = {"name": "A", "entity_type": "person", "observations": ["x"]}
    result = provider.handle_tool_call("neo4j_store_entity", args)
    wrapper.store_entity.assert_called_once_with(
        name="A",
        entity_type="person",
        observations=["x"],
        metadata=None,
    )
    assert result == ""


def test_store_preference_routes_to_wrapper(initialized_provider):
    provider, wrapper = initialized_provider
    result = provider.handle_tool_call(
        "neo4j_store_preference", {"preference": "dark mode"}
    )
    wrapper.store_preference.assert_called_once_with(
        preference="dark mode",
        metadata=None,
    )
    assert result == ""


def test_search_routes_and_json_encodes(initialized_provider):
    provider, wrapper = initialized_provider
    payload = [{"type": "fact", "text": "x"}]
    wrapper.search.return_value = payload
    result = provider.handle_tool_call("neo4j_search", {"query": "test"})
    wrapper.search.assert_called_once_with(query="test", limit=10, namespace=None)
    assert result == json.dumps(payload)


def test_get_context_routes_to_wrapper(initialized_provider):
    provider, wrapper = initialized_provider
    wrapper.get_context.return_value = "ctx string"
    result = provider.handle_tool_call("neo4j_get_context", {"query": "q"})
    wrapper.get_context.assert_called_once_with(query="q")
    assert result == "ctx string"


def test_is_available_false_when_wrapper_unavailable():
    provider = Neo4jMemoryProvider()
    mock_wrapper = MagicMock()
    mock_wrapper.is_available.return_value = False
    provider._wrapper = mock_wrapper
    assert provider.is_available() is False


def test_is_available_false_before_initialize():
    provider = Neo4jMemoryProvider()
    assert provider.is_available() is False


def test_unknown_tool_raises_value_error(initialized_provider):
    provider, _ = initialized_provider
    with pytest.raises(ValueError, match="Unknown tool"):
        provider.handle_tool_call("not_a_tool", {})


def test_name_property():
    assert Neo4jMemoryProvider().name == "neo4j"


def test_initialize_applies_env_overrides(mock_wrapper_cls, tmp_path, monkeypatch):
    MockCls, _ = mock_wrapper_cls
    monkeypatch.setenv("NEO4J_URI", "bolt://override:7687")
    monkeypatch.setenv("NEO4J_USER", "alice")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("NEO4J_DATABASE", "altdb")
    provider = Neo4jMemoryProvider()
    provider.initialize("sess1", hermes_home=str(tmp_path))
    cfg = MockCls.call_args.args[0]
    assert cfg.uri == "bolt://override:7687"
    assert cfg.user == "alice"
    assert cfg.password == "secret"
    assert cfg.database == "altdb"


def test_initialize_reads_config_file(mock_wrapper_cls, tmp_path):
    MockCls, _ = mock_wrapper_cls
    (tmp_path / "neo4j-memory.json").write_text(
        json.dumps(
            {
                "uri": "bolt://from-file:7687",
                "user": "bob",
                "password": "filepass",
                "database": "filedb",
            }
        )
    )
    provider = Neo4jMemoryProvider()
    provider.initialize("sess1", hermes_home=str(tmp_path))
    cfg = MockCls.call_args.args[0]
    assert cfg.uri == "bolt://from-file:7687"
    assert cfg.user == "bob"
    assert cfg.password == "filepass"
    assert cfg.database == "filedb"


def test_shutdown_calls_wrapper_close(initialized_provider):
    provider, wrapper = initialized_provider
    provider.shutdown()
    wrapper.close.assert_called_once()

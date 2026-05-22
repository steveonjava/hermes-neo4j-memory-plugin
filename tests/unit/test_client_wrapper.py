"""Unit tests for Neo4jMemoryClientWrapper."""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from neo4j_memory.client import Neo4jMemoryClientConfig, Neo4jMemoryClientWrapper


def make_sync_submit(loop=None):
    """Return a submit() that runs coroutines synchronously via asyncio.run()."""
    def submit(coro, timeout=30.0):
        return asyncio.run(coro)
    return submit


@pytest.fixture
def mock_loop_thread():
    with patch("neo4j_memory.client.AsyncLoopThread") as MockThread:
        instance = MagicMock()
        instance.submit.side_effect = make_sync_submit()
        MockThread.return_value = instance
        yield instance


@pytest.fixture
def mock_neo4j_lib():
    mock_client = MagicMock()
    mock_client.connect = AsyncMock()
    mock_client.close = AsyncMock()
    mock_client.get_context = AsyncMock(return_value="ctx")
    mock_client.long_term = MagicMock()
    mock_client.long_term.add_fact = AsyncMock(return_value=MagicMock())
    mock_client.long_term.add_entity = AsyncMock(return_value=(MagicMock(), MagicMock()))
    mock_client.long_term.add_preference = AsyncMock(return_value=MagicMock())
    mock_client.long_term.search_entities = AsyncMock(return_value=[])
    mock_client.long_term.search_preferences = AsyncMock(return_value=[])
    mock_client.long_term.search_facts = AsyncMock(return_value=[])

    mock_memory_client_cls = MagicMock(return_value=mock_client)
    mock_settings_cls = MagicMock()
    mock_neo4j_config_cls = MagicMock()

    with patch("neo4j_memory.client.AsyncLoopThread") as MockThread:
        thread_instance = MagicMock()
        thread_instance.submit.side_effect = make_sync_submit()
        MockThread.return_value = thread_instance

        with patch.dict("sys.modules", {
            "neo4j_agent_memory": MagicMock(
                MemoryClient=mock_memory_client_cls,
                MemorySettings=mock_settings_cls,
            ),
            "neo4j_agent_memory.config": MagicMock(),
            "neo4j_agent_memory.config.settings": MagicMock(Neo4jConfig=mock_neo4j_config_cls),
            "pydantic": MagicMock(SecretStr=lambda x: x),
        }):
            yield {
                "thread": thread_instance,
                "client": mock_client,
                "MemoryClient": mock_memory_client_cls,
                "MemorySettings": mock_settings_cls,
                "Neo4jConfig": mock_neo4j_config_cls,
            }


def make_wrapper(cfg=None):
    if cfg is None:
        cfg = Neo4jMemoryClientConfig()
    return Neo4jMemoryClientWrapper(cfg)


# 1. Config defaults
def test_config_defaults():
    cfg = Neo4jMemoryClientConfig()
    assert cfg.uri == "bolt://localhost:7687"
    assert cfg.user == "neo4j"
    assert cfg.password == ""
    assert cfg.database == "neo4j"
    assert cfg.llm == "anthropic/claude-3-5-sonnet-latest"
    assert cfg.embedding == "openai/text-embedding-3-small"
    assert cfg.enabled is True


# 2. is_available returns False before connect, True after
def test_is_available_before_connect(mock_neo4j_lib):
    w = make_wrapper()
    assert w.is_available() is False


def test_is_available_after_connect(mock_neo4j_lib):
    w = make_wrapper()
    w.connect()
    assert w.is_available() is True


# 3. connect() calls MemoryClient and client.connect() in right sequence
def test_connect_calls_sequence(mock_neo4j_lib):
    w = make_wrapper()
    w.connect()
    mock_neo4j_lib["MemoryClient"].assert_called_once()
    mock_neo4j_lib["client"].connect.assert_called_once()


# 4. connect() on failure logs warning and leaves is_available==False
def test_connect_failure_does_not_raise(mock_neo4j_lib, caplog):
    mock_neo4j_lib["MemoryClient"].side_effect = Exception("connection refused")
    w = make_wrapper()
    with caplog.at_level(logging.WARNING):
        w.connect()  # should not raise
    assert w.is_available() is False
    assert "connection refused" in caplog.text or "Neo4j connection failed" in caplog.text


# 5. store_fact calls long_term.add_fact with correct mapping
def test_store_fact(mock_neo4j_lib):
    w = make_wrapper()
    w.connect()
    w.store_fact("sky is blue", metadata={"src": "test"})
    mock_neo4j_lib["client"].long_term.add_fact.assert_called_once_with(
        subject="agent",
        predicate="knows",
        obj="sky is blue",
        metadata={"src": "test"},
        generate_embedding=True,
    )


# 6. store_entity calls add_entity with observations in metadata
def test_store_entity(mock_neo4j_lib):
    w = make_wrapper()
    w.connect()
    w.store_entity("Alice", "person", observations=["works at Acme"])
    mock_neo4j_lib["client"].long_term.add_entity.assert_called_once_with(
        name="Alice",
        entity_type="person",
        metadata={"observations": ["works at Acme"]},
        generate_embedding=True,
    )


def test_store_entity_no_observations(mock_neo4j_lib):
    w = make_wrapper()
    w.connect()
    w.store_entity("Bob", "person", observations=[])
    call_kwargs = mock_neo4j_lib["client"].long_term.add_entity.call_args
    assert call_kwargs.kwargs.get("metadata") is None


# 7. store_preference uses category='general'
def test_store_preference(mock_neo4j_lib):
    w = make_wrapper()
    w.connect()
    w.store_preference("prefers dark mode")
    mock_neo4j_lib["client"].long_term.add_preference.assert_called_once_with(
        category="general",
        preference="prefers dark mode",
        metadata=None,
        generate_embedding=True,
    )


# 8. search aggregates entities, preferences, facts
def test_search_aggregates(mock_neo4j_lib):
    entity = MagicMock()
    entity.display_name = "Alice"
    entity.type = "person"
    entity.description = "a person"

    pref = MagicMock()
    pref.category = "ui"
    pref.preference = "dark mode"

    fact = MagicMock()
    fact.subject = "agent"
    fact.predicate = "knows"
    fact.obj = "sky is blue"

    mock_neo4j_lib["client"].long_term.search_entities = AsyncMock(return_value=[entity])
    mock_neo4j_lib["client"].long_term.search_preferences = AsyncMock(return_value=[pref])
    mock_neo4j_lib["client"].long_term.search_facts = AsyncMock(return_value=[fact])

    w = make_wrapper()
    w.connect()
    results = w.search("query")

    types = [r["type"] for r in results]
    assert "entity" in types
    assert "preference" in types
    assert "fact" in types
    assert len(results) == 3


# 9. get_context returns result
def test_get_context(mock_neo4j_lib):
    mock_neo4j_lib["client"].get_context = AsyncMock(return_value="formatted context")
    w = make_wrapper()
    w.connect()
    result = w.get_context("what do I like?")
    assert result == "formatted context"
    mock_neo4j_lib["client"].get_context.assert_called_once_with("what do I like?")


# 10. Library exception wraps to RuntimeError
def test_store_fact_wraps_exception(mock_neo4j_lib):
    mock_neo4j_lib["client"].long_term.add_fact = AsyncMock(side_effect=ValueError("db error"))
    w = make_wrapper()
    w.connect()
    with pytest.raises(RuntimeError, match="Neo4j memory operation failed"):
        w.store_fact("some fact")


# 11. close() calls client.close and shuts down loop thread, sets is_available==False
def test_close(mock_neo4j_lib):
    w = make_wrapper()
    w.connect()
    assert w.is_available() is True
    w.close()
    assert w.is_available() is False
    mock_neo4j_lib["client"].close.assert_called_once()
    mock_neo4j_lib["thread"].shutdown.assert_called_once()


# 12. store_fact before connect raises RuntimeError
def test_store_fact_before_connect_raises(mock_neo4j_lib):
    w = make_wrapper()
    with pytest.raises(RuntimeError, match="not connected"):
        w.store_fact("test")


def test_search_before_connect_raises(mock_neo4j_lib):
    w = make_wrapper()
    with pytest.raises(RuntimeError, match="not connected"):
        w.search("query")


def test_get_context_before_connect_raises(mock_neo4j_lib):
    w = make_wrapper()
    with pytest.raises(RuntimeError, match="not connected"):
        w.get_context("query")

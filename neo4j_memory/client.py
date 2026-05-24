"""Neo4j memory client config and wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from neo4j_memory._async_bridge import AsyncLoopThread

logger = logging.getLogger(__name__)


@dataclass
class Neo4jMemoryClientConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    llm: str = "anthropic/claude-3-5-sonnet-latest"
    embedding: str = "openai/text-embedding-3-small"
    enabled: bool = True


class Neo4jMemoryClientWrapper:
    """Sync wrapper over async neo4j-agent-memory MemoryClient via AsyncLoopThread."""

    def __init__(self, config: Neo4jMemoryClientConfig) -> None:
        self.config = config
        self._loop_thread = AsyncLoopThread()
        self._client = None  # MemoryClient, set after connect()
        self._available = False

    def connect(self, timeout: float = 30.0) -> None:
        """Connect to Neo4j and set up schema. Logs warning on failure, leaves is_available()==False."""
        from neo4j_agent_memory import MemoryClient, MemorySettings
        from neo4j_agent_memory.config.settings import Neo4jConfig
        from pydantic import SecretStr

        try:
            self._loop_thread.start()
            settings = MemorySettings(
                neo4j=Neo4jConfig(
                    uri=self.config.uri,
                    username=self.config.user,
                    password=SecretStr(self.config.password),
                    database=self.config.database,
                )
            )
            client = MemoryClient(settings)
            self._loop_thread.submit(client.connect(), timeout=timeout)
            self._client = client
            self._available = True
        except Exception as exc:
            logger.warning(
                "Neo4j connection failed (%s: %s); memory provider will be inactive.",
                type(exc).__name__,
                exc,
            )
            self._available = False

    def close(self, timeout: float = 5.0) -> None:
        """Close the client and shut down the async loop thread."""
        if self._client is not None:
            try:
                self._loop_thread.submit(self._client.close(), timeout=timeout)
            except Exception:
                pass
            self._client = None
        self._available = False
        self._loop_thread.shutdown(timeout=timeout)

    def is_available(self) -> bool:
        """True only when successfully connected with no error."""
        return self._available

    def _require_client(self):
        if self._client is None or not self._available:
            raise RuntimeError(
                "Neo4j memory client is not connected. Call connect() first."
            )
        return self._client

    def _run(self, coro, timeout: float):
        """Submit a coroutine and wrap exceptions as RuntimeError."""
        try:
            return self._loop_thread.submit(coro, timeout=timeout)
        except Exception as exc:
            raise RuntimeError(
                f"Neo4j memory operation failed ({type(exc).__name__}): {exc}"
            ) from exc

    def store_fact(
        self, text: str, metadata: dict | None = None, timeout: float = 30.0
    ) -> None:
        """Store a free-text fact. Maps to long_term.add_fact(subject=text, predicate='states', obj=text)."""
        client = self._require_client()
        self._run(
            client.long_term.add_fact(
                subject="agent",
                predicate="knows",
                obj=text,
                metadata=metadata,
                generate_embedding=True,
            ),
            timeout=timeout,
        )

    def store_entity(
        self,
        name: str,
        entity_type: str,
        observations: list[str],
        metadata: dict | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Store an entity. observations are merged into metadata."""
        client = self._require_client()
        merged_meta = dict(metadata or {})
        if observations:
            merged_meta.setdefault("observations", observations)
        self._run(
            client.long_term.add_entity(
                name=name,
                entity_type=entity_type,
                metadata=merged_meta if merged_meta else None,
                generate_embedding=True,
            ),
            timeout=timeout,
        )

    def store_preference(
        self,
        preference: str,
        metadata: dict | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Store a preference. Uses category='general'."""
        client = self._require_client()
        self._run(
            client.long_term.add_preference(
                category="general",
                preference=preference,
                metadata=metadata,
                generate_embedding=True,
            ),
            timeout=timeout,
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        namespace: str | None = None,
        timeout: float = 30.0,
    ) -> list[dict]:
        """Search entities, preferences, and facts. Returns list of dicts with type/content keys."""
        client = self._require_client()

        async def _search():
            results = []
            entities = await client.long_term.search_entities(query=query, limit=limit)
            for e in entities:
                results.append(
                    {
                        "type": "entity",
                        "name": getattr(e, "display_name", None)
                        or getattr(e, "name", str(e)),
                        "entity_type": str(getattr(e, "type", "")),
                        "description": getattr(e, "description", None),
                    }
                )
            prefs = await client.long_term.search_preferences(query=query, limit=limit)
            for p in prefs:
                results.append(
                    {
                        "type": "preference",
                        "category": getattr(p, "category", None),
                        "preference": getattr(p, "preference", str(p)),
                    }
                )
            try:
                facts = await client.long_term.search_facts(query=query, limit=limit)
                for f in facts:
                    results.append(
                        {
                            "type": "fact",
                            "subject": getattr(f, "subject", None),
                            "predicate": getattr(f, "predicate", None),
                            "object": getattr(f, "obj", None),
                        }
                    )
            except Exception:
                pass
            return results

        return self._run(_search(), timeout=timeout)

    def get_context(self, query: str, timeout: float = 30.0) -> str:
        """Get assembled context string for use in LLM prompts."""
        client = self._require_client()
        return self._run(client.get_context(query), timeout=timeout)

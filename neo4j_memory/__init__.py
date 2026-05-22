"""Neo4j graph-backed memory provider for Hermes."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from neo4j_memory.client import Neo4jMemoryClientConfig, Neo4jMemoryClientWrapper

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


STORE_FACT_SCHEMA = {
    "name": "neo4j_store_fact",
    "description": (
        "Store a free-text fact in the Neo4j long-term memory graph. "
        "Use for durable knowledge that should be recallable across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The factual statement to store.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata to attach to the fact.",
            },
        },
        "required": ["text"],
    },
}

STORE_ENTITY_SCHEMA = {
    "name": "neo4j_store_entity",
    "description": (
        "Store a named entity (person, place, project, concept) with one or more "
        "observations. Observations are merged into the entity's metadata."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The entity's name.",
            },
            "entity_type": {
                "type": "string",
                "description": "The entity's type (e.g. 'person', 'project').",
            },
            "observations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Observations about the entity.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata to attach to the entity.",
            },
        },
        "required": ["name", "entity_type", "observations"],
    },
}

STORE_PREFERENCE_SCHEMA = {
    "name": "neo4j_store_preference",
    "description": ("Store a user preference in the Neo4j long-term memory graph."),
    "parameters": {
        "type": "object",
        "properties": {
            "preference": {
                "type": "string",
                "description": "The preference statement to store.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata to attach to the preference.",
            },
        },
        "required": ["preference"],
    },
}

SEARCH_SCHEMA = {
    "name": "neo4j_search",
    "description": (
        "Unified semantic search over stored entities, preferences, and facts. "
        "Returns a JSON list of result objects."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results per category (default 10).",
            },
            "namespace": {
                "type": "string",
                "description": "Optional namespace filter.",
            },
        },
        "required": ["query"],
    },
}

GET_CONTEXT_SCHEMA = {
    "name": "neo4j_get_context",
    "description": (
        "Return a contextual summary string assembled from the Neo4j memory "
        "graph, suitable for injecting into an LLM prompt."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to build context around.",
            },
        },
        "required": ["query"],
    },
}

ALL_TOOL_SCHEMAS = [
    STORE_FACT_SCHEMA,
    STORE_ENTITY_SCHEMA,
    STORE_PREFERENCE_SCHEMA,
    SEARCH_SCHEMA,
    GET_CONTEXT_SCHEMA,
]


class Neo4jMemoryProvider:
    """MemoryProvider exposing Neo4j-backed long-term memory to Hermes."""

    def __init__(self) -> None:
        self._wrapper: Optional[Neo4jMemoryClientWrapper] = None

    @property
    def name(self) -> str:
        return "neo4j"

    def is_available(self) -> bool:
        if self._wrapper is None:
            return False
        return self._wrapper.is_available()

    def get_config_schema(self) -> List[Dict[str, Any]]:
        from neo4j_memory.cli import get_config_schema

        return get_config_schema()

    def save_config(self, values, hermes_home):
        """Write Neo4j memory config to $HERMES_HOME/neo4j-memory.json."""
        config_path = os.path.join(str(hermes_home), "neo4j-memory.json")
        existing: Dict[str, Any] = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        existing = loaded
            except Exception:
                pass
        existing.update(values or {})
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
            f.write("\n")

    def post_setup(self, hermes_home: str, config: dict) -> Optional[dict]:
        """Run the Neo4j setup wizard after provider selection."""
        from neo4j_memory.cli import cmd_setup

        os.environ["HERMES_HOME"] = str(hermes_home)
        return cmd_setup(config or {})

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))
        config_path = os.path.join(hermes_home, "neo4j-memory.json")
        data: Dict[str, Any] = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                data = json.load(f)

        cfg = Neo4jMemoryClientConfig(
            uri=os.environ.get("NEO4J_URI", data.get("uri", "bolt://localhost:7687")),
            user=os.environ.get("NEO4J_USER", data.get("user", "neo4j")),
            password=os.environ.get("NEO4J_PASSWORD", data.get("password", "")),
            database=os.environ.get("NEO4J_DATABASE", data.get("database", "neo4j")),
            llm=data.get("llm", "anthropic/claude-3-5-sonnet-latest"),
            embedding=data.get("embedding", "openai/text-embedding-3-small"),
            enabled=data.get("enabled", True),
        )
        self._wrapper = Neo4jMemoryClientWrapper(cfg)
        self._wrapper.connect()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return list(ALL_TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._wrapper is None:
            raise RuntimeError("Neo4jMemoryProvider has not been initialized.")

        if tool_name == "neo4j_store_fact":
            self._wrapper.store_fact(
                text=args["text"],
                metadata=args.get("metadata"),
            )
            return ""

        if tool_name == "neo4j_store_entity":
            self._wrapper.store_entity(
                name=args["name"],
                entity_type=args["entity_type"],
                observations=args["observations"],
                metadata=args.get("metadata"),
            )
            return ""

        if tool_name == "neo4j_store_preference":
            self._wrapper.store_preference(
                preference=args["preference"],
                metadata=args.get("metadata"),
            )
            return ""

        if tool_name == "neo4j_search":
            results = self._wrapper.search(
                query=args["query"],
                limit=args.get("limit", 10),
                namespace=args.get("namespace"),
            )
            return json.dumps(results)

        if tool_name == "neo4j_get_context":
            return self._wrapper.get_context(query=args["query"])

        raise ValueError(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        if self._wrapper is not None:
            self._wrapper.close()


def register(ctx) -> None:
    ctx.register_memory_provider(Neo4jMemoryProvider())

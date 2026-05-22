"""Neo4j memory client config and wrapper. Full impl in F2."""
from __future__ import annotations
from dataclasses import dataclass, field

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
    """Stub — implemented in F2."""
    def __init__(self, config: Neo4jMemoryClientConfig) -> None:
        self.config = config

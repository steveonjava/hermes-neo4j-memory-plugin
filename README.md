# hermes-neo4j-memory-plugin

Neo4j graph-backed long-term memory for [Hermes Agent](https://hermes-agent.nousresearch.com/).

This plugin integrates the [`neo4j-agent-memory`](https://pypi.org/project/neo4j-agent-memory/) library
with Hermes as a `MemoryProvider`, giving the agent durable, semantically-searchable memory backed by a
property graph.

---

## 1. Features

- **Five Hermes memory tools** — `neo4j_store_fact`, `neo4j_store_entity`, `neo4j_store_preference`,
  `neo4j_search`, `neo4j_get_context`.
- **Graph-backed storage** — entities, preferences, and facts persist in Neo4j across sessions.
- **Semantic search** — queries are embedded and matched against stored knowledge using vector similarity.
- **Pluggable LLM/embedding providers** — configure any provider string supported by `neo4j-agent-memory`
  (Anthropic, OpenAI, Groq, etc.).
- **Interactive setup wizard** — `hermes neo4j-memory setup` walks you through configuration with
  auto-detection of existing API keys.
- **Graceful degradation** — if Neo4j is unreachable, the provider silently marks itself unavailable
  and Hermes continues without memory rather than crashing.
- **Sync-safe async bridge** — all `neo4j-agent-memory` async calls are driven through a dedicated
  background event loop thread, keeping the Hermes synchronous tool-dispatch path clean.

---

## 2. Requirements

- Python 3.10+
- A running Neo4j 5.x instance (local Docker, AuraDB, or self-hosted)
- `neo4j-agent-memory>=0.4.0` (installed automatically)
- At least one LLM API key for entity extraction and embedding (Anthropic, OpenAI, Groq, etc.)

---

## 3. Installation

### Via Hermes plugin manager

```bash
hermes plugins install steveonjava/hermes-neo4j-memory-plugin
```

This installs the package, runs `pip install neo4j-agent-memory>=0.4.0`, and launches the setup wizard.

### Manual / development install

```bash
git clone https://github.com/steveonjava/hermes-neo4j-memory-plugin.git
cd hermes-neo4j-memory-plugin
pip install -e ".[dev]"
```

Then register the plugin in your Hermes config:

```yaml
# ~/.hermes/config.yaml
plugins:
  - path: /path/to/hermes-neo4j-memory-plugin
```

---

## 4. Quick-start: spin up Neo4j

If you do not already have a Neo4j instance, the fastest path is Docker:

```bash
docker run --name hermes-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/yourpassword \
  -e NEO4J_PLUGINS='["apoc"]' \
  -v hermes-neo4j-data:/data \
  -d neo4j:5.20-community
```

Wait ~30 seconds, then verify: open `http://localhost:7474` in your browser.

**AuraDB (free tier):**

1. Visit [console.neo4j.io](https://console.neo4j.io) and create a free instance.
2. Copy the connection URI (`neo4j+s://…`) and generated password.
3. Pass them to the setup wizard below.

---

## 5. Setup wizard

Run the interactive wizard to configure and validate the connection:

```bash
hermes neo4j-memory setup
```

The wizard:

1. Checks that `neo4j-agent-memory` is installed (offers to install if not).
2. Prompts for the Neo4j URI, username, password, and database.
3. Auto-detects any LLM API keys present in `~/.hermes/.env` or the environment and suggests suitable
   provider strings.
4. Smoke-tests the connection (connects, calls `get_context("hello")`).
5. On success, writes validated credentials to `~/.hermes/neo4j-memory.json`.
6. On failure, lets you retry with corrected settings.

Config is never written for a failed connection.

### Environment variable overrides

You can bypass the JSON config file by exporting:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=yourpassword
export NEO4J_DATABASE=neo4j   # optional, defaults to "neo4j"
```

Environment variables take precedence over `~/.hermes/neo4j-memory.json`.

---

## 6. Hermes memory tools

Once registered, the provider exposes five tools to the Hermes agent:

### `neo4j_store_fact`

Store a free-text factual statement in the graph.

```json
{
  "text": "The capital of France is Paris.",
  "metadata": {"source": "user", "confidence": "high"}
}
```

### `neo4j_store_entity`

Store a named entity with typed observations.

```json
{
  "name": "Stephen Chin",
  "entity_type": "person",
  "observations": ["Java Champion", "author of Raspberry Pi with Java"],
  "metadata": {"role": "developer"}
}
```

### `neo4j_store_preference`

Store a user preference.

```json
{
  "preference": "Prefers concise responses without unnecessary preamble.",
  "metadata": {"category": "communication"}
}
```

### `neo4j_search`

Unified semantic search over entities, preferences, and facts.

```json
{
  "query": "What do I know about Stephen?",
  "limit": 5
}
```

Returns a JSON list of result objects, each with a `type` field (`"entity"`, `"preference"`, or `"fact"`).

### `neo4j_get_context`

Assemble a context string ready to inject into an LLM prompt.

```json
{
  "query": "upcoming travel plans"
}
```

Returns a plain string summarising relevant memory entries.

---

## 7. Plugin registration

`plugin.yaml` declares the plugin entry point:

```yaml
name: neo4j-memory
version: "0.1.0"
description: "Neo4j graph-backed long-term memory for Hermes"
pip_dependencies:
  - "neo4j-agent-memory>=0.4.0"
hooks:
  post_setup: "neo4j_memory.cli:cmd_setup"
```

Hermes calls `neo4j_memory.register(ctx)` (defined in `neo4j_memory/__init__.py`) to register the
`Neo4jMemoryProvider` instance. Hermes then calls `provider.initialize()` at session start, which reads
`~/.hermes/neo4j-memory.json` (and env overrides) and connects to Neo4j.

---

## 8. Uninstall

```bash
hermes plugins remove hermes-neo4j-memory-plugin
```

This unregisters the plugin and its tools from your Hermes session. It does **not** touch your Neo4j data.

### Drop the Neo4j data

**Local Docker container:**

```bash
docker stop hermes-neo4j
docker rm hermes-neo4j
# Delete the persisted volume (all graph data permanently deleted):
docker volume rm hermes-neo4j-data
```

**Neo4j AuraDB:**

Log in to the [AuraDB console](https://console.neo4j.io), navigate to your instance, and click
**Delete Instance**. This is irreversible.

### Clean up Hermes config

```bash
rm -f ~/.hermes/neo4j-memory.json
```

---

## 9. Development

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for internals: the async↔sync bridge, tool schema
design, provider lifecycle, and test pyramid.

### Running the test suite

**Unit tests** (no Neo4j required):

```bash
pytest tests/unit/
```

**Integration tests** (requires a running Neo4j instance):

```bash
# Start the test sidecar
docker compose -f docker-compose.test.yml up -d

# Run integration tests (Bolt readiness is auto-waited by CI)
NEO4J_TEST_URI=bolt://localhost:7687 \
NEO4J_TEST_USER=neo4j \
NEO4J_TEST_PASSWORD=testpassword \
pytest tests/integration/ --override-ini="addopts=" -v

# Tear down when done
docker compose -f docker-compose.test.yml down -v
```

Tests that require an LLM/embedding API key (store_fact, store_entity, etc.) are skipped automatically
when no key is present; the connection and schema tests run with no API key.

### Lint and format

```bash
ruff check .
ruff format .
```

CI enforces both on every push.

### Contributing

1. Fork this repo, create a feature branch.
2. Add or update tests — unit tests for logic, integration tests for Neo4j interaction.
3. Run `pytest tests/unit/` locally and confirm green.
4. Open a PR. CI runs lint, unit tests (Python 3.10/3.11/3.12 matrix), and integration tests against a
   fresh Neo4j sidecar.

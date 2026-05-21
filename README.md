# hermes-neo4j-memory-plugin

> Sections 1–7 are authored by sibling tasks in this pipeline and will be assembled by the merge task.
> This file contains sections 8–9 as authored fragments for the assembler.

---

## 8. Uninstall

To remove the plugin from Hermes:

```bash
hermes plugins remove hermes-neo4j-memory-plugin
```

This unregisters the plugin and its tools from your Hermes session. It does **not** touch your Neo4j data.

### Drop the Neo4j data

**Local Docker container:**

```bash
# Stop and remove the container
docker stop hermes-neo4j
docker rm hermes-neo4j

# If you want to also delete the persisted volume (all graph data permanently deleted):
docker volume rm hermes-neo4j-data   # only if you started the container with -v hermes-neo4j-data:/data
```

**Neo4j AuraDB:**

Log in to the [AuraDB console](https://console.neo4j.io), navigate to your instance, and click **Delete Instance**. This is irreversible.

**Local Neo4j installation (non-Docker):**

Stop the Neo4j service, then delete the database directory (default `$NEO4J_HOME/data/databases/neo4j`) and the transactions directory (`$NEO4J_HOME/data/transactions/neo4j`):

```bash
neo4j stop
rm -rf $NEO4J_HOME/data/databases/neo4j $NEO4J_HOME/data/transactions/neo4j
```

### Clean up Hermes config

After uninstalling the plugin, you can remove the credentials from your Hermes config:

```bash
# Remove env vars (if you added them to your shell profile, remove them there too)
hermes config unset neo4j.uri
hermes config unset neo4j.user
hermes config unset neo4j.password
hermes config unset neo4j.database

# Remove the profile-scoped JSON credentials file
rm -f ~/.hermes/neo4j-memory.json
```

---

## 9. Development

### Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a detailed description of the plugin's internals: the async↔sync bridge, the five tool schemas, the provider-injectable LLM/embedding design, and schema management strategy.

### Upstream library

This plugin wraps [neo4j-labs/agent-memory](https://github.com/neo4j-labs/agent-memory) (PyPI: `neo4j-agent-memory`). If you are adding support for a new namespace (e.g. `short_term` or `reasoning`) or a new search strategy, check that project's changelog first — the library exposes those namespaces directly and the plugin just needs to wire them.

### Running the test suite

**Unit tests** (no Neo4j required):

```bash
pytest tests/unit/
```

**Integration tests** (requires a running Neo4j instance):

1. Start the test sidecar:

```bash
docker-compose -f docker-compose.test.yml up -d
```

This brings up `neo4j:5-community` on `bolt://localhost:7687` with credentials `neo4j/testpassword` as defined in `docker-compose.test.yml`.

2. Run integration tests:

```bash
pytest --run-integration
```

The `--run-integration` flag is required; without it, integration tests are skipped automatically (so `pytest tests/` is safe to run offline).

3. Tear down the sidecar when done:

```bash
docker-compose -f docker-compose.test.yml down -v
```

### Contributing

1. Fork this repo, create a feature branch.
2. Add or update tests — unit tests for logic, integration tests for any Neo4j interaction.
3. Run `pytest tests/unit/` locally (no Docker needed) and confirm green.
4. Open a PR. CI will run both unit and integration tests against a fresh Neo4j sidecar.

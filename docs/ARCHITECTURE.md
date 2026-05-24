# Architecture: hermes-neo4j-memory-plugin

This document describes the internal design of the plugin for contributors and integrators.

---

## Module layout

```
hermes-neo4j-memory-plugin/
├── neo4j_memory/
│   ├── __init__.py          # Neo4jMemoryProvider + 5 tool schemas + register()
│   ├── _async_bridge.py     # AsyncLoopThread: sync→async bridge
│   ├── client.py            # Neo4jMemoryClientConfig, Neo4jMemoryClientWrapper
│   └── cli.py               # cmd_setup() wizard, get_config_schema()
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── unit/                # 51 unit tests (no Neo4j, no network)
│   │   ├── test_async_bridge.py
│   │   ├── test_cli.py
│   │   ├── test_client_wrapper.py
│   │   └── test_provider.py
│   └── integration/         # 38 integration tests
│       ├── test_live_neo4j.py    # Live Neo4j connection / CRUD tests
│       └── test_setup_wizard.py  # Wizard flow tests (mocked smoke-test)
├── docker-compose.test.yml  # Neo4j 5.20 test sidecar
├── plugin.yaml              # Hermes plugin manifest
└── pyproject.toml           # Build config, dev deps, pytest config
```

---

## Component responsibilities

### `neo4j_memory/__init__.py` — Provider and tool schemas

`Neo4jMemoryProvider` implements the Hermes `MemoryProvider` interface:

| Method | Description |
|--------|-------------|
| `initialize(session_id, **kwargs)` | Reads `~/.hermes/neo4j-memory.json` (env overrides apply), constructs `Neo4jMemoryClientWrapper`, calls `connect()`. |
| `is_available()` | Delegates to `wrapper.is_available()`. Returns False if connection failed. |
| `get_tool_schemas()` | Returns the 5 tool schema dicts (JSON Schema format). |
| `handle_tool_call(tool_name, args)` | Dispatches to the appropriate wrapper method. |
| `shutdown()` | Calls `wrapper.close()` to drain the async loop thread. |
| `name` | Returns `"neo4j"`. |

The five tool schemas are module-level constants (`STORE_FACT_SCHEMA`, `STORE_ENTITY_SCHEMA`,
`STORE_PREFERENCE_SCHEMA`, `SEARCH_SCHEMA`, `GET_CONTEXT_SCHEMA`) so they can be inspected
independently of a live provider instance.

`register(ctx)` is the Hermes plugin entry point: it instantiates a `Neo4jMemoryProvider` and registers
it with `ctx.register_memory_provider(...)`.

---

### `neo4j_memory/client.py` — Sync wrapper

`Neo4jMemoryClientConfig` is a `dataclass` holding connection parameters:

- `uri` — Bolt URI (default `bolt://localhost:7687`)
- `user` / `password` / `database`
- `llm` — LLM provider string (e.g. `anthropic/claude-3-5-sonnet-latest`)
- `embedding` — Embedding provider string (e.g. `openai/text-embedding-3-small`)
- `enabled` — If False, `connect()` is a no-op (used for testing)

`Neo4jMemoryClientWrapper` wraps the async `neo4j-agent-memory` `MemoryClient`:

```
connect() → starts AsyncLoopThread → builds MemorySettings → MemoryClient.connect() on the loop thread
store_fact/entity/preference() → _run(client.long_term.add_*(), timeout)
search() → _run(_search(), timeout)  [aggregates entities + preferences + facts]
get_context() → _run(client.get_context(query), timeout)
close() → client.close() on loop thread → loop thread shutdown
```

All `_run()` calls route through `AsyncLoopThread.submit()`. Exceptions are caught and re-raised as
`RuntimeError` to give callers a consistent exception type.

**Graceful degradation**: `connect()` catches all exceptions, logs a warning, and sets
`_available = False`. Callers check `is_available()` before trusting results. The provider passes this
through so Hermes can silently disable memory rather than raising at session start.

---

### `neo4j_memory/_async_bridge.py` — Async↔sync bridge

`AsyncLoopThread` owns a dedicated `asyncio` event loop running in a background daemon thread. This
avoids the "no running event loop" error that occurs when calling `asyncio.run()` or `await` from a
sync Hermes tool-dispatch context.

```
AsyncLoopThread.start()  → creates loop, starts thread
AsyncLoopThread.submit(coro, timeout) → loop.call_soon_threadsafe(schedule_coro) + future.result(timeout)
AsyncLoopThread.shutdown(timeout) → cancel pending tasks, stop loop, join thread
```

Key design decisions:

- **Single loop thread**: all coroutines from a given wrapper share one loop, so connection state is
  consistent.
- **`start()` is idempotent**: safe to call multiple times; only the first call creates the thread.
- **`shutdown()` is idempotent**: safe to call after the thread has already exited.
- **`submit()` after `shutdown()` restarts the bridge**: this covers reconnect scenarios without
  requiring the caller to re-instantiate the wrapper.
- **Thread safety**: `submit()` uses `threading.Event` for synchronisation, not `concurrent.futures`,
  so it works correctly even when called from multiple threads concurrently (tested with 1000 concurrent
  submits in `test_async_bridge.py`).

---

### `neo4j_memory/cli.py` — Setup wizard

`cmd_setup(args)` is the interactive wizard invoked by `hermes neo4j-memory setup` (via the
`post_setup` hook in `plugin.yaml`).

Flow:

1. **Driver check** — `_ensure_driver_installed()` verifies `neo4j_agent_memory` is importable. If not,
   offers to `pip install neo4j-agent-memory>=0.4.0` via subprocess.
2. **Key detection** — `_detect_llm_keys()` scans `~/.hermes/.env` and the environment for known API
   key names (ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, etc.).
3. **Connection loop** — prompts URI / user / password / database. Calls `_smoke_test()` which
   instantiates a `Neo4jMemoryClientWrapper`, calls `connect()`, then `get_context("hello")`. On
   failure, re-prompts (without pre-filling the password) until the user succeeds or declines retry.
4. **Config write** — `_write_config()` writes the validated config to
   `$HERMES_HOME/neo4j-memory.json` as pretty-printed JSON.

`get_config_schema()` returns a list of key descriptors compatible with the Hermes generic config UI,
including `env_var` annotations so the UI can display override hints.

The wizard is fully testable without a live Neo4j instance by mocking `_smoke_test`. All 20 wizard
tests in `tests/integration/test_setup_wizard.py` run offline.

---

## Configuration model

Config precedence (highest to lowest):

1. Environment variables: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
2. `~/.hermes/neo4j-memory.json` (written by the setup wizard)
3. `Neo4jMemoryClientConfig` dataclass defaults

The JSON config file is read by `Neo4jMemoryProvider.initialize()` and by the CLI wizard.

---

## Test pyramid

| Layer | Count | Neo4j needed? | API keys needed? |
|-------|-------|---------------|-----------------|
| Unit (async bridge) | 10 | No | No |
| Unit (CLI wizard) | 12 | No | No |
| Unit (client wrapper) | 17 | No | No |
| Unit (provider) | 12 | No | No |
| Integration (wizard flow) | 20 | No (mocked) | No |
| Integration (live Neo4j — connect) | 6 | Yes | No |
| Integration (live Neo4j — store/search) | 10 | Yes | Yes (skipped if absent) |
| **Total** | **87** | | |

Unit tests mock all I/O. Integration tests that write/search data skip gracefully when no embedding API
key is present, so the CI suite can run against a live Neo4j sidecar without secrets.

---

## CI

`.github/workflows/test.yml` runs on every push and PR to `main`:

- **lint** job: `ruff check .` + `ruff format --check .` on Python 3.12.
- **unit** job: `pytest tests/unit/` on Python 3.10, 3.11, 3.12 in parallel.
- **integration** job: starts Neo4j 5.20 community via `docker-compose.test.yml`, waits for the Bolt
  port, then runs `pytest tests/integration/`.

---

## Risk register

| Risk | Mitigation |
|------|-----------|
| `neo4j-agent-memory` async API changes | Pin `>=0.4.0`; unit tests mock the client so API drift surfaces at import time |
| Embedding provider outage | Tests skip gracefully; provider degrades silently |
| Neo4j connection lost mid-session | `is_available()` checked before each call; RuntimeError surfaced to user |
| Blocking tool-dispatch thread | All async calls run on dedicated loop thread with configurable timeout |
| Credentials in logs | Password never logged; wizard uses `getpass` on TTY |

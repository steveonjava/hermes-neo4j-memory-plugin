"""Shared pytest fixtures for neo4j-memory tests."""


def pytest_addoption(parser):
    """Register --run-integration.

    pytest only calls this hook from conftest.py or a plugin, never from a test
    module, so it has to live here for CI's `--run-integration` to be recognised.
    """
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires a live Neo4j instance)",
    )

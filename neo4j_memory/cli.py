"""CLI commands for Neo4j memory provider management.

Handles: hermes neo4j-memory setup | status
"""

from __future__ import annotations

import getpass
import json
import os
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_hermes_home() -> Path:
    """Return the active Hermes home directory."""
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except ImportError:
        pass
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return Path(hermes_home)
    return Path.home() / ".hermes"


def _config_path() -> Path:
    return _get_hermes_home() / "neo4j-memory.json"


def _read_config() -> dict:
    path = _config_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _write_config(cfg: dict, path: Optional[Path] = None) -> None:
    path = path or _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _validate_neo4j_uri(uri: str) -> tuple[bool, str]:
    """Return whether *uri* is a plausible Neo4j connection URI."""
    if not uri or not uri.strip():
        return False, "URI is required"
    try:
        parsed = urlparse(uri)
    except ValueError as exc:
        return False, str(exc)
    if parsed.scheme not in {
        "bolt",
        "bolt+s",
        "bolt+ssc",
        "neo4j",
        "neo4j+s",
        "neo4j+ssc",
    }:
        return (
            False,
            "scheme must be bolt://, bolt+s://, bolt+ssc://, neo4j://, neo4j+s://, or neo4j+ssc://",
        )
    if not parsed.netloc:
        return False, "host is required"
    return True, ""


def _redact_secret(text: str, *secrets: str) -> str:
    """Redact known secret values from printable error text."""
    redacted = str(text or "")
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted


def _write_memory_provider_config(provider: str = "neo4j") -> None:
    """Set memory.provider in $HERMES_HOME/config.yaml without printing secrets."""
    config_path = _get_hermes_home() / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None

    if yaml is not None:
        existing: dict = {}
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        existing.setdefault("memory", {})["provider"] = provider
        config_path.write_text(
            yaml.safe_dump(existing, sort_keys=False),
            encoding="utf-8",
        )
        return

    if not config_path.exists() or not config_path.read_text(encoding="utf-8").strip():
        config_path.write_text(f"memory:\n  provider: {provider}\n", encoding="utf-8")
        return
    text = config_path.read_text(encoding="utf-8")
    if "memory:" not in text:
        text = text.rstrip() + f"\n\nmemory:\n  provider: {provider}\n"
    else:
        lines = text.splitlines()
        memory_idx = next(
            (idx for idx, line in enumerate(lines) if line.strip() == "memory:"), None
        )
        if memory_idx is None:
            text = text.rstrip() + f"\n\nmemory:\n  provider: {provider}\n"
            config_path.write_text(text, encoding="utf-8")
            return
        insert_at = memory_idx + 1
        provider_idx = None
        idx = memory_idx + 1
        while idx < len(lines):
            line = lines[idx]
            if line and not line.startswith((" ", "\t")):
                break
            if line.strip().startswith("provider:"):
                provider_idx = idx
                break
            idx += 1
        if provider_idx is None:
            lines.insert(insert_at, f"  provider: {provider}")
        else:
            indent = lines[provider_idx][
                : len(lines[provider_idx]) - len(lines[provider_idx].lstrip())
            ]
            lines[provider_idx] = f"{indent}provider: {provider}"
        text = "\n".join(lines) + "\n"
    config_path.write_text(text, encoding="utf-8")


def _prompt(label: str, default: Optional[str] = None, secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    sys.stdout.write(f"  {label}{suffix}: ")
    sys.stdout.flush()
    if secret:
        if sys.stdin.isatty():
            val = getpass.getpass(prompt="")
        else:
            # Non-TTY (piped input, test runners) — read plaintext
            val = _read_line()
    else:
        val = _read_line()
    return val or (default or "")


def _read_line() -> str:
    """Read one line from stdin; raise EOFError once input is exhausted.

    readline() returns "" forever at EOF, which the retry loop in cmd_setup
    would treat as an empty answer and spin on indefinitely.
    """
    line = sys.stdin.readline()
    if line == "":
        raise EOFError("input closed")
    return line.strip()


def _detect_llm_keys() -> dict[str, str]:
    """Scan ~/.hermes/.env and environment for known LLM API keys.

    Returns a dict mapping provider names to their detected key env var names.
    """
    hermes_env = _get_hermes_home() / ".env"
    env_vars: dict[str, str] = {}

    # Load from file first
    if hermes_env.exists():
        try:
            for line in hermes_env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                if val.strip():
                    env_vars[key] = val.strip()
        except Exception:
            pass

    # Merge with actual environment (env takes precedence for live values)
    for k, v in os.environ.items():
        if v:
            env_vars[k] = v

    # Detect known LLM provider keys
    detected: dict[str, str] = {}
    checks = [
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("OPENAI_API_KEY", "openai"),
        ("COHERE_API_KEY", "cohere"),
        ("TOGETHER_API_KEY", "together"),
        ("GROQ_API_KEY", "groq"),
        ("MISTRAL_API_KEY", "mistral"),
        ("LM_API_KEY", "openai-compatible"),  # LM Studio / Ollama style
    ]
    for env_var, provider_label in checks:
        val = env_vars.get(env_var, "")
        if val and val not in {"***", "CHANGEME", "your-key-here"}:
            detected[provider_label] = env_var
    return detected


def _suggest_llm_provider(detected_keys: dict[str, str]) -> Optional[str]:
    """Return a suggested default LLM provider string based on detected API keys."""
    if "anthropic" in detected_keys:
        return "anthropic/claude-3-5-sonnet-latest"
    if "openai" in detected_keys:
        return "openai/gpt-4o"
    if "groq" in detected_keys:
        return "groq/llama-3.1-70b-versatile"
    if "together" in detected_keys:
        return "together/meta-llama/Llama-3.1-70B-Instruct-Turbo"
    if "mistral" in detected_keys:
        return "mistral/mistral-large-latest"
    if "openai-compatible" in detected_keys:
        return "openai/local-model"
    return None


def _suggest_embedding_provider(detected_keys: dict[str, str]) -> str:
    """Return a suggested default embedding provider string."""
    if "openai" in detected_keys:
        return "openai/text-embedding-3-small"
    if "cohere" in detected_keys:
        return "cohere/embed-english-v3.0"
    # Anthropic doesn't have embeddings; fall back to openai default
    return "openai/text-embedding-3-small"


def _ensure_driver_installed() -> bool:
    """Check neo4j-agent-memory is importable; offer to install if not."""
    try:
        import neo4j_agent_memory  # noqa: F401

        return True
    except ImportError:
        pass

    print("  neo4j-agent-memory is not installed.")
    answer = _prompt("Install it now? (neo4j-agent-memory>=0.4.0)", default="y")
    if answer.lower() not in {"y", "yes"}:
        print("  Skipping. Run: pip install 'neo4j-agent-memory>=0.4.0'\n")
        return False

    import subprocess

    print("  Installing neo4j-agent-memory...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "neo4j-agent-memory>=0.4.0"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("  Installed.\n")
        return True
    else:
        print(f"  Install failed:\n{result.stderr.strip()}")
        print("  Run manually: pip install 'neo4j-agent-memory>=0.4.0'\n")
        return False


def _smoke_test(cfg: dict) -> tuple[bool, str]:
    """Attempt connect() + get_context("hello") using the provided config.

    Returns (success: bool, error_message: str).
    """
    try:
        from neo4j_memory.client import (
            Neo4jMemoryClientConfig,
            Neo4jMemoryClientWrapper,
        )
    except ImportError:
        return (
            False,
            "neo4j_memory.client module not found — ensure the plugin is installed correctly",
        )

    try:
        client_cfg = Neo4jMemoryClientConfig(
            uri=cfg["uri"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg.get("database", "neo4j"),
            llm=cfg.get("llm", "anthropic/claude-3-5-sonnet-latest"),
            embedding=cfg.get("embedding", "openai/text-embedding-3-small"),
            enabled=True,
        )
        wrapper = Neo4jMemoryClientWrapper(client_cfg)
        wrapper.connect()
        wrapper.get_context("hello")
        wrapper.close()
        return True, ""
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Public wizard entry point
# ---------------------------------------------------------------------------


def _cmd_setup(args) -> Optional[dict]:
    """Interactive Neo4j memory setup wizard.

    Prompts for connection details, detects existing API keys, smoke-tests the
    connection, and writes validated config to $HERMES_HOME/neo4j-memory.json.

    Returns the config dict on success, None on failure or abort.
    """
    cfg = _read_config()
    write_path = _config_path()

    print("\nNeo4j memory setup\n" + "─" * 40)
    print("  Neo4j memory gives Hermes persistent graph-backed long-term memory.")
    print(f"  Config will be written to: {write_path}")
    print()

    if not _ensure_driver_installed():
        return None

    # -------------------------------------------------------------------------
    # Detect existing LLM API keys
    # -------------------------------------------------------------------------
    detected_keys = _detect_llm_keys()

    # -------------------------------------------------------------------------
    # Connection loop — re-prompt on smoke-test failure
    # -------------------------------------------------------------------------
    while True:
        print("  Neo4j connection details:")

        current_uri = cfg.get("uri", "bolt://localhost:7687")
        new_uri = _prompt("  URI", default=current_uri)
        if not new_uri:
            new_uri = "bolt://localhost:7687"

        uri_ok, uri_error = _validate_neo4j_uri(new_uri)
        if not uri_ok:
            print(f"  Invalid Neo4j URI: {uri_error}")
            cfg["uri"] = "bolt://localhost:7687"
            continue

        current_user = cfg.get("user", "neo4j")
        new_user = _prompt("  Username", default=current_user)
        if not new_user:
            new_user = "neo4j"

        new_password = _prompt("  Password", secret=True)
        if not new_password and cfg.get("password"):
            print("    (keeping existing password)")
            new_password = cfg["password"]
        elif not new_password:
            print("  Password is required.")
            continue

        current_db = cfg.get("database", "neo4j")
        new_db = _prompt("  Database", default=current_db)
        if not new_db:
            new_db = "neo4j"

        # Build candidate config for smoke-test
        candidate = dict(cfg)
        candidate.update(
            {
                "uri": new_uri,
                "user": new_user,
                "password": new_password,
                "database": new_db,
            }
        )

        # ----------------------------------------------------------------
        # LLM / embedding provider (needed for entity extraction)
        # ----------------------------------------------------------------
        print()
        print("  LLM provider (used for entity extraction and graph reasoning):")

        if detected_keys:
            providers_found = ", ".join(
                f"{label} ({env})" for label, env in detected_keys.items()
            )
            print(f"    Detected API keys: {providers_found}")
        else:
            print(
                "  WARNING: No LLM API keys detected in ~/.hermes/.env or environment."
            )
            print(
                "    Entity extraction requires an LLM — set one before using memory."
            )
            print(
                "    Supported: ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, etc."
            )

        suggested_llm = _suggest_llm_provider(detected_keys)
        current_llm = (
            cfg.get("llm") or suggested_llm or "anthropic/claude-3-5-sonnet-latest"
        )
        new_llm = _prompt("  LLM provider string", default=current_llm)
        if not new_llm:
            new_llm = current_llm
        candidate["llm"] = new_llm

        suggested_embedding = _suggest_embedding_provider(detected_keys)
        current_embedding = cfg.get("embedding") or suggested_embedding
        new_embedding = _prompt(
            "  Embedding provider string", default=current_embedding
        )
        if not new_embedding:
            new_embedding = current_embedding
        candidate["embedding"] = new_embedding

        # ----------------------------------------------------------------
        # Smoke test
        # ----------------------------------------------------------------
        print()
        print("  Testing connection...", flush=True)
        ok, err = _smoke_test(candidate)
        if ok:
            print("  Connection successful.")
            break
        else:
            print(f"  Connection failed: {_redact_secret(err, new_password)}")
            retry = _prompt("  Retry with different settings?", default="y")
            if retry.lower() not in {"y", "yes"}:
                print("  Setup aborted. No config written.\n")
                return None
            print()
            # Loop back — re-prompt with failed values as defaults
            cfg = candidate
            cfg.pop("password", None)  # Don't pre-fill a failed password
            continue

    # -------------------------------------------------------------------------
    # Write validated config
    # -------------------------------------------------------------------------
    _write_config(candidate)
    _write_memory_provider_config("neo4j")
    print(f"  Config written to {write_path}")
    print("  Memory provider set to 'neo4j' in config.yaml\n")
    return candidate


def cmd_setup(args) -> Optional[dict]:
    """Run the setup wizard; abort cleanly if stdin closes mid-prompt."""
    try:
        return _cmd_setup(args)
    except EOFError:
        print("\n  Setup aborted: input closed. No config written.\n")
        return None


def post_setup(
    hermes_home: str | None = None, config: Optional[dict] = None
) -> Optional[dict]:
    """Hermes plugin post_setup hook entry point."""
    if hermes_home:
        os.environ["HERMES_HOME"] = str(hermes_home)
    return cmd_setup(config or {})


# ---------------------------------------------------------------------------
# get_config_schema — for generic Hermes config UI
# ---------------------------------------------------------------------------


def get_config_schema() -> list[dict]:
    """Return config key schema compatible with the Hermes generic config UI."""
    return [
        {
            "key": "uri",
            "description": "Neo4j Bolt URI",
            "default": "bolt://localhost:7687",
            "env_var": "NEO4J_URI",
        },
        {
            "key": "user",
            "description": "Neo4j username",
            "default": "neo4j",
            "env_var": "NEO4J_USER",
        },
        {
            "key": "password",
            "description": "Neo4j password",
            "secret": True,
            "env_var": "NEO4J_PASSWORD",
        },
        {
            "key": "database",
            "description": "Neo4j database name",
            "default": "neo4j",
            "env_var": "NEO4J_DATABASE",
        },
        {
            "key": "llm",
            "description": "LLM provider string for entity extraction (e.g. anthropic/claude-3-5-sonnet-latest)",
            "default": "anthropic/claude-3-5-sonnet-latest",
        },
        {
            "key": "embedding",
            "description": "Embedding provider string (e.g. openai/text-embedding-3-small)",
            "default": "openai/text-embedding-3-small",
        },
    ]

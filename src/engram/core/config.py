"""Configuration management using pydantic-settings."""

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Storage
    data_dir: Path = Path.home() / ".engram" / "data"

    # LLM for extraction and recall
    llm_model: str = "openai/gpt-6-luna"
    llm_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = (
        "medium"
    )
    llm_temperature: float = 0.0
    # OpenAI processing tier: "fast" (priority, ~2x price, lowest latency),
    # "default", or "" to omit the parameter.
    llm_service_tier: str = "fast"
    # Recall sits on the agent's critical path; extraction and upkeep can think
    # harder than recall does.
    recall_reasoning_effort: Literal[
        "none", "low", "medium", "high", "xhigh", "max"
    ] = "low"

    # Retrieval
    max_facts_per_agent: int = 40
    retrieval_timeout: float = 15.0

    # Lifecycle: ephemeral memories without an explicit expiry get this TTL.
    ephemeral_ttl_days: int = 45

    # Upkeep: background consolidation, anchor verification, and project
    # briefs inside the MCP server. Runs only when an LLM key is available.
    maintenance_enabled: bool = True
    maintenance_interval: float = 6 * 3600.0
    maintenance_concurrency: int = 4
    # Where upkeep looks for a project's git checkout when no working
    # directory has been recorded for it yet.
    repo_search_roots: list[Path] = [
        Path.home() / "dev",
        Path.home() / "code",
        Path.home() / "src",
        Path.home() / "projects",
        Path.home(),
    ]

    # Claude Code integration
    claude_projects_dir: Path = Path.home() / ".claude" / "projects"

    # Cross-machine sync (git-backed). Disabled by default; opt in via
    # ENGRAM_SYNC_ENABLED=true. ``sync_interval`` controls the background
    # auto-sync cadence (seconds) when enabled.
    sync_enabled: bool = False
    sync_interval: float = 300.0
    sync_timeout: float = 30.0

    # Logging
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="ENGRAM_",
        case_sensitive=False,
    )


# `engram serve --transport http` bind defaults: the shared local daemon address.
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 7422


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


class _LazySettingsProxy:
    """Lazily resolve settings on attribute access."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_settings(), name)


settings: Settings = _LazySettingsProxy()  # type: ignore[assignment]


_CACHE_EXPORT_RE = re.compile(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_CACHE_VAR_RE = re.compile(r"\$(\w+)|\$\{([^}]+)\}")
_PLACEHOLDER_RE = re.compile(r"\$\{?[A-Z_][A-Z0-9_]*\}?")


def _expand_cached_value(raw: str, values: dict[str, str]) -> str:
    """Expand shell-like variables in a cached export value."""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        quote = text[0]
        text = text[1:-1]
        if quote == "'":
            return text

    context = {**os.environ, **values}

    def replace(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2) or ""
        return context.get(key, "")

    return _CACHE_VAR_RE.sub(replace, text)


def load_cached_api_keys(cache_path: Path | None = None) -> dict[str, str]:
    """Load exported API keys from the dotfiles cache file."""
    path = cache_path or Path.home() / ".cache" / "api-keys"
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    with path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            match = _CACHE_EXPORT_RE.match(stripped)
            if not match:
                continue

            key, raw = match.groups()
            values[key] = _expand_cached_value(raw, values)

    return values


def _is_unresolved_env_placeholder(value: str | None) -> bool:
    """Return True when an env value contains a shell placeholder (e.g. $VAR or ${VAR})."""
    if value is None:
        return False
    return bool(_PLACEHOLDER_RE.search(value.strip()))


def ensure_openai_api_key(cache_path: Path | None = None) -> str | None:
    """Ensure OPENAI_API_KEY is available, loading from the dotfiles cache if needed."""
    current = os.environ.get("OPENAI_API_KEY")
    if current:
        if not _is_unresolved_env_placeholder(current):
            return current
        os.environ.pop("OPENAI_API_KEY", None)

    value = load_cached_api_keys(cache_path=cache_path).get("OPENAI_API_KEY")
    if value:
        os.environ["OPENAI_API_KEY"] = value
        return value

    return None


def configure_logging() -> None:
    """Configure logging based on settings."""
    log_level = get_settings().log_level
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # The HTTP stack logs every request at INFO; upkeep makes hundreds.
    for name in ("httpx", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)

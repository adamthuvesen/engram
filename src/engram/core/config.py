"""Configuration management using pydantic-settings."""

import logging
import os
import re
from functools import lru_cache
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Storage
    data_dir: Path = Field(
        default=Path.home() / ".engram" / "data", description="Storage directory."
    )

    # LLM for extraction and recall
    llm_model: str = Field(
        default="openai/gpt-6-luna",
        description="OpenAI model for extraction, recall fallback, and upkeep.",
    )
    llm_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = (
        Field(
            default="medium", description="Reasoning effort for extraction and upkeep."
        )
    )
    llm_temperature: float = Field(
        default=0.0, description="Temperature for models without reasoning effort."
    )
    # OpenAI processing tier: "fast" (priority, ~2x price, lowest latency),
    # "default", or "" to omit the parameter.
    llm_service_tier: str = Field(
        default="fast", description="OpenAI processing tier, or empty to omit."
    )
    # Recall sits on the agent's critical path; extraction and upkeep can think
    # harder than recall does.
    recall_reasoning_effort: Literal[
        "none", "low", "medium", "high", "xhigh", "max"
    ] = Field(default="low", description="Reasoning effort for recall calls.")

    # Retrieval
    max_facts_per_agent: int = Field(
        default=40, gt=0, description="Maximum facts supplied to a recall LLM call."
    )
    retrieval_timeout: float = Field(
        default=15.0,
        gt=0,
        allow_inf_nan=False,
        description="Recall provider timeout in seconds.",
    )

    # Lifecycle: ephemeral memories without an explicit expiry get this TTL.
    ephemeral_ttl_days: int = Field(
        default=45, gt=0, description="Default expiry in days for ephemeral memories."
    )

    # Upkeep: background consolidation, anchor verification, and project
    # briefs inside the MCP server. Runs only when an LLM key is available.
    maintenance_enabled: bool = Field(
        default=True, description="Run background memory upkeep in the MCP server."
    )
    maintenance_interval: float = Field(
        default=6 * 3600.0,
        gt=0,
        allow_inf_nan=False,
        description="Seconds between background upkeep runs.",
    )
    maintenance_concurrency: int = Field(
        default=4, gt=0, description="Maximum concurrent upkeep provider calls."
    )
    # Where upkeep looks for a project's git checkout when no working
    # directory has been recorded for it yet.
    repo_search_roots: list[Path] = Field(
        default=[
            Path.home() / "dev",
            Path.home() / "code",
            Path.home() / "src",
            Path.home() / "projects",
            Path.home(),
        ],
        description="Directories searched for project repositories, up to three levels deep.",
    )

    # Claude Code integration
    claude_projects_dir: Path = Field(
        default=Path.home() / ".claude" / "projects",
        description="Claude Code projects directory for memory imports.",
    )

    # Cross-machine sync (git-backed). Disabled by default; opt in via
    # ENGRAM_SYNC_ENABLED=true. ``sync_interval`` controls the background
    # auto-sync cadence (seconds) when enabled.
    sync_enabled: bool = Field(
        default=False, description="Run background Git sync in the MCP server."
    )
    sync_interval: float = Field(
        default=300.0,
        gt=0,
        allow_inf_nan=False,
        description="Seconds between background sync runs.",
    )
    sync_timeout: float = Field(
        default=30.0,
        gt=0,
        allow_inf_nan=False,
        description="Timeout in seconds for each sync Git invocation.",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Application logging level.")

    model_config = SettingsConfigDict(
        env_prefix="ENGRAM_",
        case_sensitive=False,
    )


# `engram serve --transport http` bind defaults: the shared local daemon address.
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 7422


_CURRENT_SETTINGS: ContextVar[Settings | None] = ContextVar(
    "engram_settings", default=None
)


@lru_cache(maxsize=1)
def _environment_settings() -> Settings:
    return Settings()


class _SettingsLoader:
    def __call__(self) -> Settings:
        return _CURRENT_SETTINGS.get() or _environment_settings()

    def cache_clear(self) -> None:
        _environment_settings.cache_clear()


get_settings = _SettingsLoader()


@contextmanager
def settings_scope(settings: Settings) -> Iterator[Settings]:
    token = _CURRENT_SETTINGS.set(settings)
    try:
        yield settings
    finally:
        _CURRENT_SETTINGS.reset(token)


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

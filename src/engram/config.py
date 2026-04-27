"""Configuration management using pydantic-settings."""

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Storage
    data_dir: Path = Path.home() / ".engram" / "data"

    # LLM for observer/search agents
    llm_model: str = "openai/gpt-5.4-mini"
    llm_temperature: float = 0.0

    # Retrieval
    max_facts_per_agent: int = 200
    retrieval_timeout: float = 15.0
    # Tier selector rules. "v2" (default) caps tier at 1 when the prefilter
    # returns fewer than `tier2_min_prefilter_count` facts. "v1" restores the
    # pre-small-corpus-cap behaviour. Unknown values warn and fall back to v2.
    tier_rules: str = "v2"
    # Under v2, tier-2 requires at least this many strictly-positive-scored
    # prefilter matches. Set to 0 to disable the cap even under v2.
    tier2_min_prefilter_count: int = 11

    # Synthesis
    synthesis_batch_size: int = 25
    synthesis_timeout: float = 30.0

    # Claude Code integration
    claude_projects_dir: Path = Path.home() / ".claude" / "projects"

    # Logging
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="ENGRAM_",
        case_sensitive=False,
    )

    @property
    def facts_path(self) -> Path:
        """Path to the facts JSONL file."""
        return self.data_dir / "facts.jsonl"

    @property
    def ingestion_log_path(self) -> Path:
        """Path to the ingestion audit log."""
        return self.data_dir / "ingestion_log.jsonl"


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
    for line in path.read_text().splitlines():
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
    if current and not _is_unresolved_env_placeholder(current):
        return current
    if current and _is_unresolved_env_placeholder(current):
        os.environ.pop("OPENAI_API_KEY", None)

    cached = load_cached_api_keys(cache_path=cache_path)
    value = cached.get("OPENAI_API_KEY")
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

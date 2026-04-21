"""Tests for API key fallback behavior."""

import os

from engram.config import ensure_openai_api_key, load_cached_api_keys


def test_load_cached_api_keys_parses_exports(tmp_path):
    cache_path = tmp_path / "api-keys"
    cache_path.write_text(
        "\n".join(
            [
                "export OPENAI_API_KEY_WORK='work-key'",
                "export OPENAI_API_KEY='generic-key'",
            ]
        )
    )

    cached = load_cached_api_keys(cache_path=cache_path)

    assert cached["OPENAI_API_KEY_WORK"] == "work-key"
    assert cached["OPENAI_API_KEY"] == "generic-key"


def test_load_cached_api_keys_expands_variable_references(tmp_path):
    cache_path = tmp_path / "api-keys"
    cache_path.write_text(
        "\n".join(
            [
                "export BASE_KEY='base-value'",
                'export OPENAI_API_KEY="$BASE_KEY"',
            ]
        )
    )

    cached = load_cached_api_keys(cache_path=cache_path)

    assert cached["OPENAI_API_KEY"] == "base-value"


def test_ensure_openai_api_key_loads_from_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "api-keys"
    cache_path.write_text("export OPENAI_API_KEY='cached-key'")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    resolved = ensure_openai_api_key(cache_path=cache_path)

    assert resolved == "cached-key"
    assert os.environ["OPENAI_API_KEY"] == "cached-key"


def test_ensure_openai_api_key_prefers_env_over_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "api-keys"
    cache_path.write_text("export OPENAI_API_KEY='cached-key'")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    resolved = ensure_openai_api_key(cache_path=cache_path)

    assert resolved == "env-key"


def test_ensure_openai_api_key_ignores_placeholder_env_values(tmp_path, monkeypatch):
    cache_path = tmp_path / "api-keys"
    cache_path.write_text("export OPENAI_API_KEY='cached-key'")
    monkeypatch.setenv("OPENAI_API_KEY", "${OPENAI_API_KEY}")

    resolved = ensure_openai_api_key(cache_path=cache_path)

    assert resolved == "cached-key"
    assert os.environ["OPENAI_API_KEY"] == "cached-key"


def test_ensure_openai_api_key_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    resolved = ensure_openai_api_key(cache_path=tmp_path / "nonexistent")

    assert resolved is None

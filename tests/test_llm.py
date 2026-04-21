"""Tests for the litellm wrapper — cache-marker plumbing and usage extraction."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from engram.llm import (
    Completion,
    _build_user_content,
    _extract_usage,
    _is_anthropic_model,
    complete_with_usage,
)


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


def test_is_anthropic_model_litellm_prefix():
    assert _is_anthropic_model("anthropic/claude-sonnet-4")


def test_is_anthropic_model_bare_claude():
    assert _is_anthropic_model("claude-3-5-haiku-latest")


def test_is_anthropic_model_openai_false():
    assert not _is_anthropic_model("openai/gpt-5.4-mini")


# ---------------------------------------------------------------------------
# User-content shaping
# ---------------------------------------------------------------------------


def test_build_user_content_openai_passes_through():
    """OpenAI-family calls never get a content list — implicit cache handles it."""
    prompt = "PREFIX\n\nQUERY: x"
    out = _build_user_content(prompt, cache_prefix="PREFIX", model="openai/gpt-5.4-mini")
    assert out == prompt


def test_build_user_content_anthropic_splits_prefix():
    """Anthropic-family calls get a two-part content list with cache_control."""
    prompt = "STORED FACTS:\n1. foo\n\nQUERY: x"
    prefix = "STORED FACTS:\n1. foo\n\n"
    out = _build_user_content(prompt, cache_prefix=prefix, model="anthropic/claude-sonnet-4")
    assert isinstance(out, list)
    assert out[0] == {
        "type": "text",
        "text": prefix,
        "cache_control": {"type": "ephemeral"},
    }
    assert out[1] == {"type": "text", "text": "QUERY: x"}


def test_build_user_content_anthropic_no_prefix_passthrough():
    """No cache_prefix → plain string even on Anthropic."""
    out = _build_user_content("hi", cache_prefix=None, model="anthropic/claude-sonnet-4")
    assert out == "hi"


def test_build_user_content_anthropic_prefix_equals_prompt():
    """Prefix == prompt → only the prefix block, no empty suffix entry."""
    prompt = "exact"
    out = _build_user_content(prompt, cache_prefix=prompt, model="anthropic/claude-sonnet-4")
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["cache_control"] == {"type": "ephemeral"}


def test_build_user_content_bad_prefix_falls_back_to_string():
    """If cache_prefix isn't actually a prefix, fall back to plain string."""
    out = _build_user_content("hello world", cache_prefix="nope", model="anthropic/claude-sonnet-4")
    assert out == "hello world"


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------


def test_extract_usage_openai_style():
    usage = SimpleNamespace(
        prompt_tokens=1234,
        prompt_tokens_details=SimpleNamespace(cached_tokens=800),
    )
    resp = SimpleNamespace(usage=usage)
    assert _extract_usage(resp) == (1234, 800)


def test_extract_usage_anthropic_style():
    usage = SimpleNamespace(
        prompt_tokens=1500,
        cache_read_input_tokens=1000,
    )
    resp = SimpleNamespace(usage=usage)
    assert _extract_usage(resp) == (1500, 1000)


def test_extract_usage_no_cached_field():
    usage = SimpleNamespace(prompt_tokens=500)
    resp = SimpleNamespace(usage=usage)
    input_tokens, cached = _extract_usage(resp)
    assert input_tokens == 500
    assert cached is None


def test_extract_usage_missing_entirely():
    resp = SimpleNamespace()
    assert _extract_usage(resp) == (None, None)


def test_extract_usage_dict_shape():
    """litellm sometimes returns usage as a dict."""
    resp = SimpleNamespace(
        usage={
            "prompt_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 60},
        }
    )
    assert _extract_usage(resp) == (100, 60)


# ---------------------------------------------------------------------------
# complete_with_usage — end-to-end with a mocked litellm.acompletion
# ---------------------------------------------------------------------------


def _mock_response(text: str, prompt_tokens: int | None = None, cached: int | None = None):
    msg = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=msg)
    usage = None
    if prompt_tokens is not None:
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached)
            if cached is not None
            else None,
        )
    return SimpleNamespace(choices=[choice], usage=usage)


@dataclass
class _MockLitellm:
    last_kwargs: dict | None = None

    def __post_init__(self):
        self.suppress_debug_info = True

    async def acompletion(self, **kwargs):
        self.last_kwargs = kwargs
        return _mock_response("answer text", prompt_tokens=123, cached=80)


@pytest.fixture
def fresh_settings(monkeypatch, tmp_path):
    """Isolate settings so env vars from the shell don't leak into tests."""
    from engram.config import get_settings

    monkeypatch.setenv("ENGRAM_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_complete_with_usage_returns_text_and_tokens(monkeypatch, fresh_settings):
    mock = _MockLitellm()
    monkeypatch.setattr("engram.llm._get_litellm", lambda: mock)
    monkeypatch.setattr("engram.llm.ensure_openai_api_key", lambda: "k")

    result = asyncio.run(
        complete_with_usage(
            prompt="hello",
            system="sys",
            model="openai/gpt-5.4-mini",
        )
    )
    assert isinstance(result, Completion)
    assert result.text == "answer text"
    assert result.input_tokens == 123
    assert result.cached_tokens == 80


def test_complete_with_usage_anthropic_sends_cache_control(monkeypatch, fresh_settings):
    mock = _MockLitellm()
    monkeypatch.setattr("engram.llm._get_litellm", lambda: mock)
    monkeypatch.setattr("engram.llm.ensure_openai_api_key", lambda: "k")

    prefix = "PREFIX_BLOCK\n\n"
    prompt = prefix + "QUERY: x"
    asyncio.run(
        complete_with_usage(
            prompt=prompt,
            system="sys",
            model="anthropic/claude-sonnet-4",
            cache_prefix=prefix,
        )
    )

    user_msg = mock.last_kwargs["messages"][-1]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert user_msg["content"][0]["text"] == prefix


def test_complete_with_usage_openai_stays_plain_string(monkeypatch, fresh_settings):
    mock = _MockLitellm()
    monkeypatch.setattr("engram.llm._get_litellm", lambda: mock)
    monkeypatch.setattr("engram.llm.ensure_openai_api_key", lambda: "k")

    prefix = "PREFIX_BLOCK\n\n"
    prompt = prefix + "QUERY: x"
    asyncio.run(
        complete_with_usage(
            prompt=prompt,
            system="sys",
            model="openai/gpt-5.4-mini",
            cache_prefix=prefix,
        )
    )

    user_msg = mock.last_kwargs["messages"][-1]
    assert user_msg["content"] == prompt


def test_complete_with_usage_missing_usage_returns_none(monkeypatch, fresh_settings):
    class _NoUsage(_MockLitellm):
        async def acompletion(self, **kwargs):
            self.last_kwargs = kwargs
            return _mock_response("answer", prompt_tokens=None)

    monkeypatch.setattr("engram.llm._get_litellm", lambda: _NoUsage())
    monkeypatch.setattr("engram.llm.ensure_openai_api_key", lambda: "k")

    result = asyncio.run(complete_with_usage(prompt="hi"))
    assert result.text == "answer"
    assert result.input_tokens is None
    assert result.cached_tokens is None

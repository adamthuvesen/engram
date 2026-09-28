"""Tests for the OpenAI client wrapper — request shaping and usage extraction."""

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from engram.llm.client import (
    Completion,
    _accepts_reasoning_effort,
    _extract_usage,
    _model_name,
    _openai_strict_schema,
    _response_format_for_model,
    complete_model,
    complete_with_usage,
)
from engram.core.structured_outputs import ExtractionResponse


class _StructuredAnswer(BaseModel):
    answer: int


# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------


def test_model_name_strips_openai_prefix():
    assert _model_name("openai/gpt-6-luna") == "gpt-6-luna"
    assert _model_name("gpt-6-luna") == "gpt-6-luna"


def test_model_name_rejects_other_providers():
    with pytest.raises(ValueError, match="OpenAI API directly"):
        _model_name("anthropic/claude-sonnet-5")


def test_reasoning_models_accept_reasoning_effort():
    assert _accepts_reasoning_effort("openai/gpt-6-luna")
    assert _accepts_reasoning_effort("gpt-5.4-mini")


def test_non_reasoning_models_keep_temperature_path():
    assert not _accepts_reasoning_effort("openai/gpt-4.1-mini")


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------


def test_extract_usage_openai_style():
    usage = SimpleNamespace(
        prompt_tokens=1234,
        prompt_tokens_details=SimpleNamespace(cached_tokens=800),
    )
    assert _extract_usage(SimpleNamespace(usage=usage)) == (1234, 800)


def test_extract_usage_no_cached_field():
    usage = SimpleNamespace(prompt_tokens=500, prompt_tokens_details=None)
    assert _extract_usage(SimpleNamespace(usage=usage)) == (500, None)


def test_extract_usage_missing_entirely():
    assert _extract_usage(SimpleNamespace()) == (None, None)


# ---------------------------------------------------------------------------
# complete_with_usage — end-to-end with a mocked OpenAI client
# ---------------------------------------------------------------------------


def _mock_response(
    text: str, prompt_tokens: int | None = None, cached: int | None = None
):
    msg = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=msg)
    usage = None
    if prompt_tokens is not None:
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        )
    return SimpleNamespace(choices=[choice], usage=usage)


@dataclass
class _MockClient:
    """Stands in for ``AsyncOpenAI``; records the last request."""

    response: SimpleNamespace = field(
        default_factory=lambda: _mock_response(
            "answer text", prompt_tokens=123, cached=80
        )
    )
    last_kwargs: dict | None = None

    def __post_init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return self.response


@pytest.fixture
def fresh_settings(monkeypatch, tmp_path):
    """Isolate settings so env vars from the shell don't leak into tests."""
    from engram.core.config import get_settings

    monkeypatch.setenv("ENGRAM_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(monkeypatch):
    mock = _MockClient()
    monkeypatch.setattr("engram.llm.client._client", lambda: mock)
    monkeypatch.setattr("engram.llm.client.ensure_openai_api_key", lambda: "k")
    return mock


def test_complete_with_usage_returns_text_and_tokens(client, fresh_settings):
    result = asyncio.run(
        complete_with_usage(prompt="hello", system="sys", model="openai/gpt-6-luna")
    )
    assert isinstance(result, Completion)
    assert result.text == "answer text"
    assert result.input_tokens == 123
    assert result.cached_tokens == 80
    assert client.last_kwargs["model"] == "gpt-6-luna"
    assert client.last_kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]


def test_complete_with_usage_reasoning_model_uses_reasoning_effort(
    client, monkeypatch, fresh_settings
):
    from engram.core.config import get_settings

    monkeypatch.setenv("ENGRAM_LLM_REASONING_EFFORT", "medium")
    get_settings.cache_clear()

    asyncio.run(complete_with_usage(prompt="hello", model="openai/gpt-6-luna"))

    assert client.last_kwargs["reasoning_effort"] == "medium"
    assert "temperature" not in client.last_kwargs
    assert client.last_kwargs["service_tier"] == "fast"


def test_reasoning_effort_override_wins(client, fresh_settings):
    asyncio.run(
        complete_with_usage(
            prompt="hello", model="openai/gpt-6-luna", reasoning_effort="low"
        )
    )
    assert client.last_kwargs["reasoning_effort"] == "low"


def test_complete_with_usage_other_models_keep_temperature(client, fresh_settings):
    asyncio.run(complete_with_usage(prompt="hello", model="openai/gpt-4.1-mini"))

    assert client.last_kwargs["temperature"] == 0.0
    assert "reasoning_effort" not in client.last_kwargs


def test_empty_service_tier_is_omitted(client, monkeypatch, fresh_settings):
    from engram.core.config import get_settings

    monkeypatch.setenv("ENGRAM_LLM_SERVICE_TIER", "")
    get_settings.cache_clear()

    asyncio.run(complete_with_usage(prompt="hello"))
    assert "service_tier" not in client.last_kwargs


def test_complete_with_usage_missing_usage_returns_none(client, fresh_settings):
    client.response = _mock_response("answer", prompt_tokens=None)

    result = asyncio.run(complete_with_usage(prompt="hi"))
    assert result.text == "answer"
    assert result.input_tokens is None
    assert result.cached_tokens is None


def test_response_format_for_model_uses_strict_json_schema():
    response_format = _response_format_for_model(_StructuredAnswer)

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "_StructuredAnswer"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == (
        _openai_strict_schema(_StructuredAnswer)
    )


def test_openai_strict_schema_requires_all_extraction_properties():
    schema = _openai_strict_schema(ExtractionResponse)
    fact_schema = schema["$defs"]["ExtractedFact"]

    assert schema["required"] == ["facts", "retire", "excluded_claims"]
    assert fact_schema["required"] == list(fact_schema["properties"])
    assert "tags" in fact_schema["required"]
    assert "default" not in _schema_keys(schema)


def _schema_keys(schema: object) -> set[str]:
    if isinstance(schema, dict):
        keys = set(schema)
        for value in schema.values():
            keys.update(_schema_keys(value))
        return keys
    if isinstance(schema, list):
        keys = set()
        for item in schema:
            keys.update(_schema_keys(item))
        return keys
    return set()


def test_complete_model_returns_validated_model(monkeypatch):
    captured_kwargs: dict = {}

    async def fake_complete(**kwargs):
        captured_kwargs.update(kwargs)
        return '{"answer": 42}'

    monkeypatch.setattr("engram.llm.client.complete", fake_complete)

    result = asyncio.run(
        complete_model(
            prompt="test",
            system="sys",
            response_model=_StructuredAnswer,
        )
    )

    assert result == _StructuredAnswer(answer=42)
    assert captured_kwargs["response_format"] == _response_format_for_model(
        _StructuredAnswer
    )


def test_complete_model_invalid_json_raises_validation_error(monkeypatch):
    async def fake_complete(**kwargs):
        return "This is not JSON at all!"

    monkeypatch.setattr("engram.llm.client.complete", fake_complete)

    with pytest.raises(ValidationError):
        asyncio.run(
            complete_model(
                prompt="test",
                system="sys",
                response_model=_StructuredAnswer,
            )
        )

"""LLM client wrapper using litellm for multi-provider support."""

import copy
import importlib
import logging
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from engram.core.config import ensure_openai_api_key, get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


@dataclass
class Completion:
    """LLM completion result with optional usage counters.

    input_tokens and cached_tokens are None when the provider does not report
    them. Callers aggregating usage should treat None as "unknown" and skip it.
    """

    text: str
    input_tokens: int | None = None
    cached_tokens: int | None = None


def _get_litellm():
    """Import litellm lazily so tests can patch the wrapper without the dependency."""
    litellm = importlib.import_module("litellm")
    setattr(litellm, "suppress_debug_info", True)
    return litellm


def _is_anthropic_model(model: str) -> bool:
    """Detect Anthropic-family model strings (with or without litellm prefix)."""
    lower = model.lower()
    return lower.startswith("anthropic/") or "claude" in lower


def _is_gpt_5_6_model(model: str) -> bool:
    """Detect GPT-5.6 model names with or without a LiteLLM provider prefix."""
    return any(part.startswith("gpt-5.6") for part in model.lower().split("/"))


def _build_user_content(prompt: str, cache_prefix: str | None, model: str):
    """Build the user-message content.

    For Anthropic-family models with a cache_prefix that is an actual prefix of
    `prompt`, return a two-part content list with cache_control on the prefix.
    Otherwise return the plain string.
    """
    if cache_prefix and _is_anthropic_model(model) and prompt.startswith(cache_prefix):
        suffix = prompt[len(cache_prefix) :]
        parts = [
            {
                "type": "text",
                "text": cache_prefix,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if suffix:
            parts.append({"type": "text", "text": suffix})
        return parts
    return prompt


def _extract_usage(response) -> tuple[int | None, int | None]:
    """Pull (input_tokens, cached_tokens) out of a litellm response.

    Handles OpenAI-style (`usage.prompt_tokens_details.cached_tokens`) and
    Anthropic-style (`usage.cache_read_input_tokens`) shapes. Returns (None, None)
    when usage is unavailable.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None

    def _get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    input_tokens = _get(usage, "prompt_tokens")
    cached = _get(usage, "cache_read_input_tokens")
    if cached is None:
        details = _get(usage, "prompt_tokens_details")
        if details is not None:
            cached = _get(details, "cached_tokens")
    return input_tokens, cached


async def complete(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float | None = None,
    response_format: dict | None = None,
    cache_prefix: str | None = None,
) -> str:
    """Make an async LLM completion call via litellm.

    When `cache_prefix` is a byte-prefix of `prompt` and the configured model is
    Anthropic-family, the prefix is sent as a `cache_control: ephemeral` block so
    subsequent calls with the same prefix hit Anthropic's prompt cache. For
    OpenAI-family models the prefix is ignored — OpenAI's implicit prefix cache
    kicks in automatically when prompts share a leading run of tokens.
    """
    result = await complete_with_usage(
        prompt=prompt,
        system=system,
        model=model,
        temperature=temperature,
        response_format=response_format,
        cache_prefix=cache_prefix,
    )
    return result.text


async def complete_model(
    prompt: str,
    system: str,
    response_model: type[T],
    model: str | None = None,
) -> T:
    """Make an LLM call expecting JSON matching a Pydantic model."""
    raw = await complete(
        prompt=prompt,
        system=system,
        model=model,
        response_format=_response_format_for_model(response_model),
    )
    return response_model.model_validate_json(raw or "")


def _response_format_for_model(response_model: type[BaseModel]) -> dict:
    """Build LiteLLM's strict JSON-schema response_format payload."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "strict": True,
            "schema": _openai_strict_schema(response_model),
        },
    }


def _openai_strict_schema(response_model: type[BaseModel]) -> dict:
    """Return a JSON schema compatible with OpenAI strict structured outputs."""
    schema = copy.deepcopy(response_model.model_json_schema())
    _require_all_object_properties(schema)
    return schema


def _require_all_object_properties(schema: object) -> None:
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            schema["additionalProperties"] = False
            schema["required"] = list(properties)

        schema.pop("default", None)

        for value in schema.values():
            _require_all_object_properties(value)
    elif isinstance(schema, list):
        for item in schema:
            _require_all_object_properties(item)


async def complete_with_usage(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float | None = None,
    response_format: dict | None = None,
    cache_prefix: str | None = None,
) -> Completion:
    """Like `complete`, but also returns reported token usage."""
    litellm = _get_litellm()
    ensure_openai_api_key()
    settings = get_settings()
    model = model or settings.llm_model
    temperature = temperature if temperature is not None else settings.llm_temperature

    user_content = _build_user_content(prompt, cache_prefix, model)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "num_retries": 2,
    }
    if _is_gpt_5_6_model(model):
        kwargs["reasoning_effort"] = settings.llm_reasoning_effort
    else:
        kwargs["temperature"] = temperature
    if response_format:
        kwargs["response_format"] = response_format

    response = await litellm.acompletion(**kwargs)
    text = response.choices[0].message.content
    input_tokens, cached_tokens = _extract_usage(response)
    return Completion(
        text=text,
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
    )

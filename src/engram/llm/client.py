"""LLM client: OpenAI Chat Completions through the official SDK.

Engram talks to OpenAI directly rather than through a multi-provider layer:
the SDK imports in a fraction of the time, adds no per-call overhead, and
supports new models and request parameters (``reasoning_effort``,
``service_tier``) the day they ship.
"""

import asyncio
import copy
import logging
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from engram.core.config import ensure_openai_api_key, get_settings

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_MAX_RETRIES = 2
# Model families that take ``reasoning_effort`` instead of ``temperature``.
_REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")
# One client per event loop: the SDK's HTTP pool is bound to the loop it
# first ran on, and the CLI and tests start fresh loops.
_CLIENTS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncOpenAI]" = (
    weakref.WeakKeyDictionary()
)


@dataclass
class Completion:
    """LLM completion result with optional usage counters.

    input_tokens and cached_tokens are None when the provider does not report
    them. Callers aggregating usage should treat None as "unknown" and skip it.
    """

    text: str
    input_tokens: int | None = None
    cached_tokens: int | None = None


def _client() -> "AsyncOpenAI":
    """The OpenAI client for the running event loop (imported lazily)."""
    loop = asyncio.get_running_loop()
    client = _CLIENTS.get(loop)
    if client is None:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(max_retries=_MAX_RETRIES)
        _CLIENTS[loop] = client
    return client


def _model_name(model: str) -> str:
    """Strip an ``openai/`` prefix; reject other providers loudly."""
    provider, _, name = model.rpartition("/")
    if provider and provider != "openai":
        raise ValueError(
            f"Unsupported model {model!r}: Engram calls the OpenAI API directly. "
            "Set ENGRAM_LLM_MODEL to an OpenAI model such as openai/gpt-6-luna."
        )
    return name


def _accepts_reasoning_effort(model: str) -> bool:
    """Reasoning models take ``reasoning_effort``; others take temperature."""
    return _model_name(model).lower().startswith(_REASONING_PREFIXES)


def _extract_usage(response) -> tuple[int | None, int | None]:
    """Pull ``(input_tokens, cached_tokens)`` out of a completion response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    return getattr(usage, "prompt_tokens", None), cached


async def complete(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float | None = None,
    response_format: dict | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Make an async LLM completion call and return the text.

    OpenAI caches shared prompt prefixes automatically, so callers get cache
    hits by keeping stable content (system prompt, instructions) first.
    """
    result = await complete_with_usage(
        prompt=prompt,
        system=system,
        model=model,
        temperature=temperature,
        response_format=response_format,
        reasoning_effort=reasoning_effort,
    )
    return result.text


async def complete_model(
    prompt: str,
    system: str,
    response_model: type[T],
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> T:
    """Make an LLM call expecting JSON matching a Pydantic model."""
    raw = await complete(
        prompt=prompt,
        system=system,
        model=model,
        response_format=_response_format_for_model(response_model),
        reasoning_effort=reasoning_effort,
    )
    return response_model.model_validate_json(raw or "")


def _response_format_for_model(response_model: type[BaseModel]) -> dict:
    """Build OpenAI's strict JSON-schema response_format payload."""
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
    reasoning_effort: str | None = None,
) -> Completion:
    """Like `complete`, but also returns reported token usage.

    ``reasoning_effort`` overrides ``ENGRAM_LLM_REASONING_EFFORT`` for this
    call (reasoning models only).
    """
    ensure_openai_api_key()
    settings = get_settings()
    model = model or settings.llm_model

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {"model": _model_name(model), "messages": messages}
    if _accepts_reasoning_effort(model):
        kwargs["reasoning_effort"] = reasoning_effort or settings.llm_reasoning_effort
    else:
        kwargs["temperature"] = (
            temperature if temperature is not None else settings.llm_temperature
        )
    if settings.llm_service_tier:
        kwargs["service_tier"] = settings.llm_service_tier
    if response_format:
        kwargs["response_format"] = response_format

    response = await _client().chat.completions.create(**kwargs)
    input_tokens, cached_tokens = _extract_usage(response)
    return Completion(
        text=response.choices[0].message.content or "",
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
    )

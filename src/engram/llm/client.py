"""Typed, closeable OpenAI completion clients and compatibility helpers."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import BaseModel

from engram.core.config import ensure_openai_api_key, get_settings

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionMessage

T = TypeVar("T", bound=BaseModel)
_MAX_RETRIES = 2
_REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


class ProviderResponseError(ValueError):
    """The provider refused or returned an incomplete response."""


def is_provider_error(exc: BaseException) -> bool:
    from openai import OpenAIError

    return isinstance(exc, (OpenAIError, ProviderResponseError))


@dataclass
class Completion:
    """Provider text and optional usage. Missing counters mean unknown."""

    text: str
    input_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class ModelCompletion[T: BaseModel]:
    value: T
    input_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None


class CompletionClient(Protocol):
    """Completion dependency shared by extraction, recall, and maintenance."""

    @property
    def available(self) -> bool: ...

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> Completion: ...

    async def complete_model(
        self,
        prompt: str,
        system: str,
        response_model: type[T],
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> ModelCompletion[T]: ...

    async def aclose(self) -> None: ...


_CURRENT_CLIENT: ContextVar[CompletionClient | None] = ContextVar(
    "engram_completion_client", default=None
)


@contextmanager
def completion_client(client: CompletionClient) -> Generator[CompletionClient]:
    """Bind a caller-owned client. The caller closes it after dependent tasks stop."""
    token = _CURRENT_CLIENT.set(client)
    try:
        yield client
    finally:
        _CURRENT_CLIENT.reset(token)


def provider_available() -> bool:
    client = _CURRENT_CLIENT.get()
    return client.available if client is not None else ensure_openai_api_key() is not None


def _client() -> AsyncOpenAI:
    from openai import AsyncOpenAI

    return AsyncOpenAI(max_retries=_MAX_RETRIES)


def _model_name(model: str) -> str:
    provider, _, name = model.rpartition("/")
    if provider and provider != "openai":
        raise ValueError(
            f"Unsupported model {model!r}: Engram calls the OpenAI API directly. "
            "Set ENGRAM_LLM_MODEL to an OpenAI model such as openai/gpt-6-luna."
        )
    return name


def _accepts_reasoning_effort(model: str) -> bool:
    return _model_name(model).lower().startswith(_REASONING_PREFIXES)


def _extract_usage(response: ChatCompletion) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    return getattr(usage, "prompt_tokens", None), cached


def _request(
    prompt: str,
    system: str,
    model: str | None,
    temperature: float | None,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    ensure_openai_api_key()
    settings = get_settings()
    model = model or settings.llm_model
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs: dict[str, Any] = {"model": _model_name(model), "messages": messages}
    if _accepts_reasoning_effort(model):
        kwargs["reasoning_effort"] = reasoning_effort or settings.llm_reasoning_effort
    else:
        kwargs["temperature"] = temperature if temperature is not None else settings.llm_temperature
    if settings.llm_service_tier:
        kwargs["service_tier"] = settings.llm_service_tier
    return kwargs


def _message(response: ChatCompletion) -> ChatCompletionMessage:
    if not response.choices:
        raise ProviderResponseError("Provider returned no completion choices.")
    choice = response.choices[0]
    message = choice.message
    if getattr(message, "refusal", None):
        raise ProviderResponseError("Provider refused the completion.")
    finish_reason = getattr(choice, "finish_reason", "stop")
    if finish_reason != "stop":
        raise ProviderResponseError(f"Provider completion ended with {finish_reason!r}.")
    if not message.content or not message.content.strip():
        raise ProviderResponseError("Provider returned empty completion text.")
    return message


class OpenAICompletionClient:
    """One lazy SDK connection pool, owned and closed by its runtime."""

    def __init__(self) -> None:
        self._sdk: AsyncOpenAI | None = None
        self._closed = False

    @property
    def available(self) -> bool:
        return not self._closed and ensure_openai_api_key() is not None

    def _connection(self) -> AsyncOpenAI:
        if self._closed:
            raise RuntimeError("Completion client is closed.")
        if self._sdk is None:
            self._sdk = _client()
        return self._sdk

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            if self._sdk is not None:
                await self._sdk.close()

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> Completion:
        kwargs = _request(prompt, system, model, temperature, reasoning_effort)
        if response_format:
            kwargs["response_format"] = response_format
        response = await self._connection().chat.completions.create(**kwargs)
        message = _message(response)
        assert message.content is not None
        input_tokens, cached_tokens = _extract_usage(response)
        return Completion(
            text=message.content,
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=getattr(getattr(response, "usage", None), "completion_tokens", None),
        )

    async def complete_model(
        self,
        prompt: str,
        system: str,
        response_model: type[T],
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> ModelCompletion[T]:
        kwargs = _request(prompt, system, model, None, reasoning_effort)
        response = await self._connection().chat.completions.parse(
            **kwargs, response_format=response_model
        )
        message = _message(response)
        parsed = getattr(message, "parsed", None)
        if parsed is None:
            raise ProviderResponseError("Provider returned no parsed completion.")
        value = response_model.model_validate(parsed)
        input_tokens, cached_tokens = _extract_usage(response)
        return ModelCompletion(
            value=value,
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=getattr(getattr(response, "usage", None), "completion_tokens", None),
        )


async def complete(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float | None = None,
    response_format: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
) -> str:
    result = await complete_with_usage(
        prompt, system, model, temperature, response_format, reasoning_effort
    )
    return result.text


async def complete_with_usage(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float | None = None,
    response_format: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
) -> Completion:
    client = _CURRENT_CLIENT.get()
    owned = client is None
    client = client or OpenAICompletionClient()
    try:
        return await client.complete(
            prompt, system, model, temperature, response_format, reasoning_effort
        )
    finally:
        if owned:
            await client.aclose()


async def complete_model_with_usage[T: BaseModel](
    prompt: str,
    system: str,
    response_model: type[T],
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> ModelCompletion[T]:
    client = _CURRENT_CLIENT.get()
    owned = client is None
    client = client or OpenAICompletionClient()
    try:
        return await client.complete_model(prompt, system, response_model, model, reasoning_effort)
    finally:
        if owned:
            await client.aclose()


async def complete_model[T: BaseModel](
    prompt: str,
    system: str,
    response_model: type[T],
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> T:
    result = await complete_model_with_usage(
        prompt, system, response_model, model, reasoning_effort
    )
    return result.value


def response_format_for_model(response_model: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "strict": True,
            "schema": _openai_strict_schema(response_model),
        },
    }


def _openai_strict_schema(response_model: type[BaseModel]) -> dict[str, Any]:
    # The SDK owns schema conversion for both parsing and legacy text completions.
    from openai.lib._pydantic import to_strict_json_schema

    return to_strict_json_schema(response_model)

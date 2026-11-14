"""Deterministic provider responses with no network or key discovery."""

from collections import deque
from typing import Any, TypeVar

from pydantic import BaseModel

from engram.llm import Completion, ModelCompletion

T = TypeVar("T", bound=BaseModel)


class FixtureCompletionClient:
    """Consume fixture responses. An empty fixture disables provider selection."""

    def __init__(self, responses: list[tuple[str, int | None, int | None]] | None = None) -> None:
        self._responses = deque(responses or [])
        self.available = bool(self._responses)

    def _next(self) -> Completion:
        if not self._responses:
            raise RuntimeError("Evaluation fixture exhausted its provider responses.")
        text, input_tokens, cached_tokens = self._responses.popleft()
        return Completion(text, input_tokens, cached_tokens)

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> Completion:
        return self._next()

    async def complete_model(
        self,
        prompt: str,
        system: str,
        response_model: type[T],
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> ModelCompletion[T]:
        completion = self._next()
        return ModelCompletion(
            value=response_model.model_validate_json(completion.text),
            input_tokens=completion.input_tokens,
            cached_tokens=completion.cached_tokens,
            output_tokens=completion.output_tokens,
        )

    async def aclose(self) -> None:
        return None

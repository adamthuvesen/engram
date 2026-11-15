"""Operation results, exit codes, and shared validation failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue, ValidationError

from engram.core.interfaces import (
    Envelope,
    EnvelopeError,
    EnvelopeStatus,
    ErrorCode,
    not_found_error,
    validation_error,
)
from engram.core.models import (
    FactBase,
    FactCategory,
    MemoryCandidate,
)
from engram.core.operation_models import (
    CandidatePayload,
    FactPayload,
)
from engram.llm import is_provider_error
from engram.storage.store import StoreIntegrityError
from engram.storage.sync import (
    SyncError,
)

EXIT_OK = 0


EXIT_VALIDATION = 1


EXIT_NOT_FOUND = 2


EXIT_RUNTIME = 3


EXIT_DOCTOR_ERROR = 4


@dataclass
class OperationResult:
    """Result shared by CLI and MCP adapters."""

    envelope: Envelope
    text: str
    exit_code: int = EXIT_OK

    def __post_init__(self) -> None:
        if self.exit_code == EXIT_OK and self.envelope.status is not EnvelopeStatus.ok:
            codes = {error.code for error in self.envelope.errors}
            self.exit_code = (
                EXIT_VALIDATION
                if codes == {ErrorCode.validation_error}
                else EXIT_NOT_FOUND
                if codes == {ErrorCode.not_found}
                else EXIT_RUNTIME
            )

    @property
    def is_error(self) -> bool:
        return self.envelope.status is not EnvelopeStatus.ok

    def render(self, *, as_json: bool) -> str:
        return self.envelope.to_json() if as_json else self.text


def failure_result(
    error: EnvelopeError, text: str, exit_code: int = EXIT_RUNTIME
) -> OperationResult:
    return OperationResult(
        envelope=Envelope.failure(error),
        text=text,
        exit_code=exit_code,
    )


def validation_result(
    message: str,
    *,
    text: str | None = None,
    ids: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> OperationResult:
    return failure_result(
        validation_error(message, ids=ids, details=details),
        text or message,
        EXIT_VALIDATION,
    )


def missing_result(
    message: str, *, text: str | None = None, ids: list[str] | None = None
) -> OperationResult:
    return failure_result(
        not_found_error(message, ids=ids),
        text or message,
        EXIT_NOT_FOUND,
    )


def exception_result(exc: Exception) -> OperationResult:
    if is_provider_error(exc):
        code = ErrorCode.provider_error
    elif isinstance(exc, TimeoutError):
        code = ErrorCode.timeout
    elif isinstance(exc, (OSError, StoreIntegrityError, SyncError)):
        code = ErrorCode.storage_error
    elif isinstance(exc, (ValidationError, ValueError)):
        code = ErrorCode.validation_error
    else:
        code = ErrorCode.internal_error
    return failure_result(
        EnvelopeError(code=code, message=str(exc), details={"exception_type": type(exc).__name__}),
        text=str(exc),
        exit_code=EXIT_VALIDATION if code is ErrorCode.validation_error else EXIT_RUNTIME,
    )


def category_from_value(value: str | None) -> FactCategory | None:
    if value is None:
        return None
    try:
        return FactCategory(value)
    except ValueError:
        return None


def invalid_category_result(value: str) -> OperationResult:
    valid = ", ".join(c.value for c in FactCategory)
    return validation_result(
        f"Invalid category: {value}",
        text=f"Invalid category: {value}. Use one of: {valid}",
        details={"valid": [c.value for c in FactCategory]},
    )


def invalid_format_result(value: str) -> OperationResult:
    return validation_result(
        f"Unsupported format: {value}. Use 'text' or 'json'.",
        text=f"Unsupported format: {value}. Use 'text' or 'json'.",
        details={"parameter": "format", "value": value},
    )


def invalid_positive_int_result(parameter: str) -> OperationResult:
    message = f"{parameter} must be greater than zero"
    return validation_result(message, details={"parameter": parameter})


def fact_payload(fact: FactBase) -> dict[str, JsonValue]:
    return FactPayload.model_validate(fact).model_dump(mode="json")


def candidate_payload(candidate: MemoryCandidate) -> dict[str, JsonValue]:
    return CandidatePayload.model_validate(candidate).model_dump(mode="json")


def outcome_envelope(
    data: dict[str, JsonValue], errors: list[EnvelopeError], *, applied: bool = False
) -> Envelope:
    return Envelope(
        status=EnvelopeStatus.partial
        if errors and applied
        else EnvelopeStatus.error
        if errors
        else EnvelopeStatus.ok,
        data=data,
        errors=errors,
    )


def invalid_mode_result(value: str, valid: tuple[str, ...]) -> OperationResult:
    message = f"Unsupported mode: {value}. Use one of: {', '.join(valid)}."
    return validation_result(message, details={"parameter": "mode", "value": value})

"""Shared structured response envelopes and error codes for agent-facing surfaces.

These models are the stable contract that MCP tools and CLI commands use when
callers opt in to JSON output. The default text-oriented responses on existing
tools are unchanged; structured mode is enabled by an explicit parameter on
each tool that supports it.

Design notes:
- `Envelope` is the single shape returned by structured operations. It carries
  ``status`` (machine-branchable), ``data`` (operation-specific payload),
  ``warnings`` (non-fatal advisories), ``errors`` (machine-readable failure
  details), and ``meta`` (limits/truncation/pagination info).
- ``ErrorCode`` is a small, stable taxonomy. New codes can be added but
  existing values must not be renamed.
- ``Warning`` and ``EnvelopeError`` carry codes plus optional human-readable
  messages and structured ``details``/``ids`` so agents can branch without
  scraping prose.

Backward compatibility is enforced by tests in ``tests/test_interfaces.py``
that exercise JSON round-tripping and stable enum values.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EnvelopeStatus(str, Enum):
    ok = "ok"
    error = "error"
    partial = "partial"


class ErrorCode(str, Enum):
    """Stable error taxonomy for machine-readable failure handling.

    Agents branch on these values; do not rename existing entries. New codes
    can be added when a clearly distinct category emerges.
    """

    validation_error = "validation_error"
    not_found = "not_found"
    provider_error = "provider_error"
    storage_error = "storage_error"
    conflict = "conflict"
    timeout = "timeout"
    internal_error = "internal_error"


class WarningCode(str, Enum):
    """Stable warning taxonomy for non-fatal advisories surfaced in recall."""

    stale_fact = "stale_fact"
    superseded_fact = "superseded_fact"
    forgotten_fact = "forgotten_fact"
    conflicting_facts = "conflicting_facts"
    truncated_output = "truncated_output"
    provider_unavailable = "provider_unavailable"


class EnvelopeWarning(BaseModel):
    code: WarningCode
    message: str = ""
    ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class EnvelopeError(BaseModel):
    code: ErrorCode
    message: str = ""
    ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class EnvelopeMeta(BaseModel):
    limit: int | None = None
    requested_limit: int | None = None
    returned: int | None = None
    total: int | None = None
    truncated: bool = False
    truncation_reason: str | None = None


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: EnvelopeStatus
    data: dict[str, Any] | list[Any] | None = None
    warnings: list[EnvelopeWarning] = Field(default_factory=list)
    errors: list[EnvelopeError] = Field(default_factory=list)
    meta: EnvelopeMeta = Field(default_factory=EnvelopeMeta)

    @classmethod
    def success(
        cls,
        data: dict[str, Any] | list[Any] | None = None,
        warnings: list[EnvelopeWarning] | None = None,
        meta: EnvelopeMeta | None = None,
    ) -> Envelope:
        return cls(
            status=EnvelopeStatus.ok,
            data=data,
            warnings=warnings or [],
            errors=[],
            meta=meta or EnvelopeMeta(),
        )

    @classmethod
    def failure(
        cls,
        errors: list[EnvelopeError] | EnvelopeError,
        data: dict[str, Any] | list[Any] | None = None,
        warnings: list[EnvelopeWarning] | None = None,
        meta: EnvelopeMeta | None = None,
    ) -> Envelope:
        if isinstance(errors, EnvelopeError):
            errors = [errors]
        return cls(
            status=EnvelopeStatus.error,
            data=data,
            warnings=warnings or [],
            errors=errors,
            meta=meta or EnvelopeMeta(),
        )

    def to_json(self) -> str:
        return self.model_dump_json()


def validation_error(
    message: str,
    *,
    ids: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> EnvelopeError:
    return EnvelopeError(
        code=ErrorCode.validation_error,
        message=message,
        ids=ids or [],
        details=details or {},
    )


def not_found_error(
    message: str,
    *,
    ids: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> EnvelopeError:
    return EnvelopeError(
        code=ErrorCode.not_found,
        message=message,
        ids=ids or [],
        details=details or {},
    )


def storage_error(
    message: str,
    *,
    ids: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> EnvelopeError:
    return EnvelopeError(
        code=ErrorCode.storage_error,
        message=message,
        ids=ids or [],
        details=details or {},
    )


def provider_error(
    message: str,
    *,
    ids: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> EnvelopeError:
    return EnvelopeError(
        code=ErrorCode.provider_error,
        message=message,
        ids=ids or [],
        details=details or {},
    )


def conflict_error(
    message: str,
    *,
    ids: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> EnvelopeError:
    return EnvelopeError(
        code=ErrorCode.conflict,
        message=message,
        ids=ids or [],
        details=details or {},
    )


__all__ = [
    "Envelope",
    "EnvelopeError",
    "EnvelopeMeta",
    "EnvelopeStatus",
    "EnvelopeWarning",
    "ErrorCode",
    "WarningCode",
    "validation_error",
    "not_found_error",
    "storage_error",
    "provider_error",
    "conflict_error",
]

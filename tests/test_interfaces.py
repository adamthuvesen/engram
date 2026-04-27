"""Tests for shared structured response envelopes and error codes."""

from __future__ import annotations

import json

import pytest

from engram.interfaces import (
    Envelope,
    EnvelopeError,
    EnvelopeMeta,
    EnvelopeStatus,
    EnvelopeWarning,
    ErrorCode,
    WarningCode,
    conflict_error,
    not_found_error,
    provider_error,
    storage_error,
    validation_error,
)


def test_success_envelope_round_trips():
    env = Envelope.success(
        data={"answer": "hello", "source_fact_ids": ["f1", "f2"]},
        meta=EnvelopeMeta(limit=10, returned=2),
    )
    js = env.to_json()
    parsed = json.loads(js)
    assert parsed["status"] == "ok"
    assert parsed["data"]["answer"] == "hello"
    assert parsed["data"]["source_fact_ids"] == ["f1", "f2"]
    assert parsed["warnings"] == []
    assert parsed["errors"] == []
    assert parsed["meta"]["limit"] == 10
    assert parsed["meta"]["returned"] == 2
    assert parsed["meta"]["truncated"] is False


def test_failure_envelope_with_single_error():
    err = not_found_error("fact missing", ids=["fact_42"])
    env = Envelope.failure(err)
    parsed = json.loads(env.to_json())
    assert parsed["status"] == "error"
    assert len(parsed["errors"]) == 1
    assert parsed["errors"][0]["code"] == "not_found"
    assert parsed["errors"][0]["message"] == "fact missing"
    assert parsed["errors"][0]["ids"] == ["fact_42"]


def test_failure_envelope_with_multiple_errors():
    env = Envelope.failure(
        errors=[
            validation_error("bad input", ids=["q1"]),
            storage_error("write failed"),
        ]
    )
    parsed = json.loads(env.to_json())
    assert parsed["status"] == "error"
    codes = [e["code"] for e in parsed["errors"]]
    assert codes == ["validation_error", "storage_error"]


def test_envelope_warnings_are_serializable():
    warning = EnvelopeWarning(
        code=WarningCode.conflicting_facts,
        message="two active facts disagree",
        ids=["f1", "f2"],
        details={"category": "preference"},
    )
    env = Envelope.success(data={"answer": "..."}, warnings=[warning])
    parsed = json.loads(env.to_json())
    assert parsed["warnings"][0]["code"] == "conflicting_facts"
    assert parsed["warnings"][0]["ids"] == ["f1", "f2"]
    assert parsed["warnings"][0]["details"] == {"category": "preference"}


def test_envelope_meta_truncation_metadata():
    env = Envelope.success(
        data=[],
        meta=EnvelopeMeta(
            limit=10,
            requested_limit=1000,
            returned=10,
            total=500,
            truncated=True,
            truncation_reason="default_limit",
        ),
    )
    parsed = json.loads(env.to_json())
    assert parsed["meta"]["truncated"] is True
    assert parsed["meta"]["truncation_reason"] == "default_limit"
    assert parsed["meta"]["total"] == 500


@pytest.mark.parametrize(
    "code,expected",
    [
        (ErrorCode.validation_error, "validation_error"),
        (ErrorCode.not_found, "not_found"),
        (ErrorCode.provider_error, "provider_error"),
        (ErrorCode.storage_error, "storage_error"),
        (ErrorCode.conflict, "conflict"),
        (ErrorCode.timeout, "timeout"),
        (ErrorCode.internal_error, "internal_error"),
    ],
)
def test_error_code_string_values_are_stable(code, expected):
    """Renaming an existing error code is a breaking change for agents."""
    assert code.value == expected


@pytest.mark.parametrize(
    "code,expected",
    [
        (WarningCode.stale_fact, "stale_fact"),
        (WarningCode.superseded_fact, "superseded_fact"),
        (WarningCode.forgotten_fact, "forgotten_fact"),
        (WarningCode.conflicting_facts, "conflicting_facts"),
        (WarningCode.truncated_output, "truncated_output"),
        (WarningCode.provider_unavailable, "provider_unavailable"),
    ],
)
def test_warning_code_string_values_are_stable(code, expected):
    assert code.value == expected


@pytest.mark.parametrize(
    "status,expected",
    [
        (EnvelopeStatus.ok, "ok"),
        (EnvelopeStatus.error, "error"),
        (EnvelopeStatus.partial, "partial"),
    ],
)
def test_envelope_status_string_values_are_stable(status, expected):
    assert status.value == expected


def test_envelope_round_trip_through_validate():
    err = conflict_error("two truths", ids=["a", "b"], details={"why": "scopes match"})
    env = Envelope.failure(err)
    js = env.to_json()
    rebuilt = Envelope.model_validate_json(js)
    assert rebuilt.status == EnvelopeStatus.error
    assert rebuilt.errors[0].code == ErrorCode.conflict
    assert rebuilt.errors[0].ids == ["a", "b"]
    assert rebuilt.errors[0].details == {"why": "scopes match"}


def test_provider_error_helper():
    err = provider_error("model unreachable", details={"provider": "openai"})
    assert err.code == ErrorCode.provider_error
    assert err.details == {"provider": "openai"}


def test_envelope_forbids_extra_fields():
    """Agents can rely on the schema not silently growing fields they didn't expect."""
    with pytest.raises(Exception):
        Envelope.model_validate(
            {
                "status": "ok",
                "data": None,
                "warnings": [],
                "errors": [],
                "meta": {},
                "extra_field": True,
            }
        )


def test_envelope_error_carries_structured_ids_and_details():
    err = EnvelopeError(
        code=ErrorCode.validation_error,
        message="bad",
        ids=["x", "y"],
        details={"field": "tags"},
    )
    parsed = json.loads(err.model_dump_json())
    assert parsed["ids"] == ["x", "y"]
    assert parsed["details"] == {"field": "tags"}

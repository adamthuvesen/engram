"""Integration tests for the agent-first CLI commands."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from engram import cli
from engram.config import get_settings
from engram.models import Fact, FactCategory


@pytest.fixture
def cli_env(monkeypatch):
    """Isolate CLI to a tempdir-backed store."""
    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setenv("ENGRAM_DATA_DIR", str(tmp))
    get_settings.cache_clear()
    yield tmp
    get_settings.cache_clear()


def _seed(tmp: Path, facts: list[Fact]) -> None:
    from engram.store import FactStore

    store = FactStore(data_dir=tmp)
    store.append_facts(facts)


# ---------------------------------------------------------------------------
# is_cli_invocation
# ---------------------------------------------------------------------------


def test_is_cli_invocation_true_for_known_subcommand():
    assert cli.is_cli_invocation(["trace", "x"]) is True
    assert cli.is_cli_invocation(["doctor"]) is True


def test_is_cli_invocation_false_for_empty_argv():
    assert cli.is_cli_invocation([]) is False


def test_is_cli_invocation_true_for_unknown_argv():
    # Treat any non-empty argv as CLI intent so typos surface as argparse
    # errors instead of silently launching the MCP stdio server.
    assert cli.is_cli_invocation(["mysterious"]) is True


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------


def test_cli_trace_json_no_match(cli_env, capsys):
    exit_code = cli.run(["trace", "obscure", "--json"])
    assert exit_code == cli.EXIT_OK
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["status"] == "ok"
    assert (
        parsed["data"]["trace"] is not None
    )  # tier-0 still emits a (possibly empty) trace
    assert parsed["data"]["quality"] in ("none", "low")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def test_cli_doctor_json_healthy(cli_env, capsys):
    _seed(
        cli_env,
        [Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="x")],
    )
    exit_code = cli.run(["doctor", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_OK
    assert parsed["data"]["report"]["status"] in ("ok", "warning")


def test_cli_doctor_json_with_corrupt_jsonl(cli_env, capsys):
    facts_path = cli_env / "facts.jsonl"
    facts_path.write_text("not json\n")
    exit_code = cli.run(["doctor", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    # Corrupt JSONL is an error severity; doctor should exit non-zero.
    assert exit_code == cli.EXIT_DOCTOR_ERROR
    codes = [i["code"] for i in parsed["data"]["report"]["issues"]]
    assert "facts_jsonl_corrupt" in codes


# ---------------------------------------------------------------------------
# correct
# ---------------------------------------------------------------------------


def test_cli_correct_json_success(cli_env, capsys):
    _seed(
        cli_env,
        [
            Fact(
                id="oldaaaaaaaaa",
                category=FactCategory.preference,
                content="vim",
            )
        ],
    )
    exit_code = cli.run(["correct", "oldaaaaaaaaa", "neovim", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_OK
    assert parsed["data"]["superseded_fact_id"] == "oldaaaaaaaaa"


def test_cli_correct_json_not_found(cli_env, capsys):
    exit_code = cli.run(["correct", "missingmissi", "x", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_NOT_FOUND
    assert parsed["errors"][0]["code"] == "not_found"


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def test_cli_merge_json_validation_one_source(cli_env, capsys):
    exit_code = cli.run(["merge", "onlyoneonlyo", "--content", "merged", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_VALIDATION
    assert parsed["errors"][0]["code"] == "validation_error"


def test_cli_merge_json_success(cli_env, capsys):
    _seed(
        cli_env,
        [
            Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="a"),
            Fact(id="bbbbbbbbbbbb", category=FactCategory.preference, content="b"),
        ],
    )
    exit_code = cli.run(
        [
            "merge",
            "aaaaaaaaaaaa",
            "bbbbbbbbbbbb",
            "--content",
            "merged",
            "--json",
        ]
    )
    parsed = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_OK
    assert set(parsed["data"]["superseded_fact_ids"]) == {
        "aaaaaaaaaaaa",
        "bbbbbbbbbbbb",
    }


# ---------------------------------------------------------------------------
# stale
# ---------------------------------------------------------------------------


def test_cli_stale_marks_fact_stale(cli_env, capsys):
    _seed(
        cli_env,
        [Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="x")],
    )
    exit_code = cli.run(["stale", "aaaaaaaaaaaa", "--reason", "old", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_OK
    assert parsed["data"]["stale"] is True


def test_cli_stale_not_found(cli_env, capsys):
    exit_code = cli.run(["stale", "missingmissi", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_NOT_FOUND
    assert parsed["errors"][0]["code"] == "not_found"


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def test_cli_inspect_json_bounded(cli_env, capsys):
    _seed(
        cli_env,
        [
            Fact(
                id=f"f{i:02d}{'a' * 9}",
                category=FactCategory.preference,
                content=f"item {i}",
            )
            for i in range(5)
        ],
    )
    exit_code = cli.run(["inspect", "--limit", "3", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_OK
    assert len(parsed["data"]) == 3
    assert parsed["meta"]["truncated"] is True
    assert parsed["meta"]["truncation_reason"] == "default_limit"


def test_cli_inspect_invalid_category(cli_env, capsys):
    exit_code = cli.run(["inspect", "--category", "nonsense", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_VALIDATION
    assert parsed["errors"][0]["code"] == "validation_error"


# ---------------------------------------------------------------------------
# Non-JSON mode still works (smoke)
# ---------------------------------------------------------------------------


def test_cli_inspect_text_mode_smoke(cli_env, capsys):
    _seed(
        cli_env,
        [Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="x")],
    )
    exit_code = cli.run(["inspect"])
    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_OK
    assert "aaaaaaaaaaaa" in out

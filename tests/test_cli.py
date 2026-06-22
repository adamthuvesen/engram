"""Integration tests for the agent-first CLI commands."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engram import cli
from engram.core.config import get_settings
from engram.core.interfaces import Envelope
from engram.core.models import Fact, FactCategory, RecallRecord
from engram.operations import OperationResult
from engram.storage.store import FactStore


@pytest.fixture
def cli_env(monkeypatch):
    """Isolate CLI to a tempdir-backed store."""
    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setenv("ENGRAM_DATA_DIR", str(tmp))
    get_settings.cache_clear()
    yield tmp
    get_settings.cache_clear()


def _seed(tmp: Path, facts: list[Fact]) -> None:
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


def test_cli_normalizes_only_top_level_help():
    assert cli._normalize_argv(["help"]) == ["--help"]
    assert cli._normalize_argv(["--json", "help"]) == ["--help"]
    assert cli._normalize_argv(["remember", "help"]) == ["remember", "help"]
    assert cli._normalize_argv(["recall", "--help"]) == ["recall", "--help"]


def test_cli_content_can_be_literal_help(monkeypatch, capsys):
    captured: dict[str, str] = {}

    async def fake_remember(args):
        captured["content"] = args.content
        return OperationResult(envelope=Envelope.success(), text="stored")

    monkeypatch.setitem(cli.HANDLERS, "remember", fake_remember)

    exit_code = cli.run(["remember", "help"])

    assert exit_code == cli.EXIT_OK
    assert captured["content"] == "help"
    assert capsys.readouterr().out == "stored\n"


def test_cli_run_returns_code_for_subcommand_help(capsys):
    exit_code = cli.run(["recall", "--help"])

    assert exit_code == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "usage: engram recall" in out
    assert "--with-provenance" in out


def test_cli_run_returns_code_for_parse_errors(capsys):
    exit_code = cli.run(["definitely-not-a-command"])

    assert exit_code == 2
    assert "invalid choice" in capsys.readouterr().err


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


def test_cli_recall_invalid_max_sources(cli_env, capsys):
    exit_code = cli.run(["recall", "anything", "--max-sources", "0", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_VALIDATION
    assert parsed["errors"][0]["code"] == "validation_error"
    assert parsed["errors"][0]["details"] == {"parameter": "max_sources"}


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
    # Seed a healthy event-log file via the store, then append a corrupt line.
    # (Writing arbitrary content to facts.jsonl pre-store-init would trigger
    # the one-shot legacy-format migration which silently drops bad lines —
    # the post-migration scenario is what doctor needs to surface.)
    store = FactStore(data_dir=cli_env)
    store.append_facts(
        [Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="x")]
    )
    with store.facts_path.open("a") as fh:
        fh.write("not json\n")

    exit_code = cli.run(["doctor", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    # Corrupt JSONL is an error severity; doctor should exit non-zero.
    assert exit_code == cli.EXIT_DOCTOR_ERROR
    codes = [i["code"] for i in parsed["data"]["report"]["issues"]]
    assert "facts_jsonl_corrupt" in codes


def test_cli_doctor_repairs_orphaned_supersessions(cli_env, capsys):
    store = FactStore(data_dir=cli_env)
    store.append_facts(
        [
            Fact(
                id="orphanaaaaaa",
                category=FactCategory.preference,
                content="orphan",
                supersedes="missingmissi",
            )
        ]
    )

    exit_code = cli.run(
        ["doctor", "--repair", "--repair-orphaned-supersessions", "--json"]
    )
    parsed = json.loads(capsys.readouterr().out)
    repaired = FactStore(data_dir=cli_env).load_facts()[0]

    assert exit_code == cli.EXIT_OK
    assert parsed["data"]["repair"]["orphaned_supersessions"]["cleared"] == 1
    assert repaired.supersedes is None


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


def test_cli_recall_stats_aggregate_and_include_records(cli_env, capsys):
    store = FactStore(data_dir=cli_env)
    now = datetime.now(timezone.utc)
    store.log_recall(
        RecallRecord(
            query="old",
            tier=0,
            prefilter_count=1,
            latency_ms=10,
            timestamp=now - timedelta(days=2),
        )
    )
    store.log_recall(
        RecallRecord(
            query="new",
            tier=1,
            prefilter_count=2,
            latency_ms=20,
            timestamp=now,
        )
    )

    exit_code = cli.run(["recall-stats", "--json", "--limit", "1"])
    aggregate = json.loads(capsys.readouterr().out)
    records_exit = cli.run(
        [
            "recall-stats",
            "--json",
            "--include-records",
            "--since",
            (now - timedelta(days=1)).isoformat(),
        ]
    )
    with_records = json.loads(capsys.readouterr().out)

    assert exit_code == records_exit == cli.EXIT_OK
    assert aggregate["data"]["total_queries"] == 1
    assert "records" not in aggregate["data"]
    assert with_records["data"]["records"][0]["query"] == "new"


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
    assert parsed["meta"]["total"] == 5


def test_cli_inspect_exact_limit_not_truncated(cli_env, capsys):
    _seed(
        cli_env,
        [
            Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="a"),
            Fact(id="bbbbbbbbbbbb", category=FactCategory.preference, content="b"),
        ],
    )
    exit_code = cli.run(["inspect", "--limit", "2", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_OK
    assert len(parsed["data"]) == 2
    assert parsed["meta"]["total"] == 2
    assert parsed["meta"]["truncated"] is False


def test_cli_inspect_invalid_category(cli_env, capsys):
    exit_code = cli.run(["inspect", "--category", "nonsense", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_VALIDATION
    assert parsed["errors"][0]["code"] == "validation_error"


def test_cli_inspect_invalid_limit(cli_env, capsys):
    exit_code = cli.run(["inspect", "--limit", "0", "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_VALIDATION
    assert parsed["errors"][0]["code"] == "validation_error"
    assert parsed["errors"][0]["message"] == "limit must be greater than zero"


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

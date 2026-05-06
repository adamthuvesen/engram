"""Parity tests for MCP tools and CLI commands."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from engram import cli, server
from engram.config import get_settings
from engram.models import Fact, FactCategory, RecallRecord
from engram.store import AsyncFactStore, FactStore


@pytest.fixture
def cli_env(monkeypatch):
    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setenv("ENGRAM_DATA_DIR", str(tmp))
    get_settings.cache_clear()
    yield tmp
    get_settings.cache_clear()


def _seed(tmp: Path, facts: list[Fact]) -> FactStore:
    store = FactStore(data_dir=tmp)
    store.append_facts(facts)
    return store


async def _call(tool, **kwargs):
    fn = getattr(tool, "fn", tool)
    if asyncio.iscoroutinefunction(fn):
        return await fn(**kwargs)
    return fn(**kwargs)


def test_canonical_cli_commands_match_mcp_tools():
    tools = asyncio.run(server.mcp.list_tools())
    mcp_commands = {tool.name.replace("_", "-") for tool in tools}

    assert mcp_commands == cli.CANONICAL_COMMANDS


def test_help_invocation_is_cli_not_server(monkeypatch, capsys):
    def fail_server_start():
        raise AssertionError("help should not start the MCP server")

    monkeypatch.setattr("engram.server.main", fail_server_start)

    exit_code = cli.main_dispatch(["--help"])

    assert exit_code == cli.EXIT_OK
    assert "Engram CLI" in capsys.readouterr().out


def test_global_json_flag_dispatches_to_cli(cli_env, capsys):
    _seed(
        cli_env,
        [Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="uses vim")],
    )

    exit_code = cli.main_dispatch(["--json", "inspect"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == cli.EXIT_OK
    assert payload["status"] == "ok"
    assert payload["data"][0]["id"] == "aaaaaaaaaaaa"


def test_cli_and_mcp_inspect_json_have_matching_shape(cli_env, capsys):
    _seed(
        cli_env,
        [Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="uses vim")],
    )
    server._store = AsyncFactStore(FactStore(data_dir=cli_env))

    cli_exit = cli.run(["inspect", "--json"])
    cli_payload = json.loads(capsys.readouterr().out)
    mcp_payload = json.loads(
        asyncio.run(_call(server.inspect, format="json", limit=50))
    )

    assert cli_exit == cli.EXIT_OK
    assert cli_payload["status"] == mcp_payload["status"] == "ok"
    assert cli_payload["meta"].keys() == mcp_payload["meta"].keys()
    assert cli_payload["data"][0].keys() == mcp_payload["data"][0].keys()


def test_cli_and_mcp_correct_memory_json_have_matching_shape(cli_env, capsys):
    _seed(
        cli_env,
        [Fact(id="oldaaaaaaaaa", category=FactCategory.preference, content="vim")],
    )
    mcp_tmp = Path(tempfile.mkdtemp())
    _seed(
        mcp_tmp,
        [Fact(id="oldaaaaaaaaa", category=FactCategory.preference, content="vim")],
    )
    server._store = AsyncFactStore(FactStore(data_dir=mcp_tmp))

    cli_exit = cli.run(["correct-memory", "oldaaaaaaaaa", "neovim", "--json"])
    cli_payload = json.loads(capsys.readouterr().out)
    mcp_payload = json.loads(
        asyncio.run(
            _call(
                server.correct_memory,
                fact_id="oldaaaaaaaaa",
                new_content="neovim",
            )
        )
    )

    assert cli_exit == cli.EXIT_OK
    assert cli_payload["status"] == mcp_payload["status"] == "ok"
    assert cli_payload["data"].keys() == mcp_payload["data"].keys()
    assert cli_payload["data"]["superseded_fact_id"] == "oldaaaaaaaaa"
    assert mcp_payload["data"]["superseded_fact_id"] == "oldaaaaaaaaa"


def test_cli_and_mcp_mark_stale_json_have_matching_payload(cli_env, capsys):
    _seed(
        cli_env,
        [Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="x")],
    )
    mcp_tmp = Path(tempfile.mkdtemp())
    _seed(
        mcp_tmp,
        [Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="x")],
    )
    server._store = AsyncFactStore(FactStore(data_dir=mcp_tmp))

    cli_exit = cli.run(["mark-stale", "aaaaaaaaaaaa", "--reason", "old", "--json"])
    cli_payload = json.loads(capsys.readouterr().out)
    mcp_payload = json.loads(
        asyncio.run(
            _call(server.mark_stale, fact_id="aaaaaaaaaaaa", reason="old")
        )
    )

    assert cli_exit == cli.EXIT_OK
    assert cli_payload == mcp_payload


def test_cli_and_mcp_memory_stats_json_have_matching_shape(cli_env, capsys):
    store = _seed(
        cli_env,
        [
            Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="x"),
            Fact(id="bbbbbbbbbbbb", category=FactCategory.workflow, content="y"),
        ],
    )
    store.log_recall(
        RecallRecord(
            query="x",
            tier=0,
            prefilter_count=1,
            latency_ms=1,
            quality="high",
            llm_calls=0,
        )
    )
    server._store = AsyncFactStore(FactStore(data_dir=cli_env))

    cli_stats_exit = cli.run(["memory-stats", "--json"])
    cli_stats = json.loads(capsys.readouterr().out)
    mcp_stats = json.loads(asyncio.run(_call(server.memory_stats, format="json")))

    cli_recall_exit = cli.run(["recall-stats", "--json"])
    cli_recall = json.loads(capsys.readouterr().out)
    mcp_recall = json.loads(asyncio.run(_call(server.recall_stats, format="json")))

    assert cli_stats_exit == cli_recall_exit == cli.EXIT_OK
    assert cli_stats["data"].keys() == mcp_stats["data"].keys()
    assert cli_recall["data"].keys() == mcp_recall["data"].keys()

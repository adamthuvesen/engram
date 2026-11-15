"""Engram verify drives. Each drive starts from a fresh store and asserts what a user sees."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engram.core.config import Settings
from engram.core.models import Fact, FactCategory
from engram.storage.async_store import AsyncFactStore, run_sync
from engram.storage.store import FactStore

SEEDS = [
    (
        "verifyfact01",
        "alpha",
        FactCategory.convention,
        "Deploy previews use the zinnia staging bucket.",
    ),
    (
        "verifyfact02",
        "beta",
        FactCategory.decision,
        "Tabs over spaces in the quokka repo.",
    ),
    ("smoke0000001", None, FactCategory.convention, "Fictional smoke memory."),
]


class Failure(Exception):
    """A drive observed something other than the expected user-visible result."""


@dataclass
class Context:
    name: str
    scratch: Path
    evidence: Path
    env: dict[str, str]
    helper: Path
    root: Path
    processes: list[subprocess.Popen[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.data = self.scratch / "data"
        self.data.mkdir(parents=True)
        self.evidence.mkdir(parents=True)
        self.env = {**self.env, "ENGRAM_DATA_DIR": str(self.data)}
        self.step = 0

    def log(self, label: str, text: str) -> None:
        self.step += 1
        (self.evidence / f"{self.step:02d}-{label}.txt").write_text(text)

    def run(
        self, args: list[str], *, cwd: Path | None = None, timeout: float = 30
    ) -> subprocess.CompletedProcess[str]:
        process = subprocess.Popen(
            args,
            cwd=cwd or self.scratch,
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.processes.append(process)
        try:
            out, err = process.communicate(timeout=timeout)
        finally:
            self.stop_processes()
        self.log(
            Path(args[1] if len(args) > 1 else args[0]).name,
            f"$ {' '.join(args)}\nexit {process.returncode}\n--- stdout\n{out}\n--- stderr\n{err}",
        )
        return subprocess.CompletedProcess(args, process.returncode, out, err)

    def engram(self, *args: str) -> dict[str, Any]:
        result = self.run([str(Path(sys.executable).with_name("engram")), *args])
        try:
            return {"exit": result.returncode, **json.loads(result.stdout)}
        except json.JSONDecodeError as exc:
            raise Failure(
                f"engram {' '.join(args)} printed no JSON envelope (exit {result.returncode}): {result.stderr[-300:]}"
            ) from exc

    def stop_processes(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        self.processes.clear()


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


async def seed(data: Path) -> None:
    store = AsyncFactStore(await run_sync(FactStore, data))
    await store.append_facts([Fact(id=i, project=p, category=c, content=t) for i, p, c, t in SEEDS])


def ids(envelope: dict[str, Any]) -> set[str]:
    data = envelope.get("data") or []
    rows = data.get("facts", []) if isinstance(data, dict) else data
    return {row["id"] for row in rows}


async def drive_cli(ctx: Context, expect_id: str = "verifyfact01") -> str:
    await seed(ctx.data)
    recalled = ctx.engram("recall", "zinnia staging bucket", "--json")
    expect(
        recalled["exit"] == 0 and recalled["status"] == "ok",
        f"recall: expected status ok, got {recalled['status']} exit {recalled['exit']}",
    )
    top = [f["id"] for f in recalled["data"]["facts"]]
    expect(top[:1] == [expect_id], f"recall: expected top card {expect_id}, got {top}")
    calls = recalled["data"]["usage"]["llm_calls"]
    expect(calls == 0, f"recall cards: expected 0 LLM calls, got {calls}")
    listed = ids(ctx.engram("inspect", "--json"))
    expect(
        listed == {s[0] for s in SEEDS},
        f"inspect: expected the 3 seeded ids, got {sorted(listed)}",
    )
    stale = ctx.engram("mark-stale", "verifyfact01", "--json")
    expect(
        stale["status"] == "ok" and stale["data"]["stale"] is True,
        f"mark-stale: got {stale['status']}",
    )
    alpha = ids(ctx.engram("inspect", "--project", "alpha", "--json"))
    expect(
        alpha == {"smoke0000001"},
        f"inspect after mark-stale: expected only the global card, got {sorted(alpha)}",
    )
    missing = ctx.engram("forget", "missing-verify-fact", "--json")
    expect(
        missing["exit"] == 2 and missing["status"] == "error",
        f"forget missing: expected exit 2 error, got exit {missing['exit']} {missing['status']}",
    )
    doctor = ctx.engram("doctor", "--json")
    expect(
        doctor["data"]["report"]["status"] == "ok",
        f"doctor: expected ok, got {doctor['data']['report']}",
    )
    return "recall cards (0 LLM calls), inspect, mark-stale, missing-fact exit 2, doctor ok"


# The control passes only when its drive fails on exactly this planted expectation.
CONTROL_FAILURE = "recall: expected top card verifyfact99, got ['verifyfact01']"


async def control(ctx: Context) -> str:
    """Re-drive cli against a card id the seed never wrote; verify control passes only on FAIL."""
    return await drive_cli(ctx, expect_id="verifyfact99")


EXTRACTIONS = [
    {
        "facts": [
            {
                "memory_key": "verify heron checklist",
                "content": "Releases follow the heron checklist.",
                "category": "workflow",
                "retrieval_hints": ["heron checklist"],
                "covered_claims": ["heron checklist"],
                "why_store": "release step",
            }
        ]
    },
    {
        "facts": [
            {
                "memory_key": "verify ibis lint rule",
                "content": "The ibis lint rule blocks wildcard imports.",
                "category": "convention",
                "retrieval_hints": ["ibis lint"],
                "covered_claims": ["ibis lint"],
                "why_store": "lint rule",
            }
        ]
    },
]


def serve_fixture(data: Path, responses: Path) -> None:
    from engram.runtime import Runtime
    from engram.server import create_mcp
    from evals.provider import FixtureCompletionClient

    fixture = FixtureCompletionClient(
        [(json.dumps(r), None, None) for r in json.loads(responses.read_text())]
    )
    settings = Settings(data_dir=data, maintenance_enabled=False, sync_enabled=False)
    create_mcp(runtime=Runtime(settings=settings, provider=fixture)).run(show_banner=False)


async def drive_memory(ctx: Context) -> str:
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    responses = ctx.scratch / "extractions.json"
    responses.write_text(json.dumps(EXTRACTIONS))
    transport = StdioTransport(
        command=sys.executable,
        args=[str(ctx.helper), "_serve-fixture", str(ctx.data), str(responses)],
        env=ctx.env,
        cwd=str(ctx.scratch),
    )

    async def call(client: Client[Any], tool: str, **args: Any) -> dict[str, Any]:
        result = await client.call_tool(tool, {**args, "format": "json"}, raise_on_error=False)
        envelope = result.structured_content or {}
        ctx.log(f"mcp-{tool}", json.dumps({"args": args, "envelope": envelope}, indent=2))
        return envelope

    async with Client(transport, timeout=15) as client:
        made = await call(
            client,
            "remember",
            content="We follow the heron checklist for releases.",
            project="gamma",
        )
        new = ids(made)
        expect(
            made.get("status") == "ok" and len(new) == 1,
            f"remember: expected one new card, got {made.get('status')} {sorted(new)}",
        )
        fact_id = new.pop()
        found = ids(await call(client, "recall", query="heron checklist", project="gamma"))
        expect(
            fact_id in found,
            f"recall after remember: expected {fact_id}, got {sorted(found)}",
        )
        gone = await call(client, "forget", fact_id=fact_id)
        expect(gone.get("status") == "ok", f"forget: expected ok, got {gone.get('status')}")
        after = ids(await call(client, "recall", query="heron checklist", project="gamma"))
        expect(fact_id not in after, f"recall after forget: {fact_id} still returned")
        queued = await call(
            client,
            "suggest_memories",
            content="The ibis lint rule blocks wildcard imports.",
            project="gamma",
        )
        candidates = [c["id"] for c in (queued.get("data") or {}).get("candidates", [])]
        expect(
            len(candidates) == 1,
            f"suggest_memories: expected one candidate, got {queued.get('status')} {candidates}",
        )
        approved = await call(client, "approve_candidates", candidate_ids=candidates)
        expect(
            approved.get("status") == "ok",
            f"approve_candidates: expected ok, got {approved.get('status')}",
        )
        listed = await call(client, "inspect", project="gamma")
        contents = [row["content"] for row in listed.get("data") or []]
        expect(
            contents == ["The ibis lint rule blocks wildcard imports."],
            f"inspect after approve: got {contents}",
        )
    return "MCP stdio remember→recall→forget and suggest→approve→inspect with a scripted provider"


async def drive_serve(ctx: Context) -> str:
    from tools import smoke

    await seed(ctx.data)
    try:
        stdio = await smoke.run_stdio(ctx.data, cwd=ctx.scratch, entrypoints=True)
        http = await smoke.run_http(ctx.data, cwd=ctx.scratch, entrypoints=True)
    except RuntimeError as exc:
        raise Failure(str(exc)) from exc
    finally:
        log = ctx.data / "http-smoke.log"
        if log.exists():
            shutil.copy(log, ctx.evidence / "http-server.log")
    ctx.log("serve", f"{stdio.model_dump_json()}\n{http.model_dump_json()}")
    return f"engram serve stdio and http: {', '.join(http.checks)}"


async def drive_package(ctx: Context) -> str:
    result = ctx.run(
        [sys.executable, "-m", "tools.verify", "package"],
        cwd=ctx.root,
        timeout=120,
    )
    expect(
        result.returncode == 0 and "passed: installed wheel transports" in result.stdout,
        f"package: expected installed wheel smoke to pass, got exit {result.returncode}: {result.stdout[-400:]}",
    )
    return (
        "wheel builds, installs outside the checkout, and its console serves CLI, stdio, and HTTP"
    )


@dataclass(frozen=True)
class Drive:
    proves: str
    paths: list[str]
    run: Callable[[Context], Awaitable[str]]
    timeout: float = 30.0


CORE = ["src/engram/core/*", "src/engram/storage/*", "src/engram/operations.py"]
DRIVES = {
    "cli": Drive(
        "The engram console recalls, inspects, marks stale, rejects a missing fact, and passes doctor offline",
        [*CORE, "src/engram/cli.py", "src/engram/recall/*", "src/engram/maintenance/*"],
        drive_cli,
    ),
    "memory": Drive(
        "The MCP server stores, recalls, forgets, and approves memories through the provider boundary",
        [
            *CORE,
            "src/engram/server.py",
            "src/engram/runtime.py",
            "src/engram/extraction/*",
            "src/engram/llm/*",
            "evals/provider.py",
        ],
        drive_memory,
    ),
    "serve": Drive(
        "engram serve answers MCP over stdio and HTTP and keeps facts across a restart",
        [
            "src/engram/server.py",
            "src/engram/runtime.py",
            "src/engram/cli.py",
            "tools/smoke.py",
        ],
        drive_serve,
    ),
    "package": Drive(
        "The built wheel installs outside the checkout and its entrypoints work",
        [
            "tools/*",
        ],
        drive_package,
        150.0,
    ),
}
SMOKE = "cli"

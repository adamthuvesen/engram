"""Verify the console and MCP transports with a disabled provider."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from fastmcp import Client
from fastmcp.client.transports import ClientTransport, StdioTransport
from pydantic import BaseModel

from engram.core.config import Settings
from engram.core.models import Fact, FactCategory
from engram.runtime import Runtime
from engram.storage.async_store import AsyncFactStore, run_sync
from engram.storage.store import FactStore
from evals.provider import FixtureCompletionClient


class SmokeResult(BaseModel):
    transport: str
    checks: list[str]
    provider: Literal["disabled", "blocked"] = "disabled"


def child_environment(data_dir: Path, *, entrypoints: bool = False) -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("ENGRAM_")
        and name not in ("PYTHONPATH", "OPENAI_API_KEY", "OPENAI_BASE_URL")
    }
    env.update(
        ENGRAM_DATA_DIR=str(data_dir),
        ENGRAM_MAINTENANCE_ENABLED="false",
        ENGRAM_SYNC_ENABLED="false",
        FASTMCP_CHECK_FOR_UPDATES="off",
    )
    if entrypoints:
        env.update(
            OPENAI_API_KEY="engram-smoke-synthetic-key",
            OPENAI_BASE_URL="http://127.0.0.1:1/v1",
        )
    return env


async def _exercise[T: ClientTransport](client: Client[T]) -> list[str]:
    tools = await client.list_tools()
    names = {tool.name for tool in tools}
    if not {"inspect", "list_candidates", "forget"} <= names:
        raise RuntimeError("MCP discovery omitted required tools.")
    inspected = await client.call_tool("inspect", {"format": "json", "limit": 10})
    envelope = inspected.structured_content
    if (
        not isinstance(envelope, dict)
        or envelope.get("status") != "ok"
        or "smoke0000001" not in str(envelope)
    ):
        raise RuntimeError("MCP inspect did not return the persisted smoke fact.")
    candidates = await client.call_tool("list_candidates", {"format": "json"})
    if (
        not isinstance(candidates.structured_content, dict)
        or candidates.structured_content.get("status") != "ok"
    ):
        raise RuntimeError("MCP list_candidates failed.")
    invalid = await client.call_tool(
        "forget",
        {"fact_id": "missing-smoke-fact", "format": "json"},
        raise_on_error=False,
    )
    if (
        not invalid.is_error
        or not isinstance(invalid.structured_content, dict)
        or invalid.structured_content.get("status") != "error"
    ):
        raise RuntimeError("MCP missing-fact error did not use the error envelope.")
    return ["tools/list", "persisted inspect", "list candidates", "missing-fact error"]


def _server_args(data_dir: Path, transport: str, port: int | None = None) -> list[str]:
    args = [
        "-m",
        "tools.smoke",
        "serve",
        "--data-dir",
        str(data_dir),
        "--transport",
        transport,
    ]
    if port is not None:
        args.extend(["--port", str(port)])
    return args


def _server_command(
    python: str, data_dir: Path, transport: str, port: int | None, entrypoints: bool
) -> list[str]:
    if not entrypoints:
        return [python, *_server_args(data_dir, transport, port)]
    console = Path(python).with_name("engram")
    if not console.is_file():
        raise RuntimeError(f"Installed console entrypoint is missing: {console}")
    command = [str(console), "serve", "--transport", transport]
    if port is not None:
        command.extend(["--host", "127.0.0.1", "--port", str(port)])
    return command


async def run_cli(data_dir: Path) -> SmokeResult:
    console = Path(sys.executable).with_name("engram")
    checks: list[str] = []
    for args, expected_status, expected_code in (
        (["inspect", "--json", "--limit", "10"], "ok", 0),
        (["forget", "missing-smoke-fact", "--json"], "error", 2),
    ):
        process = await asyncio.create_subprocess_exec(
            str(console),
            *args,
            env=child_environment(data_dir, entrypoints=True),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        output, error = await process.communicate()
        if process.returncode != expected_code:
            raise RuntimeError(
                f"Console entrypoint failed: {error.decode()[-2000:]} {output.decode()[-2000:]}"
            )
        envelope = json.loads(output)
        if envelope.get("status") != expected_status:
            raise RuntimeError(f"Console returned an invalid envelope: {output.decode()[-2000:]}")
        if expected_status == "ok" and "smoke0000001" not in str(envelope):
            raise RuntimeError("Console inspect did not return the persisted smoke fact.")
        checks.append(" ".join(args))
    return SmokeResult(transport="cli", checks=checks, provider="blocked")


async def run_stdio(
    data_dir: Path,
    *,
    python: str = sys.executable,
    cwd: Path | None = None,
    entrypoints: bool = False,
) -> SmokeResult:
    transport = StdioTransport(
        command=_server_command(python, data_dir, "stdio", None, entrypoints)[0],
        args=_server_command(python, data_dir, "stdio", None, entrypoints)[1:],
        env=child_environment(data_dir, entrypoints=entrypoints),
        cwd=str(cwd) if cwd else None,
    )
    async with Client(transport, timeout=10) as client:
        checks = await _exercise(client)
    return SmokeResult(
        transport="stdio",
        checks=checks,
        provider="blocked" if entrypoints else "disabled",
    )


async def run_http(
    data_dir: Path,
    *,
    python: str = sys.executable,
    cwd: Path | None = None,
    entrypoints: bool = False,
) -> SmokeResult:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    checks: list[str] = []
    for _restart in range(2):
        with (data_dir / "http-smoke.log").open("a") as log:
            process = subprocess.Popen(
                _server_command(python, data_dir, "http", port, entrypoints),
                cwd=cwd,
                env=child_environment(data_dir, entrypoints=entrypoints),
                stdout=log,
                stderr=log,
            )
            try:
                for _attempt in range(600):
                    if process.poll() is not None:
                        raise RuntimeError(
                            "HTTP smoke server exited: "
                            + (data_dir / "http-smoke.log").read_text()[-2000:]
                        )
                    try:
                        _reader, writer = await asyncio.open_connection("127.0.0.1", port)
                        writer.close()
                        await writer.wait_closed()
                        break
                    except OSError:
                        await asyncio.sleep(0.05)
                else:
                    raise RuntimeError("HTTP smoke server did not become ready within 30 seconds.")
                async with Client(f"http://127.0.0.1:{port}/mcp", timeout=10) as client:
                    checks = await _exercise(client)
            finally:
                process.terminate()
                try:
                    await asyncio.to_thread(process.wait, timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    await asyncio.to_thread(process.wait, timeout=5)
    return SmokeResult(
        transport="http",
        checks=[*checks, "restart persistence"],
        provider="blocked" if entrypoints else "disabled",
    )


async def run_smoke(*, entrypoints: bool = False) -> list[SmokeResult]:
    with TemporaryDirectory(prefix="engram-smoke-") as temporary:
        data_dir = Path(temporary)
        store = AsyncFactStore(await run_sync(FactStore, data_dir))
        await store.append_facts(
            [
                Fact(
                    id="smoke0000001",
                    category=FactCategory.convention,
                    content="Fictional smoke memory.",
                )
            ]
        )
        results = []
        if entrypoints:
            results.append(await run_cli(data_dir))
        results.extend(
            [
                await run_stdio(data_dir, entrypoints=entrypoints),
                await run_http(data_dir, entrypoints=entrypoints),
            ]
        )
        return results


def serve(data_dir: Path, transport: str, port: int) -> None:
    from engram.server import create_mcp

    runtime = Runtime(
        settings=Settings(data_dir=data_dir, maintenance_enabled=False, sync_enabled=False),
        provider=FixtureCompletionClient(),
    )
    app = create_mcp(runtime=runtime)
    if transport == "stdio":
        app.run(show_banner=False)
    else:
        app.run(
            transport="http",
            host="127.0.0.1",
            port=port,
            path="/mcp",
            stateless_http=True,
            host_origin_protection="auto",
            show_banner=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "serve"), nargs="?", default="run")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--port", type=int, default=7422)
    parser.add_argument(
        "--entrypoints",
        action="store_true",
        help="Exercise installed console commands with provider network calls blocked.",
    )
    args = parser.parse_args()
    if args.command == "serve":
        if args.data_dir is None:
            parser.error("serve requires --data-dir")
        serve(args.data_dir, args.transport, args.port)
        return 0
    try:
        for result in asyncio.run(run_smoke(entrypoints=args.entrypoints)):
            print(result.model_dump_json())
    except Exception as exc:
        print(f"Smoke verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

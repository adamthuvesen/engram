"""Generate configuration and tool references from registered contracts."""

import argparse
import asyncio
import json
from pathlib import Path

from engram.core.config import Settings


def _default_text(value: object) -> str:
    if isinstance(value, Path):
        return str(value).replace(str(Path.home()), "~")
    if isinstance(value, list):
        return json.dumps([_default_text(item) for item in value])
    if isinstance(value, str):
        return value or '""'
    return json.dumps(value)


def configuration_markdown() -> str:
    lines = [
        "# Configuration",
        "",
        "Settings use `ENGRAM_*` environment variables. JSON arrays configure list values.",
        "",
        "Generated from `Settings` descriptions. Run `just reference` after changing a field.",
        "",
        "| Environment variable | Default | Description |",
        "| --- | --- | --- |",
    ]
    for name, field in Settings.model_fields.items():
        if not field.description:
            raise ValueError(
                f"Settings.{name} needs a description before reference generation."
            )
        default = _default_text(field.default).replace("|", "\\|")
        description = field.description.replace("|", "\\|")
        lines.append(f"| `ENGRAM_{name.upper()}` | `{default}` | {description} |")
    lines.extend(
        [
            "",
            "Positive counts and intervals reject zero, negative values, infinity, and NaN where applicable.",
            "The field types and validation constraints live in `src/engram/core/config.py`.",
            "",
            "`OPENAI_API_KEY` supplies provider credentials. Runtime reads can use an injected disabled provider.",
            "`ENGRAM_EVAL_PROVIDER=1` explicitly enables live evals. The default verification profile never enables it.",
            "",
        ]
    )
    return "\n".join(lines)


async def tools_markdown() -> str:
    from engram.cli import ALIASES, HANDLERS
    from engram.server import create_mcp

    tools = await create_mcp().list_tools()
    lines = [
        "# MCP tools and CLI commands",
        "",
        "Tool names form a stable public API. The CLI uses the corresponding hyphenated names.",
        "",
        "Generated from MCP registrations and CLI handlers. Run `just reference` after changing either surface.",
        "",
        "| MCP tool | CLI command | Description |",
        "| --- | --- | --- |",
    ]
    for tool in sorted(tools, key=lambda item: item.name):
        command = tool.name.replace("_", "-")
        if command not in HANDLERS:
            raise ValueError(f"MCP tool {tool.name} has no CLI handler.")
        description = " ".join(
            (tool.description or "").split("\n\n", 1)[0].split()
        ).replace("|", "\\|")
        lines.append(f"| `{tool.name}` | `{command}` | {description} |")
    lines.extend(
        [
            "",
            "CLI aliases: "
            + ", ".join(
                f"`{alias}` → `{command}`" for alias, command in ALIASES.items()
            )
            + ".",
            "",
            "`engram serve` owns the MCP process. It has no corresponding tool.",
            "",
            "Every tool validates input before mutation. MCP returns the same response envelope as the CLI JSON output.",
            "The envelope contains `status`, `data`, `warnings`, `errors`, and `meta`.",
            "",
            "Recall uses lexical cards by default. Weak-hit selection and answer mode can call the provider.",
            "Answer mode caps its prompt evidence before generation. The renderer creates citations from validated fact references.",
            "`max_sources` caps cards and answer evidence. The MCP `limit` alias remains supported.",
            "",
            "`RecallResult` records candidate, selected, delivered, and cited evidence. Legacy `cited_fact_ids` also contains delivered card IDs.",
            "Usage reports attempts, completions, failures, unknown usage, and token counters. Legacy `llm_calls` counts successful recall operations.",
            "",
            "Provider failures return cards with `provider_unavailable`. Delivered suspect or conflicting facts carry explicit warnings.",
            "`recall_trace` adds bounded prompt excerpts and error traces. `recall_stats` reads recorded usage.",
            "",
            "`doctor` reads storage without recovery writes. Repair flags opt into the corresponding repair operation.",
            "Review pending candidates with `list-candidates`, then approve or reject explicit IDs.",
            "",
        ]
    )
    return "\n".join(lines)


async def update_references(root: Path, *, check: bool = False) -> list[str]:
    generated = {
        "docs/config.md": configuration_markdown(),
        "docs/mcp-tools.md": await tools_markdown(),
    }
    stale: list[str] = []
    for relative, content in generated.items():
        path = root / relative
        if not path.is_file() or path.read_text() != content:
            stale.append(relative)
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = asyncio.run(update_references(Path.cwd(), check=args.check))
    if stale:
        print(
            ("Stale references: " if args.check else "Updated references: ")
            + ", ".join(stale)
        )
    return int(bool(stale) and args.check)


if __name__ == "__main__":
    raise SystemExit(main())

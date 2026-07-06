# AGENTS.md - Engram

Engram is structured, cross-project memory for coding agents, served over MCP.
It uses LLM-powered extraction and tiered retrieval over an append-only event
log. There are no embeddings and no vector DB.

User-level guidance (tone, principles, git etiquette) lives in
`~/.claude/CLAUDE.md` and `~/dotfiles/agents/AGENTS.md` and is *not* duplicated
here. This file is for project-specific facts.

## Layout

```
src/engram/
├── server.py       FastMCP entrypoint, tool definitions, auto-sync lifespan
├── cli.py          `engram` console-script command surface
├── operations.py   shared operation layer behind the MCP tools and CLI
├── core/           domain models, agent-facing contracts, config
├── storage/        append-only event-log store + AsyncFactStore + git sync
├── llm/            litellm wrapper
├── extraction/     natural language to structured facts
├── recall/         tiered retrieval + recall@k eval harness
├── maintenance/    synthesize / audit / doctor upkeep
└── dashboard/      Textual TUI (`engram-dash`)

tests/              pytest suite + eval datasets and runners
docs/               Subsystem docs; start with the Index below
```

Annotated module map and data flow: [docs/architecture.md](docs/architecture.md).

## Quickstart

```bash
uv sync --extra dev                          # install deps
uv run pre-commit install                    # git hooks: ruff check + format
uv run engram                                # start MCP server
uv run fastmcp dev src/engram/server.py      # dev mode (inspector)
uv run --extra dev pytest tests/ -q          # run tests
uv run --extra dev ruff check .              # lint (CI gate)
uv run --extra dev ruff format --check .     # format check (CI gate)
```

## Critical Conventions

- **The event log is append-only.** Mutations (`forget`, `edit_fact`,
  `mark_stale`, ...) append typed `FactEvent` records. Never rewrite
  `facts.jsonl`. The only methods that rewrite it are `purge`, `repair`, and
  `compact_event_log`; see [src/engram/storage/store.py](src/engram/storage/store.py)
  and [docs/data.md](docs/data.md).
- **Storage is sync; the MCP surface is async.** All tools are async and reach
  storage through the `AsyncFactStore` `asyncio.to_thread` facade
  ([src/engram/storage/store.py](src/engram/storage/store.py)). Don't call the
  blocking `FactStore` directly from async code.
- **MCP tool names are a stable public surface.** Agents wire to them by name;
  renaming or removing one is a breaking change. Tool defs live in
  [src/engram/server.py](src/engram/server.py); see [docs/mcp-tools.md](docs/mcp-tools.md).
- **All config is `ENGRAM_*` env vars** via pydantic-settings (`env_prefix =
  "ENGRAM_"`) in [src/engram/core/config.py](src/engram/core/config.py). Add
  settings there, not ad-hoc `os.environ` reads; document them in
  [docs/config.md](docs/config.md).
- Never commit secrets, `.env`, or AI-attribution lines.

## Read The Docs First

Before editing a subsystem, read the matching doc:

- **Architecture / data flow**: [docs/architecture.md](docs/architecture.md)
- **MCP tool surface**: [docs/mcp-tools.md](docs/mcp-tools.md)
- **Storage / event log / JSONL files**: [docs/data.md](docs/data.md)
- **Cross-machine sync**: [docs/sync.md](docs/sync.md)
- **Config / env vars**: [docs/config.md](docs/config.md)

If a doc disagrees with code, fix the doc in the same change.

## Index

Start in [docs/architecture.md](docs/architecture.md), then follow the
subsystem docs above.

# Engram

Structured, cross-project memory for coding agents via MCP. No embeddings, no vector DB, just LLM-powered extraction and retrieval.

## Quick Start

```bash
uv sync --extra dev                              # install deps
uv run pre-commit install                        # git hooks: ruff check + format
uv run engram                                    # start MCP server
uv run fastmcp dev src/engram/server.py          # dev mode (inspector)
uv run --extra dev pytest tests/ -v              # run tests
```

## Architecture

Top-level `server.py`, `cli.py`, and `operations.py` are the entrypoints; the
rest is grouped into subpackages by concern.

```
server.py                FastMCP entrypoint, tool definitions, auto-sync lifespan
cli.py                   `engram` console-script command surface
operations.py            shared operation layer behind the MCP tools and CLI
core/                    domain models + agent-facing contracts + config
  models.py              Fact, FactEvent, MemoryCandidate, IngestionRecord,
                         RecallRecord, StoreTransaction
  config.py              pydantic-settings (env prefix: ENGRAM_)
  interfaces.py          Envelope / error / warning codes (stable JSON contract)
  structured_outputs.py  pydantic schemas for structured LLM responses
  provenance.py          recall provenance and trace data structures
storage/                 persistence
  store.py               append-only event-log storage + AsyncFactStore facade,
                         prefilter, candidate review, transaction journal
  sync.py                git-backed sync of the data directory, background loop
llm/                     litellm wrapper (`engram.llm` re-exports the client)
extraction/              natural language → facts
  observer.py            fact extraction & suggestion queueing (structured output)
  importer.py            bootstrap from Claude Code memory files
recall/                  retrieval
  retriever.py           tiered: deterministic fast paths → multi-lens synthesis
  evals.py               recall@k harness used by tests/run_evals.py
maintenance/             memory upkeep
  synthesizer.py         batch LLM consolidation (keep/remove/rewrite/merge)
  memory_audit.py        no-key duplicate / stale / contradiction review
  doctor.py              read-only health diagnostics (with opt-in repair)
dashboard/               Textual TUI (`engram-dash`)
```

**Data flow:** natural language → `extraction.observer` extracts structured facts → `storage.store` persists as JSONL via `AsyncFactStore` → `recall.retriever` runs deterministic fast paths first, escalating to multi-lens search + synthesis only for complex queries.

## MCP Tools


| Tool                                       | Purpose                                               |
| ------------------------------------------ | ----------------------------------------------------- |
| `remember`                                 | Store memories (extracts facts from natural language) |
| `suggest_memories`                         | Propose candidates for human review                   |
| `list_candidates`                          | Browse pending/reviewed suggestions                   |
| `approve_candidates` / `reject_candidates` | Promote or dismiss candidates                         |
| `recall`                                   | Tiered multi-lens search (`max_sources`; MCP `limit` alias) |
| `recall_context`                           | Recall as answer or compact prompt block              |
| `recall_trace`                             | Recall + bounded prompt/output excerpts for debugging (`limit` alias) |
| `recall_stats`                             | Per-recall LLM usage and cache-hit summary            |
| `forget` / `edit_fact`                     | Soft-delete or edit a fact in place                   |
| `correct_memory` / `merge_memories`        | Agent-first correction and merge primitives           |
| `mark_stale` / `unmark_stale`              | Toggle a fact's recall eligibility                    |
| `inspect`                                  | Browse stored facts                                   |
| `import_memories`                          | Bootstrap from `~/.claude/projects/*/memory/`         |
| `memory_stats`                             | Counts, storage size, category breakdown              |
| `synthesize`                               | Batch dedupe / merge / rewrite / prune                |
| `purge` / `rename_project`                 | Permanently drop forgotten/expired or rename a scope  |
| `doctor`                                   | Read-only health diagnostics (with opt-in `repair`)   |
| `sync`                                     | Git-backed pull + push of the data directory          |


## Data

All data lives under `~/.engram/data/`:

- `facts.jsonl` — append-only event log. First line is the
  `{"meta":"event-log-v1",...}` sentinel; subsequent lines are typed
  `FactEvent` records (`created`, `edited`, `forgotten`, `restored`, `stale`,
  `unstale`, `superseded`). Current state per `fact_id` is materialized by
  replaying events in order.
- `candidates.jsonl` — suggested memories pending review
- `ingestion_log.jsonl` — audit trail
- `recall_log.jsonl` — recall quality / latency observability
- `transactions.jsonl` — prepared/committed markers for crash-safe candidate approval
- `facts.jsonl.pre-eventlog` — one-shot backup written the first time a legacy
  (rewrite-format) store is migrated to the event log. Safe to delete after
  you've verified the new file looks right.
- `.engram-sync-state` — last successful sync timestamp + commit counts (only
  exists when `engram sync` has been run).
- `.gitignore` / `.gitattributes` — managed by `engram sync` on first run.
  The gitignore excludes lock and per-machine state files; gitattributes
  configures `merge=union` for the event-log files so parallel appends from
  two machines auto-merge.

### Sync across machines

```bash
# One-time setup on machine A
cd ~/.engram/data
git init -b main
git remote add origin git@github.com:you/your-engram-data.git  # private!
engram sync       # writes managed .gitignore + .gitattributes, pushes setup

# On machine B
git clone git@github.com:you/your-engram-data.git ~/.engram/data
engram sync       # pulls A's state; subsequent syncs are pull + push
```

The first sync auto-commits a managed `.gitignore` and `.gitattributes`.
After that, regular usage is just `engram sync` whenever you want to push or
pull. Set `ENGRAM_SYNC_ENABLED=true` to have the MCP server run sync
automatically on `ENGRAM_SYNC_INTERVAL` (default 300s) and once on shutdown.

**Conflict model**: appends from two machines auto-merge thanks to
`merge=union`. Same-fact concurrent edits resolve by event timestamp on read,
with both events kept in the log for audit. `engram doctor` surfaces sync
state under the `counts.sync` group.

**Rollback**: stop Engram, replace `facts.jsonl` with
`facts.jsonl.pre-eventlog`, downgrade the package, restart.

## Config

All settings via `ENGRAM_*` env vars (pydantic-settings). Key knobs:


| Env Var                      | Default               | Description                        |
| ---------------------------- | --------------------- | ---------------------------------- |
| `ENGRAM_LLM_MODEL`           | `openai/gpt-5.4-mini` | LLM for extraction & search agents |
| `ENGRAM_MAX_FACTS_PER_AGENT` | `200`                 | Facts fed to each search agent     |
| `ENGRAM_RETRIEVAL_TIMEOUT`   | `15.0`                | Search agent timeout (seconds)     |
| `ENGRAM_TIER2_MIN_PREFILTER_COUNT` | `11`            | Tier-2 requires at least this many positive-scoring prefilter matches. `0` disables the small-corpus cap. |
| `ENGRAM_DATA_DIR`            | `~/.engram/data`      | Storage directory                  |
| `ENGRAM_SYNC_ENABLED`        | `false`               | Run background auto-sync inside the MCP server lifespan. |
| `ENGRAM_SYNC_INTERVAL`       | `300.0`               | Background auto-sync cadence (seconds). |
| `ENGRAM_SYNC_TIMEOUT`        | `30.0`                | Timeout for each underlying `git` invocation. |

The MCP `recall_stats` tool summarises LLM usage pulled from the recall log:
total LLM calls, input tokens, cached (prefix-hit) input tokens, and the
resulting cache hit ratio. Providers that don't report usage leave those
fields blank — the stats view renders `-` for any column with no data.

Recall logs keep `selector_version="v2"` for continuity with existing
`recall_stats` output.


## Dev Notes

- Python 3.11+, managed with `uv`
- FastMCP 2.x for the MCP server surface
- litellm for model-agnostic LLM calls
- All MCP tools are async; storage I/O is synchronous behind an `AsyncFactStore` `asyncio.to_thread` facade
- Facts have: category, content, confidence, timestamps, project scope, supersession chain, source metadata
- Storage is an append-only event log. Mutations (`forget`, `edit_fact`,
  `mark_stale`, etc.) append typed `FactEvent` records rather than rewriting
  the file. The only paths that still call `_rewrite` are `purge`, `repair`,
  and (future) compaction inside `synthesize`.
- Sync requires `git` on PATH. The first run of `engram sync` writes managed
  `.gitignore` and `.gitattributes` to the data dir; subsequent runs are
  idempotent.

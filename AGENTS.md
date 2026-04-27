# Engram

Structured, cross-project memory for coding agents via MCP. No embeddings, no vector DB, just LLM-powered extraction and retrieval.

## Quick Start

```bash
uv sync --extra dev                              # install deps
uv run engram                                    # start MCP server
uv run fastmcp dev src/engram/server.py          # dev mode (inspector)
uv run --extra dev pytest tests/ -v              # run tests
```

## Architecture

```
server.py      FastMCP entrypoint, tool definitions
store.py       JSONL storage + AsyncFactStore facade, prefilter, candidate review,
               transaction journal for candidate approval
observer.py    Fact extraction & suggestion queueing (LLM structured output)
retriever.py   Tiered: deterministic fast paths → multi-lens search + synthesis
synthesizer.py Batch LLM consolidation (keep/remove/rewrite/merge)
importer.py    Bootstrap from Claude Code memory files
llm.py         litellm wrapper
config.py      pydantic-settings (env prefix: ENGRAM_)
models.py      Fact, MemoryCandidate, IngestionRecord, RecallRecord, StoreTransaction
```

**Data flow:** natural language → `observer` extracts structured facts → `store` persists as JSONL via `AsyncFactStore` → `retriever` runs deterministic fast paths first, escalating to multi-lens search + synthesis only for complex queries.

## MCP Tools


| Tool                                       | Purpose                                               |
| ------------------------------------------ | ----------------------------------------------------- |
| `remember`                                 | Store memories (extracts facts from natural language) |
| `suggest_memories`                         | Propose candidates for human review                   |
| `list_candidates`                          | Browse pending/reviewed suggestions                   |
| `approve_candidates` / `reject_candidates` | Promote or dismiss candidates                         |
| `recall`                                   | Tiered multi-lens search                              |
| `recall_context`                           | Recall as answer or compact prompt block              |
| `recall_trace`                             | Recall + bounded prompt/output excerpts for debugging |
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


## Data

All data lives under `~/.engram/data/`:

- `facts.jsonl` — active + forgotten facts
- `candidates.jsonl` — suggested memories pending review
- `ingestion_log.jsonl` — audit trail
- `recall_log.jsonl` — recall quality / latency observability
- `transactions.jsonl` — prepared/committed markers for crash-safe candidate approval

## Config

All settings via `ENGRAM_*` env vars (pydantic-settings). Key knobs:


| Env Var                      | Default               | Description                        |
| ---------------------------- | --------------------- | ---------------------------------- |
| `ENGRAM_LLM_MODEL`           | `openai/gpt-5.4-mini` | LLM for extraction & search agents |
| `ENGRAM_MAX_FACTS_PER_AGENT` | `200`                 | Facts fed to each search agent     |
| `ENGRAM_RETRIEVAL_TIMEOUT`   | `15.0`                | Search agent timeout (seconds)     |
| `ENGRAM_TIER_RULES`          | `v2`                  | Tier selector rules: `v2` (default, caps tier-2 when prefilter is small) or `v1` (pre-cap behaviour) |
| `ENGRAM_TIER2_MIN_PREFILTER_COUNT` | `11`            | Under v2, tier-2 requires at least this many positive-scoring prefilter matches. `0` disables the cap. |
| `ENGRAM_DATA_DIR`            | `~/.engram/data`      | Storage directory                  |

The MCP `recall_stats` tool summarises LLM usage pulled from the recall log:
total LLM calls, input tokens, cached (prefix-hit) input tokens, and the
resulting cache hit ratio. Providers that don't report usage leave those
fields blank — the stats view renders `-` for any column with no data.

When the log contains recalls from more than one selector version (`v1` and
`v2`), `recall_stats` renders a separate breakdown per version so you can
A/B the tier thresholds on real traffic. Single-version logs get a single
summary with the active version noted inline.


## Dev Notes

- Python 3.11+, managed with `uv`
- FastMCP 2.x for the MCP server surface
- litellm for model-agnostic LLM calls
- All MCP tools are async; storage I/O is synchronous behind an `AsyncFactStore` `asyncio.to_thread` facade
- Facts have: category, content, confidence, timestamps, project scope, supersession chain, source metadata


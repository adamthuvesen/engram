# Architecture

The top-level `server.py`, `cli.py`, and `operations.py` files are the
entrypoints. The rest of `src/engram/` is grouped by concern.

```
server.py                FastMCP entrypoint, tool definitions, auto-sync lifespan
cli.py                   `engram` console-script command surface
operations.py            shared operation layer behind the MCP tools and CLI
core/                    domain models + agent-facing contracts + config
  models.py              Fact, FactEvent, MemoryCandidate, Durability,
                         RecallRecord, StoreTransaction
  config.py              pydantic-settings (env prefix: ENGRAM_)
  projects.py            project canonicalization (paths/worktrees → repo name)
  interfaces.py          Envelope / error / warning codes (stable JSON contract)
  structured_outputs.py  pydantic schemas for structured LLM responses
  provenance.py          recall provenance and trace data structures
storage/                 persistence
  store.py               append-only event-log storage + AsyncFactStore facade,
                         snapshot cache, atomic ChangeSet apply, candidate
                         review, transaction journal
  search.py              tokenizer + BM25F index (IDF, bigrams, freshness decay)
  sync.py                git-backed sync of the data directory, background loop
llm/                     litellm wrapper (`engram.llm` re-exports the client)
extraction/              natural language to facts
  observer.py            ingest: one LLM call extracts cards and reconciles
                         them against BM25 neighbors (replace/retire/duplicate)
  importer.py            bootstrap from Claude Code memory files
recall/                  retrieval
  retriever.py           cards (BM25 + relevance bar, no LLM) or answer mode;
                         zero-hit LLM selection
  evals.py               recall@k harness used by tests/run_evals.py
maintenance/             memory upkeep
  upkeep.py              orchestrator, lock, background loop
  verify.py              anchor checks against the project's git repo
  consolidate.py         LLM cluster consolidation (auto-applied)
  briefs.py              per-project brief cards
  upkeep_state.py        maintenance_state.jsonl (append-only, union-merged)
  memory_audit.py        no-key duplicate / stale / contradiction review
  doctor.py              read-only health diagnostics (with opt-in repair)
dashboard/               Textual TUI (`engram-dash`)
```

## Data flow

**Write.** `remember` canonicalizes the project (a path resolves to its git
repo's name and records the repo root in `projects.local.json`), then
`extraction.observer.ingest` pulls up to 25 BM25 neighbors of the input and
makes one structured LLM call. Each output card carries a `memory_key` (reused
from a neighbor when it is the same memory), `durability`, `anchors`, and
optional `replaces` / `duplicate_of`; the response may also `retire` neighbors
the input shows are no longer true. References are validated against the
neighbor set and project scope, then applied as one `ChangeSet` (created +
superseded + stale events in a single append). Each target carries the
`updated_at` it was read at, so a card changed during the LLM call is not
overwritten; ingest retries once, then stores the new cards without the
conflicting replacements.

**Read.** `recall` searches the `SearchIndex` (BM25F over content, key, hints,
tags; stemmed unigrams plus bigrams; ephemeral/event cards decay with a 30-day
half-life; suspect cards are down-weighted). A hit is relevant when it covers
enough of the query's IDF mass and scores near the top hit. Cards mode returns
those as dated lines with no LLM call; when nothing is relevant and a key is
configured, one small LLM call selects cards from the top 30. Answer mode
synthesizes over the relevant cards with one call.

**Upkeep.** `maintenance.upkeep.run_upkeep` runs under a per-machine file lock:
`projects` canonicalizes scopes; `verify` checks anchors against the repo
(missing everywhere → stale, missing only from the checkout → suspect, back →
restored); `consolidate` clusters new cards with their neighbors and asks the
LLM for the minimal coherent set, applied all-or-nothing per cluster; `briefs`
refreshes a `project-brief` card per project. The MCP server runs it every
`ENGRAM_MAINTENANCE_INTERVAL` seconds.

## Dev notes

- Python 3.11+, managed with `uv`.
- FastMCP 3.x for the MCP server surface.
- litellm for model-agnostic LLM calls.
- All MCP tools are async. Storage I/O is synchronous behind an
  `AsyncFactStore` `asyncio.to_thread` facade.
- Fact records have: memory key, content, retrieval hints, category,
  confidence, timestamps, project scope, supersession and consolidation
  provenance, and source metadata.
- The MCP server and CLI are adapters over the same operations. MCP provides
  local typed tools for agents over stdio or loopback HTTP (`engram serve`). The
  CLI serves humans and batch scripts.

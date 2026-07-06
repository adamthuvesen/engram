# Architecture

The top-level `server.py`, `cli.py`, and `operations.py` files are the
entrypoints. The rest of `src/engram/` is grouped by concern.

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
extraction/              natural language to facts
  observer.py            fact extraction & suggestion queueing (structured output)
  importer.py            bootstrap from Claude Code memory files
recall/                  retrieval
  retriever.py           tiered: deterministic fast paths, then multi-lens synthesis
  evals.py               recall@k harness used by tests/run_evals.py
maintenance/             memory upkeep
  synthesizer.py         batch LLM consolidation (keep/remove/rewrite/merge)
  memory_audit.py        no-key duplicate / stale / contradiction review
  doctor.py              read-only health diagnostics (with opt-in repair)
dashboard/               Textual TUI (`engram-dash`)
```

## Data flow

Natural language enters `extraction.observer`, which extracts structured facts.
`storage.store` persists those facts as JSONL through `AsyncFactStore`.
`recall.retriever` runs deterministic fast paths first and escalates to
multi-lens search plus synthesis for complex queries.

## Dev notes

- Python 3.11+, managed with `uv`.
- FastMCP 3.x for the MCP server surface.
- litellm for model-agnostic LLM calls.
- All MCP tools are async. Storage I/O is synchronous behind an
  `AsyncFactStore` `asyncio.to_thread` facade.
- Facts have: category, content, confidence, timestamps, project scope,
  supersession chain, source metadata.

# Engram

Structured, cross-project memory for coding agents via MCP.

The LLM is the search engine. No embeddings and no vector database.

## How It Works

1. **Store** — natural language → LLM extracts structured facts → persisted as JSONL
2. **Review** — optionally queue suggestions for human approval before promotion
3. **Recall** — deterministic prefilter → tiered retrieval → direct answer, single-agent synthesis, or multi-lens synthesis

## Quick Start

```bash
uv sync --extra dev                              # install deps
uv run engram                                    # start MCP server
uv run fastmcp dev src/engram/server.py          # dev mode (inspector)
uv run --extra dev pytest tests/ -v              # run tests
```

### MCP Client Config

```json
{
  "mcpServers": {
    "engram": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/engram", "engram"]
    }
  }
}
```

## Tools

| Tool                                       | Purpose                                                           |
| ------------------------------------------ | ----------------------------------------------------------------- |
| `remember`                                 | Store memories (extracts facts from natural language)             |
| `suggest_memories`                         | Propose candidates for human review                               |
| `list_candidates`                          | Browse pending/reviewed suggestions                               |
| `approve_candidates` / `reject_candidates` | Promote or dismiss candidates                                     |
| `recall`                                   | Search memory; pass `format="json"` for an agent-friendly envelope |
| `recall_trace`                             | Recall + structured debug trace (bounded prompt/output excerpts)  |
| `recall_context`                           | Recall as answer or compact prompt block                          |
| `correct_memory`                           | Replace a fact with a superseding new fact (audit preserved)      |
| `merge_memories`                           | Consolidate two or more facts into one                            |
| `mark_stale` / `unmark_stale`              | Toggle a fact's recall eligibility without deleting history       |
| `forget`                                   | Soft-delete a fact                                                |
| `inspect`                                  | Browse stored facts (`include_stale=True` to see stale facts)     |
| `doctor`                                   | Health check store, supersession graph, candidates, provider      |
| `import_memories`                          | Bootstrap from `~/.claude/projects/*/memory/`                     |
| `memory_stats`                             | Counts, storage size, category breakdown                          |

### Agent-first output

Default tool responses remain concise text for human readers. Pass an explicit
parameter to opt into structured JSON envelopes:

- `recall(query, format="json", with_provenance=True)` returns
  `{status, data: {answer, tier, source_fact_ids, cited_fact_ids, provenance, usage}, warnings, errors, meta}`.
- `recall_trace(query)` always returns JSON and adds a bounded `trace` payload
  with per-call prompt/output excerpts and timing.
- Maintenance tools (`correct_memory`, `merge_memories`, `mark_stale`,
  `doctor`) always return JSON envelopes with stable `status` and error
  codes (`validation_error`, `not_found`, `provider_error`, `storage_error`,
  `conflict`).
- All list-style outputs (sources, prefilter matches, inspection results,
  trace excerpts) carry default safety caps; truncation is reported in
  `meta.truncated` and `meta.truncation_reason`.

Stable error codes and warning codes (`stale_fact`, `superseded_fact`,
`forgotten_fact`, `conflicting_facts`, `truncated_output`,
`provider_unavailable`) live in `engram.interfaces` and will not be renamed.

### Agent loops

Common agent workflows the new tools enable:

1. **recall → trace → correct/merge/stale → eval**
   - Run `recall(query, format="json")` to get an answer plus source IDs.
   - When something looks off, run `recall_trace(query)` for tier, prompt
     excerpts, and citations.
   - Use `correct_memory`, `merge_memories`, or `mark_stale` to fix the
     underlying facts (audit history is preserved).
   - Add a regression fixture via the `engram.evals` harness so the next
     change must keep it passing.

2. **doctor → repair → re-doctor**
   - `doctor` is read-only by default. It groups issues by category
     (`storage`, `config`, `provider`, `recall_log`, `facts`, `candidates`)
     and includes repair guidance per issue.
   - Re-run with `repair=True` plus `repair_jsonl=True` and/or
     `recover_transactions=True` to roll forward safe fixes.

### CLI

The same surfaces are exposed as `engram` subcommands; bare `engram` still
starts the MCP server. Every subcommand accepts `--json` and exits non-zero
on errors (1 validation, 2 not-found, 3 runtime, 4 doctor error).

```bash
engram trace "what does adam prefer for editors?" --json
engram doctor --json
engram doctor --check-provider --json
engram correct <fact_id> "new content" --reason "user updated"
engram merge <id1> <id2> --content "merged" --reason "dedupe"
engram stale <fact_id> --reason "outdated"
engram inspect --include-stale --json --limit 50
```

### Privacy and verbosity in trace output

`recall_trace` returns bounded excerpts of every LLM prompt and output by
default (about 2,000 chars per field). Pass `verbose=True` to widen the
limit; full transcripts are never exposed unless explicitly requested.
`meta.truncated` tells callers when an excerpt has been clipped.


## Architecture

```
server.py      FastMCP entrypoint, tool definitions
store.py       Sync JSONL storage plus async facade for server access
observer.py    Fact extraction & suggestion queueing (LLM structured output)
retriever.py   Deterministic prefilter → tiered recall → synthesis
importer.py    Bootstrap from Claude Code memory files
llm.py         litellm wrapper
config.py      pydantic-settings (env prefix: ENGRAM_)
models.py      Fact, Candidate, Category, audit models
```

## Configuration

All settings via `ENGRAM_*` env vars (pydantic-settings). Key knobs:


| Env Var                      | Default               | Description                        |
| ---------------------------- | --------------------- | ---------------------------------- |
| `ENGRAM_LLM_MODEL`           | `openai/gpt-5.4-mini` | LLM for extraction & search agents |
| `ENGRAM_MAX_FACTS_PER_AGENT` | `200`                 | Facts fed to each search agent     |
| `ENGRAM_RETRIEVAL_TIMEOUT`   | `15.0`                | Search agent timeout (seconds)     |
| `ENGRAM_TIER_RULES`          | `v2`                  | Tier selector rules: `v2` (default, caps tier-2 when prefilter is small) or `v1` (pre-cap behaviour) |
| `ENGRAM_TIER2_MIN_PREFILTER_COUNT` | `11`            | Under v2, tier-2 requires at least this many positive-scoring prefilter matches. `0` disables the cap. |
| `ENGRAM_DATA_DIR`            | `~/.engram/data`      | Storage directory                  |

`recall_stats` reports LLM usage per recall when the provider exposes it:
total calls, input tokens, cached (prefix-hit) tokens, and the resulting cache
hit ratio. Missing values render as `-`.


## Data

All data lives under `~/.engram/data/`:

- `facts.jsonl` — active + forgotten facts
- `candidates.jsonl` — suggested memories pending review
- `ingestion_log.jsonl` — audit trail
- `transactions.jsonl` — prepared/committed journal for recoverable multi-file writes

## Tech

Python 3.11+ · FastMCP 2.x · litellm · pydantic-settings · JSONL storage

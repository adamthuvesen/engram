# Engram

Quick, lightweight, structured cross-project memory for coding agents via MCP or CLI.

The LLM is the search engine — no embeddings, no vector database.

## How It Works

1. **Store** — natural language → LLM extracts structured facts → JSONL on disk
2. **Review** — optionally queue suggestions for human approval
3. **Recall** — deterministic prefilter → tiered retrieval → synthesis

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

| Tool | Purpose |
| --- | --- |
| `remember` | Store memories (extracts facts from natural language) |
| `suggest_memories` | Propose candidates for human review |
| `list_candidates` / `approve_candidates` / `reject_candidates` | Manage candidates |
| `recall` | Search memory; `format="json"` for an agent envelope |
| `recall_trace` | Recall + bounded prompt/output excerpts (always JSON) |
| `recall_context` | Recall as answer or compact prompt block |
| `correct_memory` / `merge_memories` | Supersede or consolidate facts (audit preserved) |
| `mark_stale` / `unmark_stale` / `forget` | Toggle eligibility or soft-delete |
| `inspect` / `memory_stats` / `recall_stats` | Browse and inspect |
| `doctor` | Health check (read-only; opt-in `repair`) |
| `import_memories` | Bootstrap from `~/.claude/projects/*/memory/` |

### Agent-first output

Default responses are concise text. Opt into JSON envelopes:

- `recall(query, format="json", with_provenance=True)` →
  `{status, data: {answer, tier, source_fact_ids, cited_fact_ids, provenance, usage}, warnings, errors, meta}`
- Maintenance tools (`correct_memory`, `merge_memories`, `mark_stale`, `doctor`) always return JSON with stable `status` and error codes (`validation_error`, `not_found`, `provider_error`, `storage_error`, `conflict`).
- Lists carry default safety caps; truncation is reported in `meta.truncated`.

Stable codes (`stale_fact`, `superseded_fact`, `forgotten_fact`, `conflicting_facts`, `truncated_output`, `provider_unavailable`) live in `engram.interfaces`.

### CLI

The same surfaces are exposed as `engram` subcommands; bare `engram` starts the MCP server, while `engram --help` and `engram help` show CLI help. Every subcommand accepts `--json` and exits non-zero on errors (1 validation, 2 not-found, 3 runtime, 4 doctor).

```bash
engram recall "what does adam prefer for editors?" --json --with-provenance
engram recall-trace "what does adam prefer for editors?" --json
engram doctor --check-provider --json
engram correct-memory <fact_id> "new content" --reason "user updated"
engram merge-memories <id1> <id2> --content "merged" --reason "dedupe"
engram mark-stale <fact_id> --reason "outdated"
engram inspect --include-stale --json --limit 50
```

Canonical CLI names are hyphenated versions of the MCP tools: `suggest-memories`, `list-candidates`, `approve-candidates`, `reject-candidates`, `recall-context`, `edit-fact`, `import-memories`, `rename-project`, `memory-stats`, and `recall-stats`. Existing short commands remain as aliases: `trace`, `correct`, `merge`, `stale`, and `unstale`.

`approve-candidates ID... --edit id="new content"` applies edits before approval. `synthesize` is a dry run by default; pass `--apply` to write changes. `recall_trace` excerpts are bounded (~2,000 chars/field); pass `--verbose` in the CLI or `verbose=True` in MCP to widen.

## Architecture

```
operations.py  Shared application layer used by MCP and CLI adapters
server.py      FastMCP entrypoint, thin tool wrappers
cli.py         CLI entrypoint, argument parsing, aliases, exit codes
store.py       Sync JSONL storage plus async facade
observer.py    Fact extraction & suggestion queueing (LLM structured output)
retriever.py   Deterministic prefilter → tiered recall → synthesis
importer.py    Bootstrap from Claude Code memory files
llm.py         litellm wrapper
config.py      pydantic-settings (env prefix: ENGRAM_)
models.py      Fact, Candidate, Category, audit models
```

## Configuration

All settings via `ENGRAM_*` env vars. Key knobs:

| Env Var | Default | Description |
| --- | --- | --- |
| `ENGRAM_LLM_MODEL` | `openai/gpt-5.4-mini` | LLM for extraction & search |
| `ENGRAM_MAX_FACTS_PER_AGENT` | `200` | Facts fed to each search agent |
| `ENGRAM_RETRIEVAL_TIMEOUT` | `15.0` | Search agent timeout (seconds) |
| `ENGRAM_TIER_RULES` | `v2` | `v2` caps tier-2 on small prefilter; `v1` uses pre-cap behavior |
| `ENGRAM_TIER2_MIN_PREFILTER_COUNT` | `11` | Min prefilter matches for tier-2 under v2 (`0` disables) |
| `ENGRAM_TIER2_MODE` | `single` | Tier-2 strategy: `single` or `multilens` |
| `ENGRAM_DATA_DIR` | `~/.engram/data` | Storage directory |

## Data

All data under `~/.engram/data/`:

- `facts.jsonl` — active + forgotten facts
- `candidates.jsonl` — suggested memories pending review
- `ingestion_log.jsonl` — audit trail
- `transactions.jsonl` — prepared/committed journal for recoverable writes

## Tech

Python 3.11+ · FastMCP 2.x · litellm · pydantic-settings · JSONL storage

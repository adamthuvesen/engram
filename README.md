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
uv run --extra dev ruff check .                  # lint
uv run --extra dev ruff format --check .         # formatting check
uv build                                         # build sdist + wheel
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
| `sync` | Git-backed pull + push of the data directory |
| `import_memories` | Bootstrap from `~/.claude/projects/*/memory/` |

### Agent-first output

Default responses are concise text. MCP tools also expose the same envelope as
`structuredContent`, so agent clients do not need to parse JSON out of text.
Opt into JSON envelopes in CLI or text content when needed:

- `recall(query, format="json", with_provenance=True)` →
  `{status, data: {answer, tier, source_fact_ids, cited_fact_ids, provenance, usage}, warnings, errors, meta}`
- `recall` / `recall_trace` cap synthesis with `max_sources` (default 25); `limit` is an alias for `max_sources` on MCP only (CLI: `--max-sources`).
- Maintenance tools (`correct_memory`, `merge_memories`, `mark_stale`, `doctor`) always return JSON with stable `status` and error codes (`validation_error`, `not_found`, `provider_error`, `storage_error`, `conflict`).
- `recall-stats --json` returns aggregate statistics by default; pass `--include-records` for raw recall log records.
- Lists carry default safety caps; truncation is reported in `meta.truncated`.

Stable codes (`stale_fact`, `superseded_fact`, `forgotten_fact`, `conflicting_facts`, `truncated_output`, `provider_unavailable`) live in `engram.interfaces`.

### CLI

The same surfaces are exposed as `engram` subcommands; bare `engram` starts the MCP server, while `engram --help` and `engram help` show CLI help. Every subcommand accepts `--json`. Operation failures use stable exit codes: 1 validation, 2 not-found, 3 runtime, 4 doctor. Argparse usage errors, such as unknown commands or missing required arguments, also exit 2 and print usage to stderr.

```bash
engram recall "what does alex prefer for editors?" --json --with-provenance
engram recall-trace "what does alex prefer for editors?" --json
engram doctor --check-provider --json
engram doctor --repair --repair-orphaned-supersessions --json
engram recall-stats --json --limit 100 --since 2026-05-01
engram correct-memory <fact_id> "new content" --reason "user updated"
engram merge-memories <id1> <id2> --content "merged" --reason "dedupe"
engram mark-stale <fact_id> --reason "outdated"
engram inspect --include-stale --json --limit 50
engram sync --json                                    # git-backed pull + push
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
| `ENGRAM_TIER2_MIN_PREFILTER_COUNT` | `11` | Min prefilter matches for tier-2 (`0` disables the small-corpus cap) |
| `ENGRAM_TIER2_MODE` | `single` | Tier-2 strategy: `single` or `multilens` |
| `ENGRAM_DATA_DIR` | `~/.engram/data` | Storage directory |
| `ENGRAM_SYNC_ENABLED` | `false` | Run background auto-sync inside the MCP server lifespan |
| `ENGRAM_SYNC_INTERVAL` | `300.0` | Background auto-sync cadence (seconds) |
| `ENGRAM_SYNC_TIMEOUT` | `30.0` | Timeout for each underlying `git` invocation |

## Sync across machines

Engram syncs its data directory between machines via a private git repo. No
hosted service required.

```bash
# One-time setup on machine A
cd ~/.engram/data
git init -b main
git remote add origin git@github.com:you/your-engram-data.git  # PRIVATE repo
engram sync           # auto-writes .gitignore + .gitattributes, pushes

# On machine B
git clone git@github.com:you/your-engram-data.git ~/.engram/data
engram sync           # pulls machine A's state
```

After setup, run `engram sync` whenever you want to push or pull. The first
sync auto-commits a managed `.gitignore` (so lock files and per-machine
state don't get tracked) and `.gitattributes` (configures `merge=union` on
the event-log files so parallel appends from two machines auto-merge).

Set `ENGRAM_SYNC_ENABLED=true` to have the MCP server run sync
automatically on `ENGRAM_SYNC_INTERVAL` (default 300s) and once on
shutdown. `engram doctor` reports sync state (`remote_configured`,
`last_sync_at`, `unpushed_commits`) under `counts.sync` — all local,
no network calls.

**Conflict model**: append-only events from two machines merge cleanly via
`merge=union`. Same-fact concurrent edits resolve by event timestamp on
read; both events stay in the log for audit.

**Rollback**: stop Engram, replace `facts.jsonl` with the
`facts.jsonl.pre-eventlog` backup written on the first event-log migration,
downgrade the package, restart.

## Data

All data under `~/.engram/data/`:

- `facts.jsonl` — append-only fact event log; current state is materialized by replaying events
- `candidates.jsonl` — suggested memories pending review
- `ingestion_log.jsonl` — audit trail
- `recall_log.jsonl` — recall quality, latency, and LLM usage history
- `transactions.jsonl` — prepared/committed journal for recoverable writes
- `.engram-sync-state` — last successful sync metadata

`engram sync` also manages `.gitignore` and `.gitattributes` in the data
directory so lock/state files stay local and JSONL event logs use git's union
merge driver.

## Tech

Python 3.11+ · FastMCP 2.x · litellm · pydantic-settings · JSONL storage

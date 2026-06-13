# Engram

Structured, cross-project memory for coding agents. Runs as an MCP server or a CLI.

The LLM is the retrieval engine — no embeddings, no vector database. You store
facts in natural language; an LLM extracts and categorizes them into structured
records on disk. Recall starts with deterministic prefilters and only escalates to
LLM search and synthesis when a query actually needs it. The point is recall you can
audit: every fact keeps its source, supersession chain, and confidence, and storage
is a plain append-only JSONL event log you can read with `cat`.

## How it works

1. **Store** — natural language in; the LLM extracts structured facts; appended to a JSONL event log.
2. **Review** (optional) — queue suggestions as candidates for human approval before they become recallable.
3. **Recall** — deterministic prefilter → tiered retrieval → synthesis, escalating only as far as the query needs.

## Recall, measured

The whole bet is that you can skip embeddings: a deterministic keyword prefilter
handles the easy queries for free, and the LLM tier only runs when a query
actually needs it. Here's that bet measured on a **representative set of 56
labeled queries over a 57-fact corpus** ([`tests/recall_eval_dataset.json`](tests/recall_eval_dataset.json)),
spanning terse literal lookups through harder paraphrased questions:

**35% of queries resolve at tier-0 with zero LLM calls** — and on the rest the
prefilter still ranks the right memory at #1 four times out of five:

| Deterministic prefilter (no LLM, no embeddings) | value |
| ----------------------------------------------- | ----- |
| recall@1 (answer ranked #1)                     | 80%   |
| recall@5 (answer in the top 5)                  | 91%   |
| candidate recall (answer kept in the pool)      | 91%   |
| MRR                                             | 0.85  |

This is the deterministic prefilter *floor*, **not** end-to-end retrieval
accuracy. The aggregate already includes the harder queries the keyword pass
can't resolve on its own; those escalate to the LLM tier — the actual retrieval
engine — which this number deliberately does not measure.

The queries are kept honest on purpose: the literal ones use the term a user
would actually type (not a verbatim copy of the fact), the harder ones reword
around it, and each is labeled to the fact that answers it — so the scorer is
never graded on its own keywords. The `kind` field in the dataset makes the mix
auditable.

Reproduce — deterministic, no API key:

```bash
uv run python tests/run_evals.py
```

## Run it

```bash
uv sync --extra dev        # install (omit --extra dev for runtime only)
uv run engram              # no args → start the MCP server (stdio)
uv run engram --help       # any args → CLI; this lists the subcommands
uv run engram-dash         # terminal dashboard for browsing memory
```

Bare `engram` (no arguments) launches the MCP stdio server. Anything else is treated
as a CLI invocation, so a typo surfaces as an argparse error instead of silently
starting a long-running server.

Engram talks to an LLM via [litellm](https://github.com/BerriAI/litellm), so it needs
whatever credentials your configured model expects (e.g. `OPENAI_API_KEY`). The model
is set with `ENGRAM_LLM_MODEL` (see [Configuration](#configuration)).

### As an MCP server

Point your MCP client at the `engram` entrypoint. Since bare `engram` starts the
server, the command is just `uv run` in the repo:

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

### As a CLI

Every MCP tool has a hyphenated CLI subcommand; bare `engram` starts the server,
`engram --help` lists them. Every subcommand accepts `--json`.

```bash
engram recall "what does alex prefer for editors?" --json --with-provenance
engram recall-trace "what does alex prefer for editors?" --json   # + prompt/output excerpts
engram doctor --check-provider --json
engram inspect --include-stale --json --limit 50
engram correct-memory <fact_id> "new content" --reason "user updated"
engram merge-memories <id1> <id2> --content "merged" --reason "dedupe"
engram sync --json                                                 # git-backed pull + push
```

Short aliases stay available: `trace`, `correct`, `merge`, `stale`, `unstale`.
`approve-candidates ID... --edit id="new content"` edits before approving;
`synthesize` is a dry run unless you pass `--apply`. Operation failures use stable
exit codes (1 validation, 2 not-found, 3 runtime, 4 doctor); argparse usage errors
also exit 2.

## Tools

| Tool | Purpose |
| --- | --- |
| `remember` | Store memories (extracts facts from natural language) |
| `suggest_memories` | Propose candidates for human review |
| `list_candidates` / `approve_candidates` / `reject_candidates` | Manage candidates |
| `recall` | Search memory; `format="json"` for an agent envelope |
| `recall_context` | Recall as an answer or a compact prompt block |
| `recall_trace` | Recall + bounded prompt/output excerpts (always JSON) |
| `correct_memory` / `merge_memories` | Supersede or consolidate facts (audit preserved) |
| `mark_stale` / `unmark_stale` / `forget` | Toggle recall eligibility or soft-delete |
| `inspect` / `memory_stats` / `recall_stats` | Browse and inspect |
| `synthesize` | Batch dedupe / merge / rewrite / prune |
| `doctor` | Health check (read-only; opt-in `repair`) |
| `sync` | Git-backed pull + push of the data directory |
| `import_memories` | Bootstrap from `~/.claude/projects/*/memory/` |

Default tool responses are concise text. MCP tools also expose the same envelope as
`structuredContent`, so agent clients don't have to parse JSON out of text; pass
`format="json"` (or `--json` in the CLI) when you want the envelope inline:

```
recall(query, format="json", with_provenance=True) →
  {status, data: {answer, tier, source_fact_ids, cited_fact_ids, provenance, usage}, warnings, errors, meta}
```

`recall` / `recall_trace` cap synthesis with `max_sources` (default 25; `limit` is an
MCP-only alias). Maintenance tools always return JSON with a stable `status` and
error codes (`validation_error`, `not_found`, `provider_error`, `storage_error`,
`conflict`). Lists carry default safety caps; truncation is reported in
`meta.truncated`. Stable warning codes live in `engram.interfaces`.

## Configuration

All settings are `ENGRAM_*` env vars (pydantic-settings). Key knobs:

| Env var | Default | Description |
| --- | --- | --- |
| `ENGRAM_LLM_MODEL` | `openai/gpt-5.4-mini` | LLM for extraction and search |
| `ENGRAM_MAX_FACTS_PER_AGENT` | `200` | Facts fed to each search agent |
| `ENGRAM_RETRIEVAL_TIMEOUT` | `15.0` | Search agent timeout (seconds) |
| `ENGRAM_TIER2_MIN_PREFILTER_COUNT` | `11` | Min prefilter matches before tier-2 (`0` disables the small-corpus cap) |
| `ENGRAM_TIER2_MODE` | `single` | Tier-2 strategy: `single` or `multilens` |
| `ENGRAM_DATA_DIR` | `~/.engram/data` | Storage directory |
| `ENGRAM_SYNC_ENABLED` | `false` | Run background auto-sync inside the MCP server lifespan |
| `ENGRAM_SYNC_INTERVAL` | `300.0` | Background auto-sync cadence (seconds) |
| `ENGRAM_SYNC_TIMEOUT` | `30.0` | Timeout for each underlying `git` invocation |

## Data

Everything lives under `~/.engram/data/` (override with `ENGRAM_DATA_DIR`):

- `facts.jsonl` — append-only fact event log; current state is materialized by replaying events
- `candidates.jsonl` — suggested memories pending review
- `ingestion_log.jsonl` / `recall_log.jsonl` — audit and recall-quality history
- `transactions.jsonl` — prepared/committed journal for crash-safe writes

## Sync across machines

Engram syncs its data directory between machines through a private git repo — no
hosted service.

```bash
# Machine A, one-time
cd ~/.engram/data
git init -b main
git remote add origin git@github.com:you/your-engram-data.git   # PRIVATE repo
engram sync          # auto-writes managed .gitignore + .gitattributes, pushes

# Machine B
git clone git@github.com:you/your-engram-data.git ~/.engram/data
engram sync          # pulls A's state; later syncs are pull + push
```

The first sync auto-commits a managed `.gitignore` (lock and per-machine state stay
local) and `.gitattributes` (`merge=union` on the event-log files, so parallel
appends from two machines auto-merge). Set `ENGRAM_SYNC_ENABLED=true` to have the MCP
server sync on `ENGRAM_SYNC_INTERVAL` and once on shutdown. `engram doctor` reports
sync state under `counts.sync` — all local, no network calls.

## Development

```bash
uv run pre-commit install                  # git hooks: ruff check + format
uv run --extra dev pytest tests/ -v        # tests
uv run --extra dev ruff check .            # lint
uv build                                   # build sdist + wheel
```

Architecture notes and the storage/event-log model live in [AGENTS.md](AGENTS.md).

Python 3.11+ · FastMCP 2.x · litellm · pydantic-settings · JSONL storage · MIT-licensed.

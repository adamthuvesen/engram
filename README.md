# Engram

![License](https://img.shields.io/github/license/adamthuvesen/engram) ![Python](https://img.shields.io/badge/python-3.11%2B-blue)

Engram is structured, cross-project memory for coding agents. It runs as an MCP
server or a CLI.

Recall uses a score-distribution router over the prefilter results. A sharp
match returns directly with no LLM call. A focused cluster goes to one LLM call.
A broad or flat result set goes to the tier-2 path, which is one broad LLM call.
There are no embeddings and no vector database.

Facts start as natural language, then an LLM extracts structured records onto
disk. Every fact keeps its source, supersession chain, and confidence. The store
is event-sourced: a plain append-only JSONL event log you can read with `cat`.

| Tier | When it runs | LLM calls |
| --- | --- | ---: |
| 0 | Few strong matches, or no relevant matches | 0 |
| 1 | Focused matches with a clear top cluster | 1 |
| 2 | Many matches or a flat score distribution | 1 |

## How it works

1. **Store**: natural language in, structured facts out, appended to a JSONL event log.
2. **Review** (optional): queue suggestions as candidates before they become recallable.
3. **Recall**: prefilter, route by score distribution, then call the LLM only when needed.

## No-key demo

This demo uses the committed eval dataset. It does not read `~/.engram/data`, call
an LLM provider, or need an API key.

```bash
uv sync --extra dev
uv run python tests/run_evals.py
```

Expected shape:

```text
Deterministic prefilter recall — representative query mix
83 answerable labeled queries + 8 no-match queries over a 57-fact corpus  ·  no LLM, no embeddings

43% of queries resolved at tier-0 with zero LLM calls  ·  tiers {0: 39, 1: 42, 2: 10}

metric                       value
----------------------------------
recall@1                       89%
recall@5                       96%
candidate recall (hit-rate)     98%
MRR                           0.92

no-match returns nothing above floor: ok
```

What this covers:

- Recall behavior: runs the committed facts through `recall_with_provenance`.
- Eval behavior: exits non-zero if recall floors or the no-match case regress.
- Dashboard behavior: `uv run engram-dash` opens a terminal UI over local JSONL
  data and does not call an LLM.

## Recall, measured

A deterministic keyword prefilter handles easy queries for free. The LLM tier
runs only when a query needs it. The no-key eval measures that prefilter on **83
answerable labeled queries plus 8 no-match queries over a 57-fact corpus**
([`tests/recall_eval_dataset.json`](tests/recall_eval_dataset.json)):

**43% of queries resolve at tier-0 with zero LLM calls.** For the rest, the
prefilter still keeps the right memory in the deterministic candidate pool:

| Deterministic prefilter (no LLM, no embeddings) | value |
| ----------------------------------------------- | ----- |
| recall@1 (answer ranked #1)                     | 89%   |
| recall@5 (answer in the top 5)                  | 96%   |
| candidate recall (answer kept in the pool)      | 98%   |
| MRR                                             | 0.92  |

This is the deterministic prefilter *floor*. It does not measure end-to-end
retrieval accuracy. The aggregate includes harder queries the keyword pass can't
resolve on its own. Those queries escalate to the LLM retrieval engine, which
this number deliberately leaves out.

The queries are labeled to source facts. The `kind` field shows the mix of
literal, paraphrased, synonym, semantic, and cross-project queries.

Reproduce the deterministic no-key run:

```bash
uv run python tests/run_evals.py
```

## Cross-project recall quality

The cross-project benchmark adds the failure modes that are expensive in real
agent memory: wrong-project evidence, stale facts, superseded contradictory
preferences, global facts, and no-match queries. It is fictional no-secret data
and uses deterministic provenance only, so it needs no API key:

```bash
uv run python tests/run_cross_project_recall_evals.py
```

The headline metric is mean evidence quality. Answerable queries get credit for
expected evidence being retrieved and ranked well, then lose credit if excluded
stale, superseded, contradictory, or wrong-project evidence appears above the
relevance floor. No-match queries only pass when no evidence is surfaced.

Measured on the committed fixture:

| Cross-project quality | score |
| --- | ---: |
| baseline before project/supersession filtering | 0.417 |
| current | 1.000 |
| absolute gain | +0.583 |
| relative gain | +140% |

## Run it

```bash
uv sync --extra dev        # install (omit --extra dev for runtime only)
uv run engram              # no args → start the MCP server (stdio)
uv run engram --help       # any args → CLI; this lists the subcommands
uv run engram-dash         # terminal dashboard for browsing memory
```

Bare `engram` (no arguments) launches the MCP stdio server. Anything else is
treated as a CLI invocation, so a typo surfaces as an argparse error instead of
silently starting a long-running server.

No-key paths:

- `uv run python tests/run_evals.py`
- `uv run python tests/run_cross_project_recall_evals.py`
- `uv run python tests/run_memory_audit_evals.py`
- `uv run engram-dash`
- `uv run engram doctor --json`
- `uv run engram inspect --json --limit 50`
- `uv run engram audit-memories --json`

Runtime paths that can call the LLM:

- `remember` and `suggest-memories`
- `recall`, `recall-context`, and `recall-trace` when a query escalates past
  tier-0
- `doctor --check-provider`

Engram talks to an LLM via [litellm](https://github.com/BerriAI/litellm). These
paths need whatever credentials your configured model expects, such as
`OPENAI_API_KEY`. The model is set with `ENGRAM_LLM_MODEL` (see
[Configuration](#configuration)).

### As an MCP server

Point your MCP client at the `engram` entrypoint. Since bare `engram` starts the
server, the command is `uv run` in the repo:

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

Every MCP tool has a hyphenated CLI subcommand. Bare `engram` starts the server,
and `engram --help` lists the subcommands. Every subcommand accepts `--json`.

```bash
engram recall "what does alex prefer for editors?" --json --with-provenance
engram recall-trace "what does alex prefer for editors?" --json   # + prompt/output excerpts
engram doctor --check-provider --json
engram inspect --include-stale --json --limit 50
engram correct-memory <fact_id> "new content" --reason "user updated"
engram merge-memories <id1> <id2> --content "merged" --reason "dedupe"
engram audit-memories --json                                      # read-only suggestions
engram sync --json                                                 # git-backed pull + push
```

Short aliases stay available: `trace`, `correct`, `merge`, `stale`, `unstale`.
`approve-candidates ID... --edit id="new content"` edits before approving.
Operation failures use stable exit codes (1 validation, 2 not-found, 3 runtime,
4 doctor); argparse usage errors also exit 2.

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
| `audit_memories` | Read-only duplicate / stale / contradiction suggestions |
| `doctor` | Health check (read-only; opt-in `repair`) |
| `sync` | Git-backed pull + push of the data directory |
| `import_memories` | Bootstrap from `~/.claude/projects/*/memory/` |

Default tool responses are concise text. MCP tools also expose the same envelope
as `structuredContent`, so agent clients don't have to parse JSON out of text.
Pass `format="json"` (or `--json` in the CLI) when you want the envelope inline:

```
recall(query, format="json", with_provenance=True) →
  {status, data: {answer, tier, source_fact_ids, cited_fact_ids, provenance, usage}, warnings, errors, meta}
```

`recall` / `recall_trace` cap synthesis with `max_sources` (default 25; `limit`
is an MCP-only alias). Maintenance tools always return JSON with a stable
`status` and error codes (`validation_error`, `not_found`, `provider_error`,
`storage_error`, `conflict`). Lists carry default safety caps, and truncation is
reported in `meta.truncated`. Stable warning codes live in
`engram.core.interfaces`.

## Memory audit suggestions

`audit-memories` is the no-key, read-only compaction review loop. It scans active
facts for near-duplicate groups, stale time-bound memories, and contradictory
preference/update claims, then emits suggested review actions such as
`merge-memories`, `mark-stale`, or manual contradiction review. It does not
apply those actions itself.

Reproduce the measured fixture:

```bash
uv run python tests/run_memory_audit_evals.py
```

The committed fixture is fictional Acme/Alex-style memory data with labels for
duplicate, stale, and contradiction issue groups. The eval compares against the
current no-key audit floor (exact duplicate checks). It requires at least a 50
percentage point recall gain, at least 80% precision, and reviewer burden no
higher than 1.5x the expected issue count.

## Configuration

All settings are `ENGRAM_*` env vars (pydantic-settings). Key knobs:

| Env var | Default | Description |
| --- | --- | --- |
| `ENGRAM_LLM_MODEL` | `openai/gpt-5.4-mini` | LLM for extraction and search |
| `ENGRAM_MAX_FACTS_PER_AGENT` | `200` | Max facts fed to the recall LLM call |
| `ENGRAM_RETRIEVAL_TIMEOUT` | `15.0` | Recall LLM call timeout (seconds) |
| `ENGRAM_TIER2_MIN_PREFILTER_COUNT` | `11` | Minimum prefilter matches before tier-2 (`0` disables the small-corpus cap) |
| `ENGRAM_DATA_DIR` | `~/.engram/data` | Storage directory |
| `ENGRAM_SYNC_ENABLED` | `false` | Run background auto-sync inside the MCP server lifespan |
| `ENGRAM_SYNC_INTERVAL` | `300.0` | Background auto-sync cadence (seconds) |
| `ENGRAM_SYNC_TIMEOUT` | `30.0` | Timeout for each underlying `git` invocation |

## Data

Everything lives under `~/.engram/data/` (override with `ENGRAM_DATA_DIR`):

- `facts.jsonl`: append-only fact event log. Current state comes from replaying events.
- `candidates.jsonl`: suggested memories pending review.
- `recall_log.jsonl`: recall-quality and latency history.
- `transactions.jsonl`: prepared/committed journal for crash-safe writes.

## Sync across machines

Engram syncs its data directory between machines through a private git repo. It
does not need a hosted service.

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

The first sync auto-commits a managed `.gitignore` (lock and per-machine state
stay local) and `.gitattributes` (`merge=union` on the event-log files, so
parallel appends from two machines auto-merge). Set `ENGRAM_SYNC_ENABLED=true`
to have the MCP server sync on `ENGRAM_SYNC_INTERVAL` and once on shutdown.
`engram doctor` reports sync state under `counts.sync`. That check is local and
makes no network calls.

## Development

```bash
uv run pre-commit install                  # git hooks: ruff check + format
uv run --extra dev pytest tests/ -v        # tests
uv run --extra dev ruff check .            # lint
uv run --extra dev mypy                    # type check
uv build                                   # build sdist + wheel
```

Architecture notes live in [docs/architecture.md](docs/architecture.md). The
storage and event-log model lives in [docs/data.md](docs/data.md).

Python 3.11+ · FastMCP 3.x · litellm · pydantic-settings · JSONL storage · MIT-licensed.

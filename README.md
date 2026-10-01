# Engram

![License](https://img.shields.io/github/license/adamthuvesen/engram) ![Python](https://img.shields.io/badge/python-3.11%2B-blue)

Engram is structured, cross-project memory for coding agents. It runs as an MCP
server or a CLI, and keeps itself current: what an agent recalls is ranked,
dated, and scoped to the repository it is working in.

**Recall is lexical and instant.** A BM25 index over memory cards (content,
stable `memory_key`, retrieval hints, tags) returns the few cards that clear a
relevance bar, each stamped with its category, project, and date, in a few
milliseconds and with no LLM call. Only when nothing clears the bar does one
small LLM call pick relevant cards from the top candidates, and `mode="answer"`
opts into a synthesized answer. There are no embeddings and no vector database.

**Saving is one call.** `remember` takes plain language. One LLM call extracts
memory cards and reconciles them against the closest existing cards: it reuses
their `memory_key`, replaces the cards it updates or contradicts, skips what is
already known, and retires what the input shows is no longer true. All of it
lands in one atomic append to an event-sourced JSONL log.

**Memory stays current.** Background upkeep inside the MCP server:

| Step | What it does | LLM |
| --- | --- | --- |
| `projects` | Canonicalizes scopes (paths and worktrees → repo name) | no |
| `verify` | Checks file/symbol anchors against the repo; flags or retires drifted cards | no |
| `consolidate` | Merges fragmented cards, splits overloaded ones, retires junk and contradicted cards | yes |
| `briefs` | Refreshes a per-project brief for session start | yes |

Time-bound (`ephemeral`) memories fade in ranking and expire; `durable` ones hold
until something contradicts them. Every change is an event, so nothing is
rewritten in place.

## How it works

1. **Remember**: plain language in; cards extracted, reconciled, appended atomically.
2. **Recall**: BM25 → relevance bar → dated cards (LLM only for zero-hit or `answer` mode).
3. **Upkeep**: projects → verify → consolidate → briefs, on a schedule or via `engram upkeep`.
4. **Review** (optional): `suggest_memories` queues candidates before they become recallable.

The MCP server and CLI share the same operation layer. MCP is the agent-facing
interface because clients get typed local tools from a persistent process, over
stdio or a loopback HTTP daemon. The CLI is the faster interface for humans,
scripts, audits, and batch maintenance.

## No-key demo

This demo uses the committed eval dataset. It does not read `~/.engram/data`, call
an LLM provider, or need an API key.

```bash
uv sync --extra dev
uv run python tests/run_evals.py
```

Expected shape:

```text
Deterministic lexical recall — representative query mix
83 answerable labeled queries + 8 no-match queries over a 57-fact corpus  ·  no LLM, no embeddings

86% of queries resolve with zero LLM calls even when a key is configured

metric                       value
----------------------------------
recall@1                       70%
recall@5                       76%
candidate recall (hit-rate)     90%
MRR                           0.73

recall@1 by query kind (where lexical search wins vs. where the LLM earns it):
  literal        23/24    96%
  paraphrase     20/21    95%
  semantic       11/28    39%
  synonym         3/9     33%
```

What this covers:

- Recall behavior: runs the committed facts through `recall_with_provenance`.
- Eval behavior: exits non-zero if recall floors or the no-match case regress.
- Dashboard behavior: `uv run engram-dash` opens a terminal UI over local JSONL
  data and does not call an LLM.

## Recall, measured

The labeled dataset ([`tests/recall_eval_dataset.json`](tests/recall_eval_dataset.json))
deliberately over-weights synonym and semantic queries (37 of 83) that share
few words with the stored fact. Lexical search wins literal and paraphrased
queries (95%+ at rank 1); synonym and semantic queries are where the zero-hit
LLM selection earns its call. Candidate recall (90%) counts a hit when the right
card is either returned or among the 30 candidates that LLM call chooses from.

Real agent queries skew toward literal keyword bags. Against 1,826 logged
queries on a 5.6k-card store, the relevance bar returns a median of 4 cards and
sends 32% of queries to the LLM selection call; search itself takes ~2 ms.

[`tests/knowledge_update_eval_fixtures.json`](tests/knowledge_update_eval_fixtures.json)
pins staleness behavior: a superseded card never returns, a stale card is
excluded, an old time-bound event ranks below an equally matching durable card,
suspect cards carry a warning, and scoped queries never leak other projects.

Reproduce the deterministic no-key run:

```bash
uv run python tests/run_evals.py
```

Extraction has a separate no-key contract fixture for claim coverage, card
precision, fragmentation, transient and bookkeeping exclusion, stable keys, and
retrieval hints:

```bash
uv run python tests/run_extraction_quality_evals.py
uv run python tests/run_extraction_quality_evals.py --live   # with provider credentials
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
uv run engram serve --transport http   # shared daemon at http://127.0.0.1:7422/mcp
uv run engram-dash         # terminal dashboard for browsing memory
```

Bare `engram` (no arguments) launches the MCP stdio server. Anything else is
treated as a CLI invocation, so a typo surfaces as an argparse error instead of
silently starting a long-running server. `engram serve` starts the server
explicitly: `--transport stdio` (the default) or `--transport http` with
`--host` (default `127.0.0.1`) and `--port` (default `7422`).

No-key paths:

- `uv run python tests/run_evals.py`
- `uv run python tests/run_cross_project_recall_evals.py`
- `uv run python tests/run_memory_audit_evals.py`
- `uv run engram-dash`
- `uv run engram doctor --json`
- `uv run engram inspect --json --limit 50`
- `uv run engram audit-memories --json`

Runtime paths that can call the LLM:

- `remember` and `suggest-memories` (one call each)
- `recall`, `recall-context`, and `recall-trace` when nothing clears the
  relevance bar, or with `mode="answer"`
- `upkeep` steps `consolidate` and `briefs` (also run in the background)
- `doctor --check-provider`

Engram calls the OpenAI API directly through the official SDK, so these paths
need `OPENAI_API_KEY`. The model is set with `ENGRAM_LLM_MODEL` (default
`openai/gpt-6-luna`, run in OpenAI's fast tier; see
[Configuration](#configuration)).

### As an MCP server

Point your MCP client at the `engram` entrypoint. Since bare `engram` starts the
server, the command is `uv run` in the repo:

Engram pins FastMCP 4.0.0b2 so clients can negotiate MCP `2026-07-28` or an
older protocol revision. Over HTTP, `engram serve` runs stateless streamable
HTTP at `/mcp`: the server keeps no MCP session, so clients keep working across a
daemon restart. One process then serves every client, so concurrent tool calls
share one store and its in-process locks.

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
engram remember "We moved CI to uv; never pip install in workflows" --project ~/dev/app
engram recall "editor preference" --project ~/dev/app            # dated cards, no LLM
engram recall "why did we drop pandas?" --mode answer            # one LLM call
engram recall-context --mode brief --project ~/dev/app           # session-start brief
engram upkeep --dry-run                                          # what upkeep would change
engram recall "what does alex prefer for editors?" --json --with-provenance
engram recall-trace "what does alex prefer for editors?" --json   # + prompt/output excerpts
engram doctor --check-provider --json
engram inspect --include-stale --json --limit 50
engram correct-memory <fact_id> "new content" --reason "user updated"
engram merge-memories <id1> <id2> --content "merged" \
  --memory-key "editor-preference" \
  --retrieval-hints "preferred editor" "editor setup" \
  --reason "dedupe"
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
| `remember` | Save plain language: extract, reconcile, replace, retire (one LLM call) |
| `suggest_memories` | Same extraction, queued as candidates for review |
| `list_candidates` / `approve_candidates` / `reject_candidates` | Manage candidates |
| `recall` | Dated memory cards (`mode="cards"`, default) or an answer (`mode="answer"`) |
| `recall_context` | `brief` (project brief), `cards`, or `answer` |
| `recall_trace` | Recall + bounded prompt/output excerpts (always JSON) |
| `correct_memory` / `merge_memories` | Supersede or consolidate facts (audit preserved) |
| `mark_stale` / `unmark_stale` / `forget` | Toggle recall eligibility or soft-delete |
| `inspect` / `memory_stats` / `recall_stats` | Browse and inspect |
| `upkeep` | Run projects / verify / consolidate / briefs now (`dry_run` available) |
| `audit_memories` | Read-only duplicate / stale / contradiction suggestions |
| `doctor` | Health check incl. volume metrics (read-only; opt-in `repair`) |
| `sync` | Git-backed pull + push of the data directory |
| `import_memories` | Bootstrap from `~/.claude/projects/*/memory/` |

Pass `project` as the agent's working directory (or the repo name). Paths and
git worktrees resolve to the repository's name, and the repo root is recorded
locally so `verify` can check anchors.

Default tool responses are concise text. MCP tools also expose the same envelope
as `structuredContent`, so agent clients don't have to parse JSON out of text.
Pass `format="json"` (or `--json` in the CLI) when you want the envelope inline:

```
recall(query, format="json", with_provenance=True) →
  {status, data: {answer, mode, facts, tier, source_fact_ids, cited_fact_ids, provenance, usage}, warnings, errors, meta}
```

`recall` / `recall_trace` return at most `max_sources` cards (default 10; `limit`
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
| `ENGRAM_LLM_MODEL` | `openai/gpt-6-luna` | LLM for extraction, recall fallback, and upkeep |
| `ENGRAM_LLM_REASONING_EFFORT` | `medium` | Reasoning effort for extraction and upkeep (reasoning models) |
| `ENGRAM_LLM_SERVICE_TIER` | `fast` | OpenAI processing tier: `fast` (lowest latency, ~2x price), `default`, or empty to omit |
| `ENGRAM_RECALL_REASONING_EFFORT` | `low` | Reasoning effort for recall calls (reasoning models) |
| `ENGRAM_MAX_FACTS_PER_AGENT` | `40` | Max facts fed to a recall LLM call |
| `ENGRAM_RETRIEVAL_TIMEOUT` | `15.0` | Recall LLM call timeout (seconds) |
| `ENGRAM_EPHEMERAL_TTL_DAYS` | `45` | Default expiry for time-bound memories |
| `ENGRAM_MAINTENANCE_ENABLED` | `true` | Run background upkeep in the MCP server |
| `ENGRAM_MAINTENANCE_INTERVAL` | `21600` | Seconds between background upkeep runs |
| `ENGRAM_MAINTENANCE_CONCURRENCY` | `4` | Parallel consolidation LLM calls |
| `ENGRAM_REPO_SEARCH_ROOTS` | `~/dev`, `~/code`, `~/src`, `~/projects`, `~` | Where `verify` looks for a project's checkout when none is recorded |
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
- `maintenance_state.jsonl`: upkeep progress (synced, union-merged).
- `projects.local.json`: project → repo root on this machine (not synced).

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
makes no network calls. Upkeep's lock is per machine, so with sync enabled run
background upkeep on one machine (`ENGRAM_MAINTENANCE_ENABLED=false` elsewhere)
to avoid two machines consolidating the same cards.

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

Python 3.11+ · FastMCP · OpenAI SDK · snowballstemmer · pydantic-settings · JSONL storage · MIT-licensed.

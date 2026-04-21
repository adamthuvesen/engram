# Engram

Structured, cross-project memory for coding agents via MCP. No embeddings, no vector DB, just LLM-powered extraction and retrieval.

## Quick Start

```bash
uv sync                                          # install deps
uv run engram                                    # start MCP server
uv run fastmcp dev src/engram/server.py          # dev mode (inspector)
uv run pytest tests/ -v                           # run tests
```

## Architecture

```
server.py      FastMCP entrypoint, tool definitions
store.py       JSONL storage, filtering, candidate review, prefiltering
observer.py    Fact extraction & suggestion queueing (LLM structured output)
retriever.py   Deterministic prefilter → 3 parallel search agents → synthesis
importer.py    Bootstrap from Claude Code memory files
llm.py         litellm wrapper
config.py      pydantic-settings (env prefix: ENGRAM_)
models.py      Fact, Candidate, Category, audit models
```

**Data flow:** natural language → `observer` extracts structured facts → `store` persists as JSONL → `retriever` searches with 3 agents (direct, contextual, temporal) and synthesizes.

## MCP Tools


| Tool                                       | Purpose                                               |
| ------------------------------------------ | ----------------------------------------------------- |
| `remember`                                 | Store memories (extracts facts from natural language) |
| `suggest_memories`                         | Propose candidates for human review                   |
| `list_candidates`                          | Browse pending/reviewed suggestions                   |
| `approve_candidates` / `reject_candidates` | Promote or dismiss candidates                         |
| `recall`                                   | Search with 3 parallel agentic search agents          |
| `recall_context`                           | Recall as answer or compact prompt block              |
| `forget`                                   | Soft-delete a fact                                    |
| `inspect`                                  | Browse stored facts                                   |
| `import_memories`                          | Bootstrap from `~/.claude/projects/*/memory/`         |
| `memory_stats`                             | Counts, storage size, category breakdown              |


## Data

All data lives under `~/.engram/data/`:

- `facts.jsonl` — active + forgotten facts
- `candidates.jsonl` — suggested memories pending review
- `ingestion_log.jsonl` — audit trail

## Config

All settings via `ENGRAM_*` env vars (pydantic-settings). Key knobs:


| Env Var                      | Default               | Description                        |
| ---------------------------- | --------------------- | ---------------------------------- |
| `ENGRAM_LLM_MODEL`           | `openai/gpt-5.4-mini` | LLM for extraction & search agents |
| `ENGRAM_MAX_FACTS_PER_AGENT` | `200`                 | Facts fed to each search agent     |
| `ENGRAM_RETRIEVAL_TIMEOUT`   | `15.0`                | Search agent timeout (seconds)     |
| `ENGRAM_RECALL_PIPELINE`     | `multilens`           | Tier-2 pipeline: `multilens` (single search + synthesis, prompt-cache friendly) or `legacy` (3 parallel agents + synthesis) |
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
- All tools are async; store operations are synchronous (file I/O)
- Facts have: category, content, confidence, timestamps, project scope, supersession chain, source metadata


# Engram

Structured, cross-project memory for coding agents via MCP.

The LLM is the search engine. No embeddings and no vector database.

## How It Works

1. **Store** — natural language → LLM extracts structured facts → persisted as JSONL
2. **Review** — optionally queue suggestions for human approval before promotion
3. **Recall** — deterministic prefilter → 3 parallel search agents (direct, contextual, temporal) → synthesized answer

## Quick Start

```bash
uv sync                                          # install deps
uv run engram                                    # start MCP server
uv run fastmcp dev src/engram/server.py          # dev mode (inspector)
uv run pytest tests/ -v                           # run tests
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

## Configuration

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

`recall_stats` reports LLM usage per recall when the provider exposes it:
total calls, input tokens, cached (prefix-hit) tokens, and the resulting cache
hit ratio. Missing values render as `-`.


## Data

All data lives under `~/.engram/data/`:

- `facts.jsonl` — active + forgotten facts
- `candidates.jsonl` — suggested memories pending review
- `ingestion_log.jsonl` — audit trail

## Tech

Python 3.11+ · FastMCP 2.x · litellm · pydantic-settings · JSONL storage

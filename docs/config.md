# Configuration

All settings come from `ENGRAM_*` env vars (pydantic-settings, `env_prefix =
"ENGRAM_"`). Key knobs:

| Env var                      | Default               | Description                        |
| ---------------------------- | --------------------- | ---------------------------------- |
| `ENGRAM_LLM_MODEL`           | `openai/gpt-6-luna` | LLM for extraction, recall fallback, and upkeep |
| `ENGRAM_LLM_REASONING_EFFORT` | `medium`             | Reasoning effort for extraction and upkeep (reasoning models): `none`, `low`, `medium`, `high`, `xhigh`, or `max` |
| `ENGRAM_LLM_SERVICE_TIER`    | `fast`                | OpenAI processing tier: `fast` (lowest latency, ~2x price), `default`, or empty to omit |
| `ENGRAM_RECALL_REASONING_EFFORT` | `low`             | Reasoning effort for recall calls (reasoning models) |
| `ENGRAM_MAX_FACTS_PER_AGENT` | `40`                  | Max facts fed to a recall LLM call |
| `ENGRAM_RETRIEVAL_TIMEOUT`   | `15.0`                | Recall LLM call timeout (seconds)  |
| `ENGRAM_EPHEMERAL_TTL_DAYS`  | `45`                  | Default expiry for ephemeral memories without one |
| `ENGRAM_MAINTENANCE_ENABLED` | `true`                | Run background upkeep in the MCP server lifespan |
| `ENGRAM_MAINTENANCE_INTERVAL` | `21600`              | Seconds between upkeep runs (tracked across restarts) |
| `ENGRAM_MAINTENANCE_CONCURRENCY` | `4`               | Parallel consolidation LLM calls |
| `ENGRAM_REPO_SEARCH_ROOTS`   | `["~/dev","~/code","~/src","~/projects","~"]` | JSON list of dirs `verify` searches (3 levels deep) for a project's checkout |
| `ENGRAM_DATA_DIR`            | `~/.engram/data`      | Storage directory                  |
| `ENGRAM_SYNC_ENABLED`        | `false`               | Run background auto-sync in the MCP server lifespan. |
| `ENGRAM_SYNC_INTERVAL`       | `300.0`               | Background auto-sync cadence (seconds). |
| `ENGRAM_SYNC_TIMEOUT`        | `30.0`                | Timeout for each underlying `git` invocation. |

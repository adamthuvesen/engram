# Config

All settings come from `ENGRAM_*` env vars (pydantic-settings, `env_prefix =
"ENGRAM_"`). Key knobs:

| Env Var                      | Default               | Description                        |
| ---------------------------- | --------------------- | ---------------------------------- |
| `ENGRAM_LLM_MODEL`           | `openai/gpt-5.4-mini` | LLM for extraction & search agents |
| `ENGRAM_MAX_FACTS_PER_AGENT` | `200`                 | Facts fed to each search agent     |
| `ENGRAM_RETRIEVAL_TIMEOUT`   | `15.0`                | Search agent timeout (seconds)     |
| `ENGRAM_TIER2_MIN_PREFILTER_COUNT` | `11`            | Tier-2 requires at least this many positive-scoring prefilter matches. `0` disables the small-corpus cap. |
| `ENGRAM_TIER2_MODE`          | `single`              | Tier-2 strategy: `single` or `multilens` |
| `ENGRAM_DATA_DIR`            | `~/.engram/data`      | Storage directory                  |
| `ENGRAM_SYNC_ENABLED`        | `false`               | Run background auto-sync inside the MCP server lifespan. |
| `ENGRAM_SYNC_INTERVAL`       | `300.0`               | Background auto-sync cadence (seconds). |
| `ENGRAM_SYNC_TIMEOUT`        | `30.0`                | Timeout for each underlying `git` invocation. |

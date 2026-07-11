# Configuration

All settings come from `ENGRAM_*` env vars (pydantic-settings, `env_prefix =
"ENGRAM_"`). Key knobs:

| Env var                      | Default               | Description                        |
| ---------------------------- | --------------------- | ---------------------------------- |
| `ENGRAM_LLM_MODEL`           | `openai/gpt-5.4-mini` | LLM for extraction, dedup, and recall |
| `ENGRAM_MAX_FACTS_PER_AGENT` | `200`                 | Max facts fed to the recall LLM call |
| `ENGRAM_RETRIEVAL_TIMEOUT`   | `15.0`                | Recall LLM call timeout (seconds)  |
| `ENGRAM_TIER2_MIN_PREFILTER_COUNT` | `11`            | Minimum positive-scoring prefilter matches before tier-2. `0` disables the small-corpus cap. |
| `ENGRAM_DATA_DIR`            | `~/.engram/data`      | Storage directory                  |
| `ENGRAM_SYNC_ENABLED`        | `false`               | Run background auto-sync in the MCP server lifespan. |
| `ENGRAM_SYNC_INTERVAL`       | `300.0`               | Background auto-sync cadence (seconds). |
| `ENGRAM_SYNC_TIMEOUT`        | `30.0`                | Timeout for each underlying `git` invocation. |

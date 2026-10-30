# Configuration

Settings use `ENGRAM_*` environment variables. JSON arrays configure list values.

Generated from `Settings` descriptions. Run `just reference` after changing a field.

| Environment variable | Default | Description |
| --- | --- | --- |
| `ENGRAM_DATA_DIR` | `~/.engram/data` | Storage directory. |
| `ENGRAM_LLM_MODEL` | `openai/gpt-6-luna` | OpenAI model for extraction, recall fallback, and upkeep. |
| `ENGRAM_LLM_REASONING_EFFORT` | `medium` | Reasoning effort for extraction and upkeep. |
| `ENGRAM_LLM_TEMPERATURE` | `0.0` | Temperature for models without reasoning effort. |
| `ENGRAM_LLM_SERVICE_TIER` | `fast` | OpenAI processing tier, or empty to omit. |
| `ENGRAM_RECALL_REASONING_EFFORT` | `low` | Reasoning effort for recall calls. |
| `ENGRAM_MAX_FACTS_PER_AGENT` | `40` | Maximum facts supplied to a recall LLM call. |
| `ENGRAM_RETRIEVAL_TIMEOUT` | `15.0` | Recall provider timeout in seconds. |
| `ENGRAM_EPHEMERAL_TTL_DAYS` | `45` | Default expiry in days for ephemeral memories. |
| `ENGRAM_MAINTENANCE_ENABLED` | `true` | Run background memory upkeep in the MCP server. |
| `ENGRAM_MAINTENANCE_INTERVAL` | `21600.0` | Seconds between background upkeep runs. |
| `ENGRAM_MAINTENANCE_CONCURRENCY` | `4` | Maximum concurrent upkeep provider calls. |
| `ENGRAM_REPO_SEARCH_ROOTS` | `["~/dev", "~/code", "~/src", "~/projects", "~"]` | Directories searched for project repositories, up to three levels deep. |
| `ENGRAM_CLAUDE_PROJECTS_DIR` | `~/.claude/projects` | Claude Code projects directory for memory imports. |
| `ENGRAM_SYNC_ENABLED` | `false` | Run background Git sync in the MCP server. |
| `ENGRAM_SYNC_INTERVAL` | `300.0` | Seconds between background sync runs. |
| `ENGRAM_SYNC_TIMEOUT` | `30.0` | Timeout in seconds for each sync Git invocation. |
| `ENGRAM_LOG_LEVEL` | `INFO` | Application logging level. |

Positive counts and intervals reject zero, negative values, infinity, and NaN where applicable.
The field types and validation constraints live in `src/engram/core/config.py`.

`OPENAI_API_KEY` supplies provider credentials. Runtime reads can use an injected disabled provider.
`ENGRAM_EVAL_PROVIDER=1` explicitly enables live evals. The default verification profile never enables it.

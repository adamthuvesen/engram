# Data

All data lives under `~/.engram/data/` by default:

- `facts.jsonl`: append-only event log. The first line is the
  `{"meta":"event-log-v1",...}` sentinel; subsequent lines are typed
  `FactEvent` records (`created`, `edited`, `forgotten`, `restored`, `stale`,
  `unstale`, `superseded`). Current state per `fact_id` comes from replaying
  events in order.
- `candidates.jsonl`: suggested memories pending review.
- `recall_log.jsonl`: recall quality and latency history.
- `transactions.jsonl`: prepared/committed markers for crash-safe candidate
  approval.
- `.engram-sync-state`: last successful sync timestamp and commit counts (only
  exists when `engram sync` has been run).
- `.gitignore` / `.gitattributes`: managed by `engram sync` on first run.
  The gitignore excludes lock and per-machine state files; gitattributes
  configures `merge=union` for the event-log files so parallel appends from
  two machines auto-merge.

## Event-log invariant

Storage is an append-only event log. Mutations (`forget`, `edit_fact`,
`mark_stale`, etc.) append typed `FactEvent` records rather than rewriting the
file. The only paths that rewrite `facts.jsonl` are `purge`, `repair`, and
`compact_event_log`. Repair drops corrupt event records only when the file
starts with a valid event-log sentinel. If that sentinel is missing or invalid,
repair refuses to change the file; restore it from backup or inspect and move it
aside before starting a new store.

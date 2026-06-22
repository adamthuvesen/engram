# Data

All data lives under `~/.engram/data/`:

- `facts.jsonl` — append-only event log. First line is the
  `{"meta":"event-log-v1",...}` sentinel; subsequent lines are typed
  `FactEvent` records (`created`, `edited`, `forgotten`, `restored`, `stale`,
  `unstale`, `superseded`). Current state per `fact_id` is materialized by
  replaying events in order.
- `candidates.jsonl` — suggested memories pending review.
- `ingestion_log.jsonl` — audit trail.
- `recall_log.jsonl` — recall quality / latency observability.
- `transactions.jsonl` — prepared/committed markers for crash-safe candidate
  approval.
- `facts.jsonl.pre-eventlog` — one-shot backup written the first time a legacy
  (rewrite-format) store is migrated to the event log. Safe to delete after
  you've verified the new file looks right.
- `.engram-sync-state` — last successful sync timestamp + commit counts (only
  exists when `engram sync` has been run).
- `.gitignore` / `.gitattributes` — managed by `engram sync` on first run.
  The gitignore excludes lock and per-machine state files; gitattributes
  configures `merge=union` for the event-log files so parallel appends from
  two machines auto-merge.

## Event-log invariant

Storage is an append-only event log. Mutations (`forget`, `edit_fact`,
`mark_stale`, etc.) append typed `FactEvent` records rather than rewriting the
file. The only paths that rewrite `facts.jsonl` are `purge`, `repair`, and
`compact_event_log` (compaction inside `synthesize`).

**Rollback**: stop Engram, replace `facts.jsonl` with
`facts.jsonl.pre-eventlog`, downgrade the package, restart.

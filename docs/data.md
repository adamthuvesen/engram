# Data

All data lives under `~/.engram/data/` by default:

- `facts.jsonl`: append-only event log. The first line is the
  `{"meta":"event-log-v1",...}` sentinel; subsequent lines are typed
  `FactEvent` records (`created`, `edited`, `forgotten`, `restored`, `stale`,
  `unstale`, `superseded`). Current state per `fact_id` comes from replaying
  events in order.
- `candidates.jsonl`: append-only candidate snapshots, including review decisions.
- `recall_log.jsonl`: recall quality and latency history.
- `transactions.jsonl`: prepared/committed markers for crash-safe candidate
  approval. New prepared entries include exact fact events and candidate snapshots.
- `.transaction-intent.json`: durable local intent for a fact-event batch. It holds
  exact event IDs, timestamps, and payloads until the batch is complete. Not synced.
- `.transaction-intent.tmp`: temporary intent file used before atomic replacement.
  Not synced.
- `maintenance_state.jsonl`: upkeep progress (last run, per-project
  consolidation snapshot time, retry seeds, own writes). Append-only, synced
  with `merge=union`; the latest line per key wins.
- `projects.local.json`: project name → repo root on this machine. Written when
  a caller passes a path as `project`; read by `verify`. Not synced.
- `upkeep.lock`: per-machine upkeep lock. Not synced.
- `.engram-sync-state`: last successful sync timestamp and commit counts (only
  exists when `engram sync` has been run).
- `.gitignore` / `.gitattributes`: managed by `engram sync` on first run.
  The gitignore excludes lock and per-machine state files; gitattributes
  configures `merge=union` for the event-log files so parallel appends from
  two machines auto-merge.

## Memory-card fields

Each current fact record is one coherent memory card, not necessarily one
sentence. `memory_key` is its stable semantic identity. `retrieval_hints` hold
likely future queries and vocabulary. `source_group_id` links cards extracted
from the same input. `supersedes` tracks correction lineage, while
`consolidates` lists every source fact joined into a replacement card.

`observed_at` is when a card last took in new information. A card that an
ingest replacement, a consolidation, or `merge_memories` builds from older
cards also records `first_observed_at`, the date of the oldest claim it
carried forward, so a merge never makes an old claim look new. A correction
keeps the value its card already had. Recall, briefs, and the LLM prompts show
such a card's date as `oldest..newest`. Ranking and the audit's age heuristics
still use `observed_at`.

`durability` is `evergreen` (never ages), `durable` (true until contradicted),
or `ephemeral` (in-flight state; gets an `expires_at` of now +
`ENGRAM_EPHEMERAL_TTL_DAYS` when none is given and decays in ranking).
`anchors` lists repo-relative paths or code symbols the card depends on;
`verify` checks them. `suspect_reason` is set when some anchors are missing:
the card stays recallable, flagged "unverified", and ranks lower.

Older records load with empty values for these fields (durability `durable`). New extraction,
correction, editing, and merge paths preserve them. A merge appends one new
card plus supersession events for its sources. It never rewrites the event-log
prefix.

## Event-log invariant

New logs use the canonical `{"meta":"event-log-v1"}` sentinel without a creation timestamp.
Readers preserve creation timestamps from legacy headers.
Readers accept valid duplicate v1 sentinels retained by legacy union merges.
Loading, doctor, and repair use the same decoder. Unknown versions and malformed sentinels remain invalid.
Ordinary appends preserve existing log bytes.

Storage is an append-only event log. Mutations (`forget`, `edit_fact`,
`mark_stale`, etc.) append typed `FactEvent` records rather than rewriting the
file. The only paths that rewrite `facts.jsonl` are `purge`, `repair`, and
`compact_event_log`. Repair drops corrupt event records only when the file
starts with a valid event-log sentinel. If that sentinel is missing or invalid,
repair refuses to change the file; restore it from backup or inspect and move it
aside before starting a new store.


## Snapshot and recovery contracts

`FactStore.load_facts()` returns independent snapshots, including nested collections.
Changing a returned fact cannot change the cache or another reader's result.
Materialized facts expose `lifecycle` as `active`, `forgotten`, or `superseded`.
These fields do not change persisted v1 fact rows.
Inactive public snapshots retain the confidence-zero compatibility projection.
Replay preserves the stored confidence, so restoration can recover its original value.
Only forgotten facts can be restored. Superseded facts remain inactive.
`core.fact_state.is_fact_eligible` supplies the shared active-memory policy.

One per-directory lock serializes snapshots, writes, recovery, and sync.
`ChangeSet.expected` checks every recorded source revision under that lock.
`ChangeSet.require_eligible=True` also checks each source against the shared eligibility policy at one timestamp.
The default remains false so explicit stale restoration can retain its existing behavior.
A fact batch writes and fsyncs its local intent before appending events.
Recovery appends missing event IDs with their original timestamps and payloads.
Repeated recovery preserves the same events.
An interrupted JSON row remains an integrity error until explicit repair removes it.
Recovery retains the intent while that error remains unresolved.
Ordinary snapshots report the corrupt file and line through `StoreIntegrityError`.
Corrupt recall-log rows make `recall_stats` fail with `storage_error`.
Doctor can inspect valid recall rows and report the corrupt line numbers.
Diagnostic readers can inspect valid rows without recovering transactions or creating files.

Constructing `FactStore` does not recover transactions.
Runtime startup must call `recover_transactions()` before serving reads or writes.
`FactStore.open_readonly()` creates no directory or lock files and never recovers transactions.
Prepared approval journals block ordinary snapshots until explicit recovery completes them.
Unknown transaction types remain unresolved and raise an error.
Legacy approval journals retain their recorded fact IDs during recovery.

## Candidate review history

Normal candidate updates append complete snapshots in the existing row format.
Readers materialize one snapshot per candidate ID before applying status filters.
The latest `updated_at` wins. Equal timestamps use terminal status and a canonical row order.
Conflicting approval and rejection decisions for one revision produce `conflicted` status.
Conflicted candidates remain inspectable and cannot be approved as pending candidates.
Rejection does not forget an already promoted fact.

New approvals derive fact identity from the candidate ID and promotable content revision.
Two offline approvals of the same revision therefore produce one materialized fact.
Approval edits and pending status checks share one storage transaction.
An unavailable edit target aborts the complete approval request before any edit or promotion.
Candidates that replace the same source in one request remain pending with explicit conflicts.
Independent candidates in that request can still be approved.
The prepared journal holds the exact edited candidate snapshot and promoted fact.
Approval results report only the candidate IDs and edits that transaction applied.
New replacement proposals record all source revisions in `replaces_expected`.
Approval rejects incomplete maps and sources that changed or became ineligible, including through expiry alone.
Legacy proposals with no map retain their earlier approval behavior and deterministic promoted identities.
Pending proposal deduplication runs under the same lock and reads the complete pending queue.
Rejected content can be proposed again.
Upgrade all writers before concurrent review. Older writers can still rewrite candidate history.

Purge and compaction refuse stores with a Git repository.
Synced history rewrites need a coordinated offline procedure across every replica.

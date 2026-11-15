# Sync across machines

Engram syncs `~/.engram/data` through a private git repo. The data stays in your
repo. Engram does not need a hosted service.

```bash
# One-time setup on machine A
cd ~/.engram/data
git init -b main
git remote add origin git@github.com:you/your-engram-data.git  # private!
engram sync       # commits supported data and managed setup, then syncs

# On machine B
git clone git@github.com:you/your-engram-data.git ~/.engram/data
engram sync       # pulls A's state; subsequent syncs are pull + push
```

The first sync auto-commits a managed `.gitignore` and `.gitattributes`.
Each sync commits an allowlisted snapshot of facts, candidates, recall logs,
transactions, and maintenance state before integrating remote changes.
Unrelated staged files stay outside that commit.
A successful remote inspection must confirm an empty remote before the first publication skips fetching.
A missing branch on a nonempty remote and failed authentication remain errors.
Commit counts fail explicitly when Git cannot produce a valid count.
Run `engram sync` whenever you want to push or pull. Set
`ENGRAM_SYNC_ENABLED=true` to have the MCP server run sync automatically on
`ENGRAM_SYNC_INTERVAL` (default 300s) and once on shutdown.

Sync requires `git` on PATH. The first run of `engram sync` writes managed
`.gitignore` and `.gitattributes` to the data dir; subsequent runs are
idempotent.

**Conflict model**: appends from two machines auto-merge through `merge=union`.
Same-fact concurrent edits resolve by event timestamp, then event ID.
Both events remain in the log for audit.
Candidate readers materialize one snapshot per candidate before filtering by status.
Same-revision offline approvals share a deterministic promoted fact ID.
Conflicting approval and rejection decisions remain visible as `conflicted` candidates. `engram doctor` surfaces sync
state under the `counts.sync` group.

`projects.local.json` (repo roots) and `upkeep.lock` stay per machine.
`maintenance_state.jsonl` syncs with `merge=union`. The upkeep lock does not span
machines, so run background upkeep on one machine and set
`ENGRAM_MAINTENANCE_ENABLED=false` on the others; otherwise both may
consolidate the same new cards before they sync.


Manual, background, and shutdown sync share the store's per-directory transaction lock.
Sync recovers local transactions before committing the data snapshot.
It also recovers merged prepared transactions before publishing the integrated snapshot.
Readers and writers wait until integration completes.
Cancelling an async sync waits for its worker thread to stop before returning.
The next sync therefore cannot overlap a cancelled worker.

Local transaction intent files stay outside Git.
An unresolved intent or corrupt row prevents publication until recovery or explicit repair succeeds.
Validation covers facts, candidates, transactions, recall logs, and maintenance state before local commits and after integration.
Replacement approval checks recorded source revisions within one local store.
Concurrent offline approvals based on divergent source snapshots still require review after integration.
Git union merges do not provide distributed serializability.
Purge and compaction refuse Git-backed stores because union merges can resurrect rewritten history.
Coordinate any history rewrite across all replicas while they are offline.

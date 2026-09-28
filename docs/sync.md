# Sync across machines

Engram syncs `~/.engram/data` through a private git repo. The data stays in your
repo. Engram does not need a hosted service.

```bash
# One-time setup on machine A
cd ~/.engram/data
git init -b main
git remote add origin git@github.com:you/your-engram-data.git  # private!
engram sync       # writes managed .gitignore + .gitattributes, pushes setup

# On machine B
git clone git@github.com:you/your-engram-data.git ~/.engram/data
engram sync       # pulls A's state; subsequent syncs are pull + push
```

The first sync auto-commits a managed `.gitignore` and `.gitattributes`.
After that, run `engram sync` whenever you want to push or pull. Set
`ENGRAM_SYNC_ENABLED=true` to have the MCP server run sync automatically on
`ENGRAM_SYNC_INTERVAL` (default 300s) and once on shutdown.

Sync requires `git` on PATH. The first run of `engram sync` writes managed
`.gitignore` and `.gitattributes` to the data dir; subsequent runs are
idempotent.

**Conflict model**: appends from two machines auto-merge thanks to
`merge=union`. Same-fact concurrent edits resolve by event timestamp on read,
with both events kept in the log for audit. `engram doctor` surfaces sync
state under the `counts.sync` group.

`projects.local.json` (repo roots) and `upkeep.lock` stay per machine.
`maintenance_state.jsonl` syncs with `merge=union`. The upkeep lock does not span
machines, so run background upkeep on one machine and set
`ENGRAM_MAINTENANCE_ENABLED=false` on the others; otherwise both may
consolidate the same new cards before they sync.

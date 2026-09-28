# MCP Tools

The MCP tool names below are a stable public surface. Renaming or removing one
is a breaking change for every agent wired to Engram. Treat them like an API.

| Tool                                       | Purpose                                               |
| ------------------------------------------ | ----------------------------------------------------- |
| `remember`                                 | Save plain language: extract + reconcile in one LLM call |
| `suggest_memories`                         | Same, queued as candidates for review                  |
| `list_candidates`                          | Browse pending/reviewed suggestions                   |
| `approve_candidates` / `reject_candidates` | Promote or dismiss candidates                         |
| `recall`                                   | Dated cards (`mode="cards"`) or `mode="answer"` (`max_sources`; MCP `limit` alias) |
| `recall_context`                           | `brief`, `cards` (`prompt` alias), or `answer`        |
| `recall_trace`                             | Recall + bounded prompt/output excerpts for debugging (`limit` alias) |
| `recall_stats`                             | Per-recall LLM usage and cache-hit summary            |
| `forget` / `edit_fact`                     | Soft-delete or edit a fact in place                   |
| `correct_memory` / `merge_memories`        | Agent-first correction and merge primitives           |
| `mark_stale` / `unmark_stale`              | Toggle a fact's recall eligibility                    |
| `inspect`                                  | Browse stored facts                                   |
| `import_memories`                          | Bootstrap from `~/.claude/projects/*/memory/`         |
| `memory_stats`                             | Counts, storage size, category breakdown              |
| `upkeep`                                   | Run projects / verify / consolidate / briefs now      |
| `audit_memories`                           | Read-only duplicate / stale / contradiction suggestions |
| `purge` / `rename_project`                 | Permanently drop forgotten/expired or rename a scope  |
| `doctor`                                   | Read-only health diagnostics (with opt-in `repair`)   |
| `sync`                                     | Git-backed pull + push of the data directory          |

Every tool that takes `project` accepts a repo name or a working-directory
path; paths and git worktrees resolve to the repository's name.

`recall` defaults to cards: ranked, dated card lines with no LLM call. When no
card clears the relevance bar and a key is configured, one small LLM call picks
cards from the top 30 candidates. `mode="answer"` synthesizes an answer with one
call and falls back to cards (with a `provider_unavailable` warning) if the
provider fails. Warnings: `suspect_fact` when a delivered card has missing
anchors, `conflicting_facts` when two delivered cards share a `memory_key`.

`recall_stats` summarizes LLM usage pulled from the recall log: total LLM
calls, input tokens, cached (prefix-hit) input tokens, and the resulting cache
hit ratio. Recall logs stamp each record with its selector version (currently
`"v4"`), mode, and the delivered fact IDs.

`upkeep` applies its changes (they are ordinary append-only events) unless
`dry_run=True`. Steps run in order `projects`, `verify`, `consolidate`,
`briefs`; `consolidate` and `briefs` need an LLM key and are skipped without
one.

`edit_fact`, `correct_memory`, and `merge_memories` accept `memory_key` and
`retrieval_hints` where relevant. `merge_memories` returns `consolidates`, the
complete source-ID list retained on the new card for provenance.

The MCP process uses local stdio, or stateless streamable HTTP on loopback via
`engram serve --transport http` (default `http://127.0.0.1:7422/mcp`) so many
clients can share one long-lived daemon. It exists for typed discovery and
persistent agent integration, not as a remote network service. The `engram` CLI exposes the same
operations for direct inspection, scripting, and batch maintenance.

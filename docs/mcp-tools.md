# MCP Tools

The MCP tool names below are a stable public surface. Renaming or removing one
is a breaking change for every agent wired to Engram. Treat them like an API.

| Tool                                       | Purpose                                               |
| ------------------------------------------ | ----------------------------------------------------- |
| `remember`                                 | Store memories (extracts facts from natural language) |
| `suggest_memories`                         | Propose candidates for human review                   |
| `list_candidates`                          | Browse pending/reviewed suggestions                   |
| `approve_candidates` / `reject_candidates` | Promote or dismiss candidates                         |
| `recall`                                   | Tiered search (`max_sources`; MCP `limit` alias) |
| `recall_context`                           | Recall as answer or compact prompt block              |
| `recall_trace`                             | Recall + bounded prompt/output excerpts for debugging (`limit` alias) |
| `recall_stats`                             | Per-recall LLM usage and cache-hit summary            |
| `forget` / `edit_fact`                     | Soft-delete or edit a fact in place                   |
| `correct_memory` / `merge_memories`        | Agent-first correction and merge primitives           |
| `mark_stale` / `unmark_stale`              | Toggle a fact's recall eligibility                    |
| `inspect`                                  | Browse stored facts                                   |
| `import_memories`                          | Bootstrap from `~/.claude/projects/*/memory/`         |
| `memory_stats`                             | Counts, storage size, category breakdown              |
| `audit_memories`                           | Read-only duplicate / stale / contradiction suggestions |
| `purge` / `rename_project`                 | Permanently drop forgotten/expired or rename a scope  |
| `doctor`                                   | Read-only health diagnostics (with opt-in `repair`)   |
| `sync`                                     | Git-backed pull + push of the data directory          |

`recall_stats` summarizes LLM usage pulled from the recall log: total LLM
calls, input tokens, cached (prefix-hit) input tokens, and the resulting cache
hit ratio. Providers that don't report usage leave those fields blank. The
stats view renders `-` for any column with no data. Recall logs keep
`selector_version="v2"` for continuity with existing `recall_stats` output.

# MCP tools and CLI commands

Tool names form a stable public API. The CLI uses the corresponding hyphenated names.

Generated from MCP registrations and CLI handlers. Run `just reference` after changing either surface.

| MCP tool | CLI command | Description |
| --- | --- | --- |
| `approve_candidates` | `approve-candidates` | Promote reviewed candidates into active memory. |
| `audit_memories` | `audit-memories` | Suggest duplicate, stale, and contradictory memory cleanup. |
| `correct_memory` | `correct-memory` | Replace ``fact_id`` with a new active fact that supersedes it. |
| `doctor` | `doctor` | Run a structured health check on the memory store. |
| `edit_fact` | `edit-fact` | Edit a fact in place without losing its ID or supersession chain. |
| `forget` | `forget` | Mark a fact as forgotten (soft delete). |
| `import_memories` | `import-memories` | Import existing memories from Claude Code's per-project memory files. |
| `inspect` | `inspect` | Browse stored facts for transparency and debugging. |
| `list_candidates` | `list-candidates` | Browse queued memory candidates for review. |
| `mark_stale` | `mark-stale` | Exclude a fact from active recall while preserving it for inspection. |
| `memory_stats` | `memory-stats` | Show memory system statistics: fact counts, storage size, category breakdown. |
| `merge_memories` | `merge-memories` | Consolidate two or more active facts into one new fact. |
| `purge` | `purge` | Permanently remove forgotten and expired facts from storage. |
| `recall` | `recall` | Find memories relevant to ``query``. |
| `recall_context` | `recall-context` | Context for the current task. |
| `recall_stats` | `recall-stats` | Show recall quality and performance statistics from the recall log. |
| `recall_trace` | `recall-trace` | Run recall and return a structured trace for debugging. |
| `reject_candidates` | `reject-candidates` | Reject proposed candidates without storing them as active memory. |
| `remember` | `remember` | Save durable knowledge from plain language. |
| `rename_project` | `rename-project` | Bulk-rename facts and candidates from one project to another. |
| `suggest_memories` | `suggest-memories` | Propose memories for review without storing them as active facts. |
| `sync` | `sync` | Git-backed sync of the data directory against its configured remote. |
| `unmark_stale` | `unmark-stale` | Reverse a previous ``mark_stale`` call so the fact is recall-eligible. |
| `upkeep` | `upkeep` | Run memory upkeep now (it also runs in the background). |

CLI aliases: `trace` → `recall-trace`, `correct` → `correct-memory`, `merge` → `merge-memories`, `stale` → `mark-stale`, `unstale` → `unmark-stale`.

`engram serve` owns the MCP process. It has no corresponding tool.

Every tool validates input before mutation. MCP returns the same response envelope as the CLI JSON output.
The envelope contains `status`, `data`, `warnings`, `errors`, and `meta`.

Recall uses lexical cards by default. Weak-hit selection and answer mode can call the provider.
Answer mode caps its prompt evidence before generation. The renderer creates citations from validated fact references.
`max_sources` caps cards and answer evidence. The MCP `limit` alias remains supported.

`RecallResult` records candidate, selected, delivered, and cited evidence. Legacy `cited_fact_ids` also contains delivered card IDs.
Usage reports attempts, completions, failures, unknown usage, and token counters. Legacy `llm_calls` counts successful recall operations.

Provider failures return cards with `provider_unavailable`. Delivered suspect or conflicting facts carry explicit warnings.
`recall_trace` adds bounded prompt excerpts and error traces. `recall_stats` reads recorded usage.

`doctor` reads storage without recovery writes. Repair flags opt into the corresponding repair operation.
Review pending candidates with `list-candidates`, then approve or reject explicit IDs.

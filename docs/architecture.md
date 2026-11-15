# Architecture

CLI and MCP adapters call shared operations. One `Runtime` owns the store facade, provider client, settings scope, and background tasks.

```text
cli.py / server.py
        ↓
operations/ ← runtime.py
        ↓
extraction / recall / maintenance
        ↓
storage public APIs + core contracts + llm provider protocol
```

## Ownership and entrypoints

`runtime.py` binds settings and the completion client for each operation.
It recovers the store before serving ordinary snapshots. Shutdown waits for owned tasks and closes the provider connection pool.
Separate runtimes can run in one process without sharing settings or provider clients.

`server.py` registers stable MCP tools with FastMCP. It validates rendering options before dispatch.
`cli.py` registers CLI handlers and aliases. Both surfaces render the same `OperationResult` envelope.
`operations` exports the operations both surfaces call. `operations.results` owns `OperationResult`, exit codes, and shared validation failures. `operations.stores` resolves the store and project scope.
The domain modules are `memories`, `candidates`, `recall`, `facts`, and `maintenance`.
`core.operation_models` contains typed operation payloads. `core.interfaces` contains JSON values, errors, warnings, and limits.

## Domain and persistence

`core.models` defines persisted facts, events, candidate snapshots, recall records, and materialized lifecycle state.
Legacy event categories decode through an explicit compatibility path. Provider proposals use the current strict schema.
`core.fact_state.is_fact_eligible` supplies the shared active-memory policy.
`core.projects` canonicalizes project paths and records checkout locations.

`core.fact_format` renders facts as dated card lines.
`storage.store` exposes sync operations. `storage.async_store` is the async facade over them.
`storage.changes` defines `ChangeSet` and the results that the store returns.
`storage.transactions` owns local exclusion, durable intent, recovery, and event-log writes.
`storage.candidates` materializes append-only review history and plans candidate approval transactions.
`storage.compaction` collapses the event log to its minimal events.
`storage.jsonl` provides validated JSONL reading.
`storage.search` owns BM25 scoring and coverage. `storage.sync` integrates Git snapshots under the store lock.

`ChangeSet.expected` records revisions for every read dependency, including facts that the batch does not mutate.
With `require_eligible`, storage also checks each dependency against the active-memory policy under the same lock.

Fact batches first persist local intent with exact event IDs, payloads, and timestamps.
Recovery appends only missing events. Candidate approvals journal exact promoted identities and candidate decisions.
Ordinary snapshots refuse unresolved intent and corrupt rows. Diagnostic readers can inspect valid rows without recovery writes.
These guarantees apply to one data directory. Git integration does not provide distributed serializable transactions.
See [data](data.md) and [sync](sync.md) for migration constraints and crash behavior.

## Extraction

`extraction.observer` selects neighboring facts, requests one structured extraction response, and validates references before applying changes.
Replacement, retirement, and duplicate decisions require unchanged, eligible dependencies. This validation also applies before queuing proposals.
Ingest retries once with fresh neighbors, then reports conflicting decisions as unresolved.
It preserves independent valid proposals when another provider decision is unresolved.
New queued replacements record source revisions for storage to validate again during approval.
`extraction.importer` imports reviewed source files through the same operation contracts.
`core.structured_outputs` owns strict provider schemas. Invalid provider data must remain visible as a failure or unresolved decision.

## Recall

`recall.retriever.recall_result` returns the authoritative `RecallResult`.
It contains candidate, selected, delivered, and cited evidence with unchanged lexical scores and coverage.
The tuple-returning `recall_with_provenance` remains a compatibility adapter.

Lexical cards require both the coverage floor and the relative-score cutoff.
Weak hits can use one provider selection call. Answer mode caps the provider evidence before generating structured claims.
The renderer validates fact references and generates citations. It preserves ordinary hashes, dates, and numbers in claim text.
Provider failures return bounded cards with an explicit warning and error trace when requested.

`core.provenance` derives source metadata and traces from the same evidence.
Legacy `cited_fact_ids` includes delivered cards. Internal `cited` evidence tracks answer references.
Usage distinguishes attempts, completions, failures, and unknown counters. Legacy `llm_calls` counts successful recall operations.

## Provider and upkeep

`llm.CompletionClient` is the typed injectable provider dependency.
`OpenAICompletionClient` creates the SDK connection lazily and exposes `aclose`.
Runtime binding shares one connection pool across extraction, recall, and upkeep.
Standalone compatibility helpers close their temporary client after each call.
Structured model calls use the installed SDK parser.

`maintenance.upkeep` owns step ordering and per-machine upkeep exclusion.
Anchor verification confines file checks to checkout roots and keeps missing evidence distinct from unreadable evidence.
Consolidation validates every cluster input before mutation, including unchanged cards.
Changed or ineligible inputs abort the cluster. Upkeep reports conflicts and schedules bounded retries.
Project briefs record the exact source revisions used to create them.
Recall invalidates a brief when an eligible source changes or disappears.

## Evaluation and checks

The top-level `evals` package contains typed runners, unchanged labeled datasets, fixture providers, and report provenance.
The old test-script paths remain thin entrypoints. Default runners inject disabled or fixture providers without changing retriever globals.
Reports include source digests and explicit live status.

The top-level `tools` package holds the development lanes. Neither `evals` nor `tools` ships in the wheel.
`tools.verify` defines the commands behind `just check`, `just slow`, and CI.
`tools.boundaries` checks private storage imports and sync-store bypasses in async application code.
`tools.reference` generates Settings and command references from source registrations.
`tools.smoke` verifies stdio, loopback HTTP, errors, and restart persistence using temporary data.
The package profile installs locked dependencies and the wheel outside the checkout, then runs `tools.smoke` from the checkout against that installed package.

`pyproject.toml` runs basedpyright in strict mode on `src`, `evals`, `tools`, and `tests`. The verify skill has its own `pyrightconfig.json`, and `check` runs it as a second project because basedpyright skips dot folders. The type check must report zero errors, with no baseline file.
Default pytest cannot run Git transport fixtures without the explicit `--run-sync-transport` option.
Live provider quality requires a separate explicit run. Offline scores establish only their labeled fixture contracts.

# Agentic development implementation and verification

The eight-phase plan is complete locally at implementation revision `09c8c03`.
Both tested Python versions passed the full offline verification profile.
The result is `passed_with_limits`, because live provider and Git transport integration remain unverified.

## Scope and evidence

- Date: 2026-11-04.
- Branch: `adam/refactor-agentic-development`, based on `149cee2`.
- Worktree: `/Users/adamthuvesen/.codex/worktrees/engram-agentic/engram`.
- Original checkout: `/Users/adamthuvesen/dev/tools/engram`, clean at `149cee2` when checked.
- Implementation: `09c8c03`, `refactor: make memory operations transactional and verifiable`.

The approved review context included commits `75750fa`, `1f21bb3`, `33da069`, and PR #26.
The claims below concern implemented behavior and current local evidence.
Verification ran against the implementation source subsequently committed as `09c8c03`.
This report is a separate documentation artifact.

## 1. Domain, lifecycle, and isolation

[Core models](../../src/engram/core/models.py) own fact identity, typed events, and materialized lifecycle state.
[Fact eligibility](../../src/engram/core/fact_state.py) owns the shared active-memory policy.
Forgotten and superseded facts remain distinct. Restore reverses forgetting and preserves the original stored confidence.
Returned snapshots copy nested values, so a caller cannot change another reader's state or the storage cache.
Persisted v1 facts retain their format. Explicit legacy decoding preserves supported historical event shapes.
Current mutations and provider proposals use validated models before storage changes.
[Storage contract tests](../../tests/test_storage_transactions.py) cover isolation, restore rules, validation, and concurrent reads.

## 2. Transactions and recovery

[Transactions](../../src/engram/storage/transactions.py) own per-directory exclusion and durable event intent.
[The store](../../src/engram/storage/store.py) validates complete mutation batches before creating intent.
Intent records retain exact event IDs, timestamps, and payloads. Recovery appends only missing events.
Repeated recovery does not invent another mutation. Unknown transaction types remain unresolved.
Ordinary snapshots reject unresolved intent and corrupt rows. Read-only diagnostics can inspect valid rows without recovery writes.
Legacy approval journals preserve their recorded promoted identities during recovery.
Crash, partial-tail, recovery, and reader-exclusion cases have local behavioral tests.
These contracts protect one data directory. They do not establish distributed serializable transactions.

## 3. Candidates and sync

[Candidate snapshots](../../src/engram/storage/candidates.py) preserve append-only review history.
Readers materialize candidate state before filtering. Pending deduplication reads the complete queue under the shared lock.
Same-revision approvals use deterministic promoted IDs. Conflicting approval and rejection remain visible as `conflicted`.
Rejection does not forget a previously promoted fact.
[Sync](../../src/engram/storage/sync.py) commits allowlisted data and managed setup files before integration.
Unrelated staged files remain outside that snapshot. Integration and all synced-log writers share the directory lock.
Async cancellation waits for the sync worker to stop before returning.
Purge and compaction refuse Git-backed stores because union merging can resurrect rewritten history.
Upgrade every writer before concurrent review. Older writers can still rewrite candidate history.
See the [data](../data.md) and [sync](../sync.md) contracts for migration and operational constraints.

## 4. Runtime and typed operations

[Runtime](../../src/engram/runtime.py) owns settings, lazy store construction, recovery, provider connections, and background tasks.
Runtime scopes isolate injected settings and providers within one process.
[Typed operation payloads](../../src/engram/core/operation_models.py) connect shared operations to CLI and MCP envelopes.
Adapters validate rendering options before dispatch and classify failures by their actual source.
Partial ingest and review outcomes remain visible. Invalid input cannot silently perform a subset of an intended edit.
A CLI invocation does not acquire background shutdown-sync ownership merely because sync is enabled in configuration.
Provider closure still occurs when final server sync fails.
[Runtime tests](../../tests/test_runtime.py) cover ownership, lazy I/O, failure envelopes, and CLI output behavior.

## 5. Recall evidence and metrics

[RecallResult](../../src/engram/core/provenance.py) records candidate, selected, delivered, and cited evidence.
[Recall](../../src/engram/recall/retriever.py) derives output metadata from those artifacts.
Answer evidence is capped before generation by both `max_sources` and the configured fact limit.
Structured claims carry validated references. The renderer generates citations and preserves ordinary hashes, dates, and numbers.
Fabricated references and malformed provider output produce explicit failures and bounded card fallback.
Selection retains reported token usage. Attempts, completions, failures, and unknown usage have separate counters.
Legacy `llm_calls` still counts successful recall operations. Legacy card `cited_fact_ids` and the tuple adapter remain supported.
Selector version `v4`, lexical scores, coverage thresholds, and relative cutoffs remain unchanged.
[Recall tests](../../tests/test_retriever.py) cover caps, source consistency, selector IDs, malformed output, and failure telemetry.

## 6. Maintenance and extraction

[Extraction](../../src/engram/extraction/observer.py) keeps independent valid proposals when another decision remains unresolved.
[Imports](../../src/engram/extraction/importer.py) reject malformed frontmatter and expose per-file failures.
[Anchor verification](../../src/engram/maintenance/verify.py) confines paths to the checkout and distinguishes missing evidence from unavailable evidence.
Unavailable Git or file evidence cannot justify retiring a memory.
[Consolidation](../../src/engram/maintenance/consolidate.py) checks source revisions before mutation and retains unresolved work for retry.
[Briefs](../../src/engram/maintenance/briefs.py) record exact source revisions and reject changed or missing source membership.
[Upkeep](../../src/engram/maintenance/upkeep.py) keeps a nonzero recurring wait after skipped runs.
[Doctor](../../src/engram/maintenance/doctor.py) shares supported legacy decoding and preserves read-only diagnostic behavior.
Corrupt recall-log rows make `recall_stats` fail with `storage_error`, instead of presenting partial history as complete.
[Maintenance tests](../../tests/test_maintenance_correctness.py) cover these evidence, revision, locking, and diagnostic boundaries.

## 7. Tooling, evals, dashboard, and documentation

[Typed evaluation modules](../../src/engram/evaluation) own reusable execution. Existing `tests/run_*evals.py` paths remain thin entrypoints.
Fixture providers replace global retriever monkeypatching and never discover provider keys.
All five relocated JSON datasets are byte-identical to their original files.
Score formulas, thresholds, labels, and query populations remain unchanged. This work makes no measured improvement claim.
Reports record dataset digests, selector version, execution status, and loaded-source revision when available.
Caller-supplied outputs have unknown live provenance. An installed wheel cannot borrow an unrelated host repository's revision.

[Dashboard](../../src/engram/dashboard) uses the public lifecycle policy.
It counts superseded facts separately, refreshes on expiry without file changes, and queues only successful forget operations for undo.
The required `textual-plotext` dependency supplies the timeline widget.

[Verification](../../src/engram/tooling/verify.py) provides `just check`, `just verify`, and the shared CI entrypoint.
Hooks use locked Ruff. Builds use locked Hatchling without build isolation.
[Generated references](../../src/engram/tooling/reference.py) derive settings and command inventories from source contracts.
The [architecture guide](../architecture.md) and short [agent instructions](../../AGENTS.md) identify ownership and enforced boundaries.

## 8. Independent review and final reassessment

Independent storage, application, and tooling reviews finished with zero unresolved findings after local reproduction and fixes.
The reviews closed these demonstrated gaps:

| Review finding | Implemented behavior and evidence |
| --- | --- |
| Synced-log writers could escape integration exclusion | Recall and upkeep-state writers share the directory lock. Tests and a local rebase reproduction verified exclusion. |
| CLI shutdown could perform unintended final sync | Only a runtime that started background sync owns final sync. Runtime tests cover reads, errors, and explicit sync. |
| Skipped upkeep could repeat without a useful wait | Each skipped iteration retains a nonzero wait and opens dependencies only after waiting. |
| Corrupt recall history could appear complete | Ordinary stats fail with `storage_error`. Doctor can inspect valid rows and report corrupt lines. |
| Doctor could disagree with snapshot legacy decoding | Doctor uses the same supported legacy event decoder. |
| Package smoke bypassed production entrypoints | Installed `engram` CLI and `engram serve` now run through stdio and HTTP outside the checkout. |
| Eval provenance could identify the dataset repository | Revision lookup validates the loaded source location. Supplied-output provenance remains explicit. |

The first full attempt also found an obsolete safety assertion that expected the previous permissive behavior.
The assertion now checks the intentional strict contract. Final acceptance used the corrected suite.

## Enforcement choices

| Contract | Primary enforcement | Why behavioral checks or instructions remain |
| --- | --- | --- |
| One active-memory policy and safe snapshots | Shared domain policy and independent snapshots | Tests verify nested mutation and expiry behavior. |
| Mutation consistency and crash recovery | Validated batches, directory locks, durable intent, and revision guards | Types cannot prove filesystem interruption or process ordering. |
| Provider and operation boundaries | Typed protocols, structured models, and validated envelopes | Offline SDK and adapter tests verify refusal, empty output, usage, and error behavior. |
| Private storage and async access boundaries | Focused AST checks and scoped stricter mypy options | These checks enforce named seams, not every possible dynamic call path. |
| Locked tooling and generated references | `uv.lock`, local hooks, source generators, and shared CI commands | Installed-package smokes verify assets and real entrypoints beyond import checks. |
| Sync transport opt-in | Pytest collection requires `--run-sync-transport` | A marker typo cannot enable pushes. The external push safeguard remains authoritative. |
| Cross-machine upgrades | Explicit writer-upgrade constraint | This process cannot prevent an older binary on another machine from rewriting history. |

## Final acceptance

Both runs used isolated environments and the same profile as `just verify`:

```bash
UV_PYTHON=3.13 UV_PROJECT_ENVIRONMENT=/tmp/engram-agentic-verify-313 uv run --locked --extra dev python -m engram.tooling.verify verify
UV_PYTHON=3.14 UV_PROJECT_ENVIRONMENT=/tmp/engram-agentic-verify-314 uv run --locked --extra dev python -m engram.tooling.verify verify
```

| Check | Python 3.13.17 | Python 3.14.9 |
| --- | --- | --- |
| Offline pytest | 556 passed, 13 deselected | 556 passed, 13 deselected |
| Ruff lint and format | Passed, 104 files | Passed, 104 files |
| Mypy | Passed, 66 source files | Passed, 66 source files |
| Lock, references, and import boundaries | Passed | Passed |
| Installed-wheel CLI, stdio, HTTP, and restart | Passed | Passed |
| Installed dashboard CSS, datasets, and headless UI | Passed | Passed |
| Overall profile | `passed_with_limits` | `passed_with_limits` |

Installed-wheel checks verified import origin outside the source checkout and installed dependencies from locked requirements.
Console smokes used temporary data, disabled background work, a synthetic key, and a blocked loopback provider endpoint.
Injected smoke runtimes used a disabled provider.
All four documented no-provider eval scripts passed on Python 3.13.
An sdist build and wheel build from that sdist also passed with locked Hatchling.
Local verification logs were `/tmp/engram-agentic-verify-313.log` and `/tmp/engram-agentic-verify-314.log`.
Those paths are temporary evidence, not durable dependencies of this report.

## External limits

The 13 deselected tests comprise 12 Git transport tests and one live-provider test.
Safe sync unit, local snapshot, union-merge, locking, and cancellation coverage remained in the offline suite.
Earlier legacy fixture attempts reached the push safeguard and were blocked. No push succeeded.
The explicit collection opt-in now prevents default and marker-typo test commands from attempting those pushes.
Run `just verify-sync` only where temporary-local-remote pushes are permitted.

No live OpenAI call, real memory store, hosted transport, or cross-machine deployment was exercised.
Linux CI has not run because this branch has not been pushed.
The local macOS checks establish the implemented fixture and packaging contracts, including both supported Python versions.
They do not establish live model quality, production reliability, cross-machine convergence under real transport, or a performance improvement.

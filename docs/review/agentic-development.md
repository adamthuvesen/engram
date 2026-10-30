# Agentic development implementation and verification

The second local iteration passed complete verification profiles on macOS and Linux, with Python 3.13 and 3.14.
The second-pass acceptance matrix is authoritative. First-pass results remain historical evidence.
All four profiles report `passed_with_limits`. Live provider and actual Git publication remain unverified.

## Scope and evidence

- Date: 2026-11-04.
- Branch: `adam/refactor-agentic-development`, based on `149cee2`.
- Worktree: `/Users/adamthuvesen/.codex/worktrees/engram-agentic/engram`.
- Original checkout: `/Users/adamthuvesen/dev/tools/engram`, clean at `149cee2` when checked.
- First-pass implementation: `09c8c03`, `refactor: make memory operations transactional and verifiable`.
- Second-pass storage: `897ba18`, `fix(storage): guard complete memory decisions and sync snapshots`.
- Second-pass dashboard and tooling: `c7f8a19`, `fix(dx): keep dashboard responsive and verify cold-cache installs`.

The approved review context included commits `75750fa`, `1f21bb3`, `33da069`, and PR #26.
Sections 1–8 and the first-pass acceptance matrix describe the implementation reviewed at `09c8c03`.
The second-iteration section records additional changes and their bounded local evidence.
First-pass verification ran against the source subsequently committed as `09c8c03`.
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

## 8. First-pass independent review and reassessment

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

## Second iteration: review fixes and local guarantees

The second iteration reviewed the implementation beyond the first-pass acceptance matrix.
Demonstrated gaps below came from local reproductions or inspection of the affected implementation.
Added guarantees describe bounded behavioral tests. They do not establish production failures or distributed guarantees.
The independent second-pass reviews closed their reported findings after the corresponding fixes.

| Area and evidence | Finding or guarantee | Current contract |
| --- | --- | --- |
| [Dashboard workers](../../src/engram/dashboard/app.py), [UI tests](../../tests/test_dashboard.py) | Demonstrated gaps: synchronous startup, forget, and undo could block the UI. Failed refreshes could suppress retries. | Storage work runs through serialized async workers. Failed loads retain the last snapshot and remain retryable without a file change. |
| Dashboard undo queue | Added guarantees: partial forget, failed reload, queued undo, expiry, and cancellation interact. | Only successful forgets enter undo. Requested undo survives storage contention. Completed entries leave the queue. Cancellation does not prevent eventual expiry. |
| [Package asset smoke](../../src/engram/tooling/smoke.py), [negative smoke tests](../../tests/test_tooling.py) | Demonstrated verification gap: a headless launch did not prove that persisted data reached the UI. | The smoke opens the facts tab and requires the seeded fact ID and content. A failed data load fails verification. |
| [Observer](../../src/engram/extraction/observer.py), [consolidation](../../src/engram/maintenance/consolidate.py) | Demonstrated gaps: duplicate decisions and unchanged cluster members were missing from source dependency checks. | Duplicate decisions and all consolidation read dependencies carry revision guards. Eligibility is checked under the same lock at one timestamp. |
| [Dependency tests](../../tests/test_observer.py), [storage tests](../../tests/test_storage_transactions.py), [upkeep tests](../../tests/test_upkeep.py) | Added guarantees: expiry can invalidate evidence without changing its revision. Explicit stale restoration still needs its earlier behavior. | `require_eligible=True` protects semantic operations. Its default remains false for compatibility. Retries preserve independent valid proposals and expose unresolved dependencies. |
| [Candidate approval](../../src/engram/storage/store.py), [operation adapter](../../src/engram/operations.py) | Demonstrated gap: edits and approval used separate transactions, so a losing approval could still change candidate content. | One journal owns pending-state checks, edits, promoted facts, and reported edited IDs. An unavailable edit aborts the full request before writing. |
| Candidate replacement groups | Demonstrated gap: competing candidates could replace the same source within one approval request. | Overlapping groups remain pending with explicit conflicts. Independent candidates can proceed. The list-returning compatibility wrapper remains supported. |
| [Candidate revisions](../../src/engram/storage/candidates.py) and storage tests | Added guarantees: new proposals need source-revision checks without changing legacy promotion identity. | `replaces_expected` records all replacement sources. Incomplete, changed, or ineligible dependencies block approval. Legacy `None` proposals retain earlier behavior and hashes. |
| [Event-log scanner](../../src/engram/storage/transactions.py) and storage tests | Demonstrated gap: independent first-writer headers could become false corruption after a union merge. | New headers are canonical. Valid legacy duplicate headers and timestamps remain readable. Loading, doctor, and repair share one scanner. Unknown or malformed headers still fail. |
| Edited-category validation and storage tests | Demonstrated gap: array or object category values could escape normal corruption diagnostics. | The shared decoder reports the corrupt line. Doctor diagnoses it. Explicit repair removes the invalid row and preserves valid facts. |
| `correct_fact` and storage tests | Demonstrated gap: correction rebuilt a subset of source metadata. | Corrections preserve the complete source metadata unless an explicit correction field replaces it. |
| [Sync validation](../../src/engram/storage/sync.py) and storage tests | Demonstrated gaps: empty-remote bootstrap failed, and Git count errors could appear as zero. | Only successful inspection of an empty remote permits bootstrap. Invalid counts fail explicitly. Every allowlisted log is validated before commit and after integration. |
| [Doctor](../../src/engram/maintenance/doctor.py), [doctor tests](../../tests/test_doctor.py) | Demonstrated gap: sync rejected maintenance-state corruption that doctor did not diagnose. | Both use `StateLine` and identify the same corrupt file and line. Doctor does not advertise unsupported automatic repair. |
| [Portable fixtures](../../tests/test_hardening.py) | Added guarantee: project-name fixtures must cover home paths on both operating-system families. | Cases explicitly cover `/Users/alex`, `/home/alex`, and `/root`, including short and hyphenated repository names. |

These tests use fictional data, temporary stores, local Git operations, and injected or blocked providers.
Safe empty-remote inspection uses local Git. Bootstrap publication is intercepted, so it does not prove a successful remote push.
Concurrent offline approvals against divergent source snapshots still require review after integration.
All writers must upgrade before relying on the new candidate dependency contract.

A fresh Linux run exposed a cold-cache package-install metadata failure after its tests completed.
Exporting requirements had caused package installation to require metadata that the lock-based setup had not cached.
The package profile now runs `uv sync --locked --offline --no-dev --no-install-project --python` directly from `uv.lock`.
Only dependency setup overrides `UV_PROJECT_ENVIRONMENT`. The explicit interpreter overrides inherited `UV_PYTHON`.
Isolation tests reject imports from the checkout and preserve the caller environment. Independent review found no remaining tooling findings.
The complete profiles below passed after this fix.

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

## First-pass acceptance

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

## Second-pass acceptance

All four complete profile processes exited zero with `passed_with_limits` against the source committed at `c7f8a19`.
All 115 tracked source, test, CI, and configuration files matched both tested Linux snapshots byte for byte.
Each ran locked lint, format, types, reference generation, boundary checks, the offline suite, and installed-wheel verification.

| Environment | Python | Offline pytest | Complete profile |
| --- | --- | --- | --- |
| macOS | 3.13.17 | 637 passed, 13 deselected | Passed with limits |
| macOS | 3.14.9 | 637 passed, 13 deselected | Passed with limits |
| Debian bookworm container | 3.13.11 | 637 passed, 13 deselected | Passed with limits |
| Debian bookworm container | 3.14.9 | 637 passed, 13 deselected | Passed with limits |

Each profile passed Ruff lint and format across 104 files, plus mypy across 66 source modules.
Lock consistency, generated references, and import boundaries passed in every environment.
Installed wheels loaded outside the checkout and passed real CLI, stdio, HTTP, and restart checks.
Asset checks loaded CSS and datasets, opened the facts tab, and required the persisted fact's ID and content to appear.

Linux profiles used fresh tracked-file copies without host caches, virtual environments, or credentials.
Setup used a cached bookworm Python 3.13 image, installed Git, certificates, uv 0.13.5, and locked development dependencies.
Python 3.14 was downloaded during setup. The complete verification profiles then ran with `UV_OFFLINE=true`.
This separates permitted dependency setup from the offline acceptance run.
Mac profiles used isolated Python environments. This iteration made no pushes and ran no GitHub CI workflow.
The original checkout remained clean at `149cee2`. No test containers remained after verification.

Final logs are `/tmp/engram-excellence-313.log`, `/tmp/engram-excellence-314.log`,
`/tmp/engram-excellence-linux-313.log`, and `/tmp/engram-excellence-linux-314.log`.
These temporary paths identify the observed runs. They are not durable dependencies of this report.

## Final reassessment

No material findings remain in the audited local scope after the independent reviews and complete profile reruns.
Tests exercised intended behavior and explicit failure cases, including blocked storage, invalid dependencies, corrupt rows, and failed package imports.
The dataset contents, scoring thresholds, and populations remain unchanged. More checks do not establish a measured quality or performance improvement.

## External limits

The 13 deselected tests comprise 12 Git transport tests and one live-provider test.
Safe sync unit, local snapshot, union-merge, locking, and cancellation coverage remained in the offline suite.
Earlier legacy fixture attempts reached the push safeguard and were blocked. No push succeeded.
The explicit collection opt-in now prevents default and marker-typo test commands from attempting those pushes.
Run `just verify-sync` only where temporary-local-remote pushes are permitted.

No live OpenAI call, real memory store, hosted transport, or cross-machine deployment was exercised.
Hosted GitHub CI has not run because this branch has not been pushed.
The completed local macOS and Linux profiles establish their tested fixture, failure-handling, and packaging contracts.
They do not establish live model quality, production reliability, actual Git publication, or cross-machine convergence under real transport.

# Engram

Engram serves structured coding-agent memory through MCP and a CLI. It uses BM25, an event log, and the OpenAI SDK.

```bash
uv sync --locked --extra dev
just check
just fix
just verify
just reference
```

`just check` runs static checks, then the offline suite on parallel pytest workers. `just slow` adds subprocess smokes and an installed-wheel smoke.
Both use the same verification entrypoint as CI. Prove changes with `just verify` (`verify-engram` skill at `.agents/skills/verify-engram/`). Missing prerequisites and excluded profiles appear in the report.
`just verify-sync` pushes to temporary local Git remotes. CI runs it after `verify`; locally it stays separate because a machine-level push freeze can refuse it.

Read [architecture](docs/architecture.md) before subsystem changes.
Read [data](docs/data.md) for persistence, [sync](docs/sync.md) for Git integration, and [configuration](docs/config.md) for settings.
The generated [tool reference](docs/mcp-tools.md) comes from actual MCP registrations and CLI handlers.
Regenerate references after changing either surface or Settings descriptions.

## Enforced Rules

`tests/test_tooling.py::test_enforced_rules_cite_existing_paths` fails when a cited path or test is missing.

| Rule | Enforced by | Added |
| --- | --- | --- |
| Validate and recover fact batches before ordinary snapshots. | `src/engram/storage/transactions.py`, `tests/test_storage_transactions.py::test_recovery_reuses_exact_events_and_replays_twice` | 2026-11-04 |
| Use the shared lifecycle eligibility policy. | `src/engram/core/fact_state.py`, `tests/test_storage_transactions.py::test_required_dependencies_must_be_eligible_without_revision_change` | 2026-11-04 |
| Use public storage APIs and async facades. | `src/engram/tooling/boundaries.py`, `tests/test_tooling.py::test_boundary_rejects_private_policies_and_sync_async_bypasses` | 2026-11-04 |
| Keep runtime resource ownership explicit. | `src/engram/runtime.py`, `tests/test_runtime.py::test_provider_closes_when_final_sync_fails` | 2026-11-04 |
| Type every module strictly. | `pyproject.toml::typeCheckingMode`, run by `src/engram/tooling/verify.py::commands` | 2026-11-06 |
| Keep generated references aligned with source. | `src/engram/tooling/reference.py`, `tests/test_tooling.py::test_generated_references_are_current` | 2026-11-04 |
| Tests never call a live provider. | `tests/conftest.py`, `tests/test_tooling.py::test_tests_cannot_reach_live_provider` | 2026-07-16 |
| Live evals refuse without `ENGRAM_EVAL_PROVIDER=1`. | `src/engram/evaluation/extraction_quality.py::extract_live_provider_outputs`, `tests/test_extraction_quality_evals.py::test_live_extraction_refuses_without_approval` | 2026-11-06 |
| `just verify control` must observe FAIL before a verify pass counts. | `.agents/skills/verify-engram/drives.py::control` | 2026-11-06 |
| Git transport tests need explicit opt-in. | `tests/conftest.py`, `tests/test_tooling.py::test_wrong_marker_expression_cannot_enable_git_push_fixtures` | 2026-11-04 |

Compatibility changes still require judgment. Preserve MCP tool names, persisted event formats, and documented adapters.
Add settings to `Settings` with descriptions, then regenerate references.
Upgrade all writers before concurrent review. Local storage guarantees do not imply distributed serializability.

Evals run as modules, for example `uv run --locked python -m engram.evaluation.recall`.

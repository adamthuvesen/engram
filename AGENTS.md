# Engram

Engram serves structured coding-agent memory through MCP and a CLI. It uses BM25, an event log, and the OpenAI SDK.

```bash
uv sync --locked --extra dev
just check
just verify
just reference
```

`just check` runs focused offline checks. `just verify` runs the full offline suite and an installed-wheel smoke.
Both use the same verification entrypoint as CI. Missing prerequisites and excluded profiles appear in the report.
`just verify-sync` pushes to temporary local Git remotes. CI runs it after `verify`; locally it stays separate because a machine-level push freeze can refuse it.
Live evals require `ENGRAM_EVAL_PROVIDER=1`. Offline verification uses disabled or fixture providers.

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
| Type every module strictly. | `pyproject.toml` `[tool.mypy] strict`, run by `src/engram/tooling/verify.py` | 2026-11-06 |
| Keep generated references aligned with source. | `src/engram/tooling/reference.py`, `tests/test_tooling.py::test_generated_references_are_current` | 2026-11-04 |
| Tests never call a live provider. | `tests/conftest.py`, `tests/test_tooling.py::test_tests_cannot_reach_live_provider` | 2026-07-16 |
| Git transport tests need explicit opt-in. | `tests/conftest.py`, `tests/test_tooling.py::test_wrong_marker_expression_cannot_enable_git_push_fixtures` | 2026-11-04 |

Compatibility changes still require judgment. Preserve MCP tool names, persisted event formats, and documented adapters.
Add settings to `Settings` with descriptions, then regenerate references.
Upgrade all writers before concurrent review. Local storage guarantees do not imply distributed serializability.

Evals run as modules, for example `uv run --locked python -m engram.evaluation.recall`.

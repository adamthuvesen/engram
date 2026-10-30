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
`just verify-sync` adds tests that push to temporary local Git remotes. Run it only where pushes are allowed.
Live evals require `ENGRAM_EVAL_PROVIDER=1`. Offline verification uses disabled or fixture providers.

Read [architecture](docs/architecture.md) before subsystem changes.
Read [data](docs/data.md) for persistence, [sync](docs/sync.md) for Git integration, and [configuration](docs/config.md) for settings.
The generated [tool reference](docs/mcp-tools.md) comes from actual MCP registrations and CLI handlers.
Regenerate references after changing either surface or Settings descriptions.

## Enforced Rules

These controls live in code and checks. This table points to their implementation.

| Contract | Enforcement and check |
| --- | --- |
| Validate and recover fact batches before ordinary snapshots. | [Store](src/engram/storage/store.py) and [durable intent](src/engram/storage/transactions.py). `just verify` runs storage interruption and recovery tests. |
| Use the shared lifecycle eligibility policy. | [Domain policy](src/engram/core/fact_state.py), guarded storage changes, and [behavioral tests](tests/test_storage_transactions.py). `just verify`. |
| Use public storage APIs and async facades. | [Focused boundary checks](src/engram/tooling/boundaries.py) reject named private imports, direct async constructors, and sync-store bypasses. `just check`. |
| Keep runtime resource ownership explicit. | [Runtime](src/engram/runtime.py) owns settings, provider closure, recovery, and background shutdown. [Runtime tests](tests/test_runtime.py). `just verify`. |
| Validate provider and operation contracts. | [Provider schemas](src/engram/core/structured_outputs.py), [typed payloads](src/engram/core/operation_models.py), and scoped mypy checks. `just check` and `just verify`. |
| Keep tools and references aligned with source. | [Locked hooks](.pre-commit-config.yaml), [CI](.github/workflows/ci.yml), and [reference generation](src/engram/tooling/reference.py). `just check` verifies drift. |
| Require explicit Git transport test opt-in. | [Pytest collection](tests/conftest.py) skips transport fixtures without `--run-sync-transport`, even when marker selection changes. `just verify-sync` supplies the option. |

Compatibility changes still require judgment. Preserve MCP tool names, persisted event formats, and documented adapters.
Add settings to `Settings` with descriptions, then regenerate references.
Upgrade all writers before concurrent review. Local storage guarantees do not imply distributed serializability.

Typed evals and verification code live under `src/engram/evaluation` and `src/engram/tooling`.
Files under `tests/run_*evals.py` remain thin compatibility entrypoints.

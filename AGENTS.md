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
Default pytest excludes those tests and also requires `--run-sync-transport` to enable them.
Live evals require `ENGRAM_EVAL_PROVIDER=1`. Offline verification uses disabled or fixture providers.

Read [architecture](docs/architecture.md) before subsystem changes.
Read [data](docs/data.md) for persistence, [sync](docs/sync.md) for Git integration, and [configuration](docs/config.md) for settings.
The generated [tool reference](docs/mcp-tools.md) comes from actual MCP registrations and CLI handlers.
Regenerate references after changing either surface or Settings descriptions.

Keep these boundaries:

- Fact events are append-only. Purge, repair, and compaction are explicit exceptions with storage-level guards.
- Async application code uses `AsyncFactStore` or `run_sync`. Import boundary checks reject direct sync-store construction and bypasses.
- Lifecycle eligibility lives in `core.fact_state`. Application code must use public storage APIs.
- `Runtime` owns settings, provider connections, store recovery, and background task shutdown.
- MCP tool names are stable public interfaces. Preserve persisted event formats and documented compatibility adapters.
- Add settings to `Settings` with descriptions. Hooks and CI use the locked dependencies.
- Upgrade all writers before concurrent review. Local storage guarantees do not imply distributed serializability.

Typed evals and verification code live under `src/engram/evaluation` and `src/engram/tooling`.
Files under `tests/run_*evals.py` remain thin compatibility entrypoints.
Scoped mypy checks and `tooling.boundaries` enforce the changed domain, provider, eval, and tooling contracts.

# Engram

Engram stores structured memory for coding agents. It serves the same operations through MCP and a CLI.
Recall ranks dated memory cards with BM25. Saving extracts and reconciles cards through the OpenAI SDK.

```bash
uv sync --locked --extra dev
just check
uv run --locked engram --help
uv run --locked python -m engram.evaluation.recall
```

The eval command uses fictional committed data and a disabled provider. It does not read your memory store.
Python 3.13 and 3.14 run the same verification profiles in CI.

## Run Engram

```bash
uv run --locked engram                              # MCP over stdio
uv run --locked engram serve --transport http       # http://127.0.0.1:7422/mcp
uv run --locked engram-dash                         # terminal dashboard
uv run --locked engram inspect --json --limit 20
uv run --locked engram recall "editor preference" --project ~/dev/app
```

Bare `engram` starts the MCP server. Arguments select the CLI, so an unknown command reports a usage error.
The HTTP server uses stateless streamable HTTP on loopback. Clients can reconnect after a daemon restart.
FastMCP is pinned to the version in `uv.lock`.

For stdio clients:

```json
{
  "mcpServers": {
    "engram": {
      "command": "uv",
      "args": ["run", "--locked", "--directory", "/path/to/engram", "engram"]
    }
  }
}
```

See the generated [tool and command reference](docs/mcp-tools.md) for the full inventory.
Each tool also returns the response envelope as MCP structured content.
CLI commands accept `--json` for the same envelope.

## Memory behavior

`remember` extracts cards and reconciles them against neighboring memories. Candidate review can defer publication until approval.
Events retain corrections and supersessions. Local durable intent allows interrupted batches to resume with their original event IDs.
A process owns one store facade, one provider client, and its background tasks through `Runtime`.

Recall returns cards without a provider call when lexical evidence clears the relevance bar.
Weak hits can require provider selection. Answer mode generates structured claims over a capped evidence pool.
The renderer creates citations from validated fact references. Provider failures return cards with an explicit warning.
There are no embeddings or vector databases.

Background upkeep verifies anchors, consolidates cards, and builds project briefs.
Briefs record source revisions, so changing a source invalidates the previous brief.
The dashboard uses the shared lifecycle policy and distinguishes forgotten facts from superseded facts.

## Verify changes

```bash
just check           # focused offline checks
just verify          # full offline suite and installed-wheel smoke
just verify-package  # installed-wheel smoke only
just reference       # regenerate configuration and tool references
```

`just check` runs lock consistency, lint, format, types, generated references, import boundaries, and focused tests.
`just verify` adds the full offline test suite and installs a wheel outside the checkout.
The wheel smoke checks stdio, loopback HTTP, restart persistence, dashboard CSS, packaged datasets, and the headless dashboard.
Smoke runtimes use temporary data and disable maintenance and sync.
Injected runtimes use a disabled provider. Installed console smokes use a synthetic key and a blocked loopback provider endpoint.

Default pytest runs exclude live provider tests and sync transport tests.
The sync transport tests also require `--run-sync-transport`, so changing the marker expression cannot enable a push.
Safe sync unit, snapshot, merge, and cancellation tests remain in the offline profile.

```bash
just verify-sync     # additional integration profile, pushes to temporary local Git remotes
```

Run that profile only where Git pushes are allowed. The local push safeguard can prohibit it.
The verification report lists excluded profiles and reports `passed_with_limits` when they remain unverified.
CI runs the offline profile and the temporary-remote integration profile separately on both Python versions.

Git hooks use Ruff from `uv.lock`. Install them with `uv run --locked --extra dev pre-commit install`.
The build also uses the locked Hatchling dependency through `uv build --no-build-isolation`.

## Evaluate quality

Each evaluation is a module under `engram.evaluation`:

```bash
uv run --locked python -m engram.evaluation.recall
uv run --locked python -m engram.evaluation.cross_project
uv run --locked python -m engram.evaluation.extraction_quality
uv run --locked python -m engram.evaluation.memory_audit
uv run --locked python -m engram.evaluation.tiers
```

The score formulas, labels, floors, and query populations live beside the evaluation code.
Datasets ship under [evaluation/datasets](src/engram/evaluation/datasets).
Reports record the dataset SHA256, selector version, code revision, dirty-tree state, and live-provider status.
A missing code revision is explicit when an installed wheel runs outside a Git checkout.
Fixture scores verify deterministic contracts. They do not establish live model quality.

Live extraction requires an explicit opt-in and provider credentials:

```bash
ENGRAM_EVAL_PROVIDER=1 uv run --locked python -m engram.evaluation.extraction_quality --live
```

## Configuration and storage

See the generated [configuration reference](docs/config.md) for every `ENGRAM_*` setting.
`OPENAI_API_KEY` supplies provider credentials. Extraction, provider selection, answer mode, and some upkeep steps can spend provider tokens.

The default data directory is `~/.engram/data`. See [data and transactions](docs/data.md) and [cross-machine sync](docs/sync.md).
Upgrade every writer before concurrent candidate review. Older writers can rewrite candidate history.
Local locking and durable intent protect one data directory. Git sync does not provide distributed serializable transactions.
Purge and compaction refuse Git-backed stores because union merges can resurrect rewritten history.

Start subsystem work from the [architecture guide](docs/architecture.md).

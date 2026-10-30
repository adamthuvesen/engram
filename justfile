# Show available development commands.
default:
    @just --list

# Run locked lint, formatting, types, generated references, boundaries, and focused tests.
check:
    uv run --locked --extra dev python -m engram.tooling.verify check

# Run the full offline suite and an installed-wheel transport and asset smoke.
verify:
    uv run --locked --extra dev python -m engram.tooling.verify verify

# Run temporary-local-remote Git transport integration when pushes are allowed.
verify-sync:
    uv run --locked --extra dev python -m engram.tooling.verify sync-transport

# Build and install the wheel outside this checkout, then verify transports and assets.
verify-package:
    uv run --locked --extra dev python -m engram.tooling.verify package

# Regenerate configuration and command references from registered source contracts.
reference:
    uv run --locked --extra dev python -m engram.tooling.reference

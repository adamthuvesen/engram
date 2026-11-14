# Show available development commands.
default:
    @just --list

# Run locked lint, formatting, types, generated references, boundaries, then the offline suite on parallel pytest workers.
check:
    uv run --locked --extra dev python -m tools.verify check

# Apply every auto-fixer: ruff lint fixes, then ruff format.
fix:
    uv run --locked --extra dev ruff check --fix . .agents/skills/verify-engram/verify
    uv run --locked --extra dev ruff format . .agents/skills/verify-engram/verify

# Run check, then subprocess smokes and an installed-wheel transport and asset smoke.
slow: check
    uv run --locked --extra dev python -m tools.verify verify

# Drive the real CLI, MCP server, and wheel (verify-engram skill). Pass all, list, control, or drive names.
verify *args:
    .agents/skills/verify-engram/verify {{args}}

# Run temporary-local-remote Git transport integration when pushes are allowed.
verify-sync:
    uv run --locked --extra dev python -m tools.verify sync-transport

# Build and install the wheel outside this checkout, then verify transports and assets.
verify-package:
    uv run --locked --extra dev python -m tools.verify package

# Regenerate configuration and command references from registered source contracts.
reference:
    uv run --locked --extra dev python -m tools.reference

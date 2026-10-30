---
name: verify-engram
description: "Drives Engram's real CLI, MCP server, Textual dashboard, and built wheel against a throwaway store. Use to run, launch, drive, or verify engram, or to prove a change to it works."
---

# Verify Engram

`just check` proves the code is consistent. This skill proves the behavior. It drives the real `engram` console, the MCP server over stdio and HTTP, the dashboard through a Textual pilot, and an installed wheel. Every drive starts from a fresh temp store.

## Run

- `just verify` after a change. It runs the drives whose paths match files changed against `main`, plus uncommitted and untracked files. When nothing matches, it runs the `cli` smoke drive.
- `just verify all` before you hand over larger work. It takes about 15 to 20 seconds on an idle machine. It takes longer when many agents share the machine.
- `just verify list` shows each drive, what it proves, and the paths it covers.
- `just verify <drive> ...` runs only the named drives.
- `just verify control` drives `cli` with a wrong expected card id. It exits 0 only when it observes FAIL.

The helper is `.agents/skills/verify-engram/verify`. It runs itself under `uv run --locked --extra dev` when started outside the project venv.

## Reading results

- Each drive prints `PASS|FAIL <drive> <seconds>s <detail>`. A FAIL detail names what was expected and what happened.
- Evidence is in `.verify/runs/<utc>-<sha7>[-dirty]/`. It holds `summary.json` and `env.json` (secrets redacted). Each drive has a directory with one file per command (command, exit code, stdout, stderr), the MCP envelopes, `stderr.log`, the HTTP server log, and a dashboard SVG screenshot.
- The helper never retries. Any failure is a FAIL. There is no FLAKY result.

## Isolation

- Children get a temp HOME and temp XDG directories, `ENGRAM_DATA_DIR` in the scratch directory, and maintenance and sync turned off. Their cwd is outside any Git repository.
- The OpenAI endpoint is `http://127.0.0.1:1/v1` with a synthetic key, so a real provider call fails fast. The `cli` drive also asserts that recall used 0 LLM calls.
- The `memory` drive serves the production MCP app with a scripted `FixtureCompletionClient` at the `Runtime(provider=...)` boundary. This exercises extraction parsing and storage writes without a model.
- A preflight check confirms that `engram` imports from this checkout and that child settings resolve to the temp store. If a check fails, the run stops before any drive starts.
- Each drive runs its subprocesses in their own process group and stops that group when the drive ends.
- The run fails when it changes this checkout: HEAD, the branch, the git index, local or worktree git config, or any file outside `.git`, `.venv`, `.verify`, and tool caches, ignored paths included. `summary.json` records `ok` and `checkout_changes`.

## Limits

- The drives do not prove live extraction or recall-answer quality. To prove those, run the live evals by hand with `ENGRAM_EVAL_PROVIDER=1`. They spend money.
- Git sync is not driven. Run `just verify-sync`, which pushes to temporary local remotes.
- The dashboard drive uses Textual's pilot in process. It does not run a real terminal.

## Gotchas

- The dashboard `f` key forgets the facts that are marked with space, or the row opened with Enter. A highlighted row alone is not a target.
- `inspect --project X` also returns global cards (cards with no project). Assert on the project's own ids, not on an empty result.
- The `package` drive needs the uv cache for its `--offline` install. The helper passes the real `UV_CACHE_DIR` into the temp HOME.

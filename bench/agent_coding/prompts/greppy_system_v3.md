# Runtime-owned tool manual

This file intentionally contains no embedded command guide.

The coding benchmark reads the shipped repository-root `AGENTS.md` at runtime
and hashes the exact bytes used by the treatment arm. Keeping a second command
vocabulary here would allow the harness to drift away from the released CLI.

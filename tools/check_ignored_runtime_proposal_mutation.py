#!/usr/bin/env python3
"""Actual proposal publication, original ignored-runtime fault, exact restoration."""
from pathlib import Path
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/agent/src/portable_workspace.rs"
PREFIX = "portable_workspace::tests::"
NEW = "ordinary_finish_ignored_runtime_metadata_does_not_block_proposal"
EXISTING = "ordinary_finish_preserves_user_web_ancestry_and_excludes_new_routes"
COMMAND = ["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy-agent",
           "--lib", PREFIX + "ordinary_finish_",
           "--", "--test-threads=1", "--nocapture"]


def run(red=False):
    result = subprocess.run(COMMAND, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "CARGO_BUILD_JOBS": "2", "RUST_TEST_THREADS": "1"},
                            timeout=300)
    output = result.stdout + result.stderr
    print(output, flush=True)
    if red:
        assert result.returncode == 101, "counter must reach a real Rust test failure"
        assert "ignored runtime exclusion must not block a source proposal" in output
        assert "following paths are ignored" in output
        assert PREFIX + NEW + " ... FAILED" in output
        print("mutation: redundant ignored runtime exclusion blocks actual proposal publication", flush=True)
    else:
        assert result.returncode == 0, "actual proposal contract failed"
        for name in (NEW, EXISTING):
            assert PREFIX + name + " ... ok" in output
        print("mutation: source proposal saved, new runtime omitted, baseline user route and index preserved", flush=True)


original = SOURCE.read_bytes()
run()
anchor = "filter_ignored_paths(&self.worktree, &self.private_index, excluded_paths)?"
assert original.decode().count(anchor) == 1, "ignored exclusion source target is not unique"
try:
    SOURCE.write_text(original.decode().replace(anchor, "excluded_paths", 1))
    run(red=True)
finally:
    SOURCE.write_bytes(original)
run()
assert SOURCE.read_bytes() == original
subprocess.run(["git", "diff", "--exit-code", "--", str(SOURCE)], cwd=ROOT, check=True)

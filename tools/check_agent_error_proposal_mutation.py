#!/usr/bin/env python3
"""Prove real edit -> HTTP error -> saved partial proposal without apply."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/cli/src/agent.rs"
TEST = "greppy_p_http_error_saves_partial_proposal_without_applying"
FIXED = """                // The common finish path saves edits, keeps the error exit,
                // and refuses automatic application of a failed proposal.
                summary.error = Some(message);
                Ok(summary)
"""
LEGACY = "                Err(message)\n"


def run(red=False):
    env = os.environ.copy()
    env["RUST_MIN_STACK"] = "8388608"
    result = subprocess.run(
        ["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy",
         "--features", "ci-test-assets", "--test", "agent_p", TEST, "--",
         "--exact", "--test-threads=1"],
        cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=600,
    )
    if red:
        if not (result.returncode == 101 and TEST + " ... FAILED" in result.stdout
                and "HTTP error must publish the partial proposal" in result.stdout):
            raise AssertionError("Wrong mutation cause or unexpected success:\n" + result.stdout)
    elif not (result.returncode == 0 and TEST + " ... ok" in result.stdout):
        raise AssertionError("Real HTTP-error proposal contract did not pass:\n" + result.stdout)


run()
original = SOURCE.read_text()
if original.count(FIXED) != 1:
    raise AssertionError("Partial proposal mutation anchor must be unique")
try:
    SOURCE.write_text(original.replace(FIXED, LEGACY, 1))
    run(red=True)
    print("mutation: legacy early HTTP-error return loses the real partial proposal (exit101)", flush=True)
finally:
    SOURCE.write_text(original)
run()
print("mutation: restored error result retains exact edits, refuses apply and releases the worktree", flush=True)

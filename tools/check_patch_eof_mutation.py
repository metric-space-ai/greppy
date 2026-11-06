#!/usr/bin/env python3
"""Prove the reported EOF-only patch writes bytes, not an already-as-sent receipt."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/cli/src/edit.rs"
NAME = "edit::patch_rollback_tests::unified_patch_eof_newline_is_a_real_atomic_byte_change"
CONTROL_NAMES = [
    NAME,
    "edit::patch_rollback_tests::unified_patch_eof_markers_preserve_each_side_and_surrounding_bytes",
    "edit::patch_rollback_tests::unified_patch_eof_invalid_marker_never_publishes_any_file",
]
COMMAND = ["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy",
           "--features", "ci-test-assets", "--lib", "unified_patch_eof_",
           "--", "--test-threads=1"]


def run(red=False):
    result = subprocess.run(COMMAND, cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=600)
    if red:
        assert (result.returncode == 101 and "test result: FAILED" in result.stdout
                and NAME + " ... FAILED" in result.stdout
                and "EOF-only addition is a real byte edit" in result.stdout), result.stdout
    else:
        assert result.returncode == 0, result.stdout
        assert all(name + " ... ok" in result.stdout for name in CONTROL_NAMES), result.stdout
    return result


run()
original = SOURCE.read_text()
fixed = "&& (had_final_ending || hunk.old_no_final_newline)"
assert original.count(fixed) == 1, "EOF newline emission anchor must be unique"
try:
    SOURCE.write_text(original.replace(fixed, "&& had_final_ending", 1))
    run(red=True)
    print("mutation: restoring inherited-only EOF newline behavior breaks the real published-byte contract", flush=True)
finally:
    SOURCE.write_text(original)
run()
subprocess.run(["git", "diff", "--exit-code", "--", str(SOURCE)], cwd=ROOT, check=True)
print("mutation: restored EOF byte edit, per-side markers and atomic refusal contracts green", flush=True)

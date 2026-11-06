#!/usr/bin/env python3
"""Prove EOF-only edits and empty replacement hunks publish the exact requested bytes."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/cli/src/edit.rs"
NAME = "edit::patch_rollback_tests::unified_patch_eof_newline_is_a_real_atomic_byte_change"
EMPTY_NAME = "edit::patch_rollback_tests::unified_patch_eof_empty_hunk_removes_the_range_without_a_blank_line"
CONTROL_NAMES = [
    NAME,
    EMPTY_NAME,
    "edit::patch_rollback_tests::unified_patch_eof_markers_preserve_each_side_and_surrounding_bytes",
    "edit::patch_rollback_tests::unified_patch_eof_invalid_marker_never_publishes_any_file",
]
COMMAND = ["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy",
           "--features", "ci-test-assets", "--lib", "unified_patch_eof_",
           "--", "--test-threads=1"]


def run(red=None):
    result = subprocess.run(COMMAND, cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=600)
    if red:
        name, message = red
        assert (result.returncode == 101 and "test result: FAILED" in result.stdout
                and name + " ... FAILED" in result.stdout
                and message in result.stdout), result.stdout
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
    run(red=(NAME, "EOF-only addition is a real byte edit"))
    print("mutation: restoring inherited-only EOF newline behavior breaks the real published-byte contract", flush=True)
finally:
    SOURCE.write_text(original)
empty = "if !hunk.new_lines.is_empty()\n            && !hunk.new_no_final_newline"
assert original.count(empty) == 1, "empty replacement emission anchor must be unique"
try:
    SOURCE.write_text(original.replace(empty, "if !hunk.new_no_final_newline", 1))
    run(red=(EMPTY_NAME, "delete-only hunk must remove bytes instead of inserting a blank line"))
    print("mutation: restoring newline emission for empty replacement breaks the delete-only publish contract", flush=True)
finally:
    SOURCE.write_text(original)
run()
subprocess.run(["git", "diff", "--exit-code", "--", str(SOURCE)], cwd=ROOT, check=True)
print("mutation: restored EOF byte edit, delete-only hunks, per-side markers and atomic refusal contracts green", flush=True)

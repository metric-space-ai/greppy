#!/usr/bin/env python3
"""Compile the actual sparse-coverage helpers and prove two regressions fail.

This narrow std-only Rust harness needs Git, not models or an index. The normal
CLI tests still compile these same functions in the complete crate. Mutants are
isolated source copies: repository source is never overwritten.
"""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(argv, **kwargs):
    return subprocess.run(argv, capture_output=True, text=True, timeout=90, **kwargs)


def main():
    vcs = (ROOT / "crates/cli/src/vcs.rs").read_text()
    indexing = (ROOT / "crates/cli/src/indexing.rs").read_text()
    start = indexing.index("fn published_coverage_warning(")
    end = indexing.index("fn background_health_state(", start)
    original = vcs + "\n" + indexing[start:end]
    variants = [
        ("positive", original, None),
        (
            "counts_sparse_exclusions_as_source",
            original.replace("u64::from(record[0] == b'S')", "0", 1),
            "real_git_sparse_checkout_excludes_only_skip_worktree_entries",
        ),
        (
            "uses_full_git_denominator",
            original.replace("let tracked = tracked.non_skip_worktree();", "let tracked = tracked.total;", 1),
            "coverage_warning_respects_sparse_scope_without_hiding_real_gaps",
        ),
        ("restored", original, None),
    ]
    with tempfile.TemporaryDirectory(prefix="greppy-sparse-coverage-") as tmp:
        for label, source, expected_failure in variants:
            if expected_failure and source == original:
                raise AssertionError(f"mutation {label} did not change the source")
            unit = Path(tmp) / f"{label}.rs"
            binary = Path(tmp) / (f"{label}.exe" if os.name == "nt" else label)
            unit.write_text(source)
            build = run(["rustc", "--edition=2021", "--test", str(unit), "-o", str(binary)])
            if build.returncode:
                raise AssertionError(f"{label}: actual-source Rust compile failed\n{build.stdout}{build.stderr}")
            result = run([str(binary), "--test-threads=1"], env={**os.environ, "TMPDIR": tmp})
            output = result.stdout + result.stderr
            if expected_failure:
                if result.returncode == 0 or f"{expected_failure} ... FAILED" not in output:
                    raise AssertionError(f"{label}: named counterexample did not fail\n{output}")
                print(f"mutation: {label}: expected named regression detected ({expected_failure})", flush=True)
            elif result.returncode:
                raise AssertionError(f"{label}: contract failed\n{output}")
            else:
                print(f"mutation: {label}: real Git sparse scope and genuine-gap contracts passed", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Real CLI definition bodies, old preview-only fault, exact restoration."""
from pathlib import Path
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/cli/src/search.rs"
TEST = "code_reads_rust_python_definitions_without_repeating_bodies"
COMMAND = ["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy",
           "--features", "ci-test-assets", "--test", "search_pattern_preview"]


def run(red=None):
    argv = COMMAND + ([TEST] if red else []) + ["--", "--test-threads=1", "--nocapture"]
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "CARGO_BUILD_JOBS": "2", "RUST_MIN_STACK": "8388608"},
                            timeout=300)
    output = result.stdout + result.stderr
    print(output, flush=True)
    if red:
        assert result.returncode == 101, "counter must reach a real compiled test failure"
        assert TEST + " ... FAILED" in output and "test result: FAILED" in output
        assert red in output, "counter failed outside its target contract"
        print("mutation: original source-return regression rejected: " + red, flush=True)
    else:
        assert result.returncode == 0 and TEST + " ... ok" in output
        print("mutation: real Rust/Python source bodies, reuse, bounded previews and pagination passed", flush=True)


original = SOURCE.read_bytes()
run()
faults = [
    ("definition body must be present",
     "    let Some(node) = row.node.as_ref() else {",
     "    if row.node.is_some() { return Ok(false); }\n    let Some(node) = row.node.as_ref() else {"),
    ("definition body must not be repeated",
     "    if shown_definitions.contains(&identity) {",
     "    if false && shown_definitions.contains(&identity) {"),
    ("live definition must exclude its unrelated neighbor",
     "    let end_line = node.end_line;",
     "    let end_line = read_span_with_meta(root_path, &node.file_path, start_line, node.end_line, usize::MAX, false).map(|span| span.end_line).unwrap_or(node.end_line);"),
    ("JSON source must equal its addressed bytes",
     "        source: span.text().into_owned(),",
     "        source: span.text().trim_end_matches('\\n').to_owned(),"),
]
for diagnostic, anchor, mutant in faults:
    assert original.decode().count(anchor) == 1, "source target must be unique"
    try:
        SOURCE.write_text(original.decode().replace(anchor, mutant, 1))
        run(red=diagnostic)
    finally:
        SOURCE.write_bytes(original)
    assert SOURCE.read_bytes() == original
    run()
subprocess.run(["git", "diff", "--exit-code", "--", str(SOURCE)], cwd=ROOT, check=True)

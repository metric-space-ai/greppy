#!/usr/bin/env python3
"""Prove saved ANSI/Oxc replay contracts against production parser mutations."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/cli/src/bash_smart.rs"
FILTER = "ansi_oxc_"
HEADERS = "bash_smart::tests::ansi_oxc_headers_preserve_capture_bytes_and_reject_source"
TOTALS = "bash_smart::tests::ansi_oxc_summary_totals_cover_rules_without_losing_other_work"
ANSI = "ansi_oxc_saved_ansi_error_reaches_verdict_and_compact_block"
OXC = "ansi_oxc_saved_synthetic_oxc_recap_reaches_verdict_and_compact_block"


def run(kind, expected, red=None):
    target = ["--lib"] if kind == "lib" else ["--test", "bash_smart"]
    args = ["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy",
            "--features", "ci-test-assets", *target, FILTER, "--", "--test-threads=1"]
    env = os.environ.copy()
    env["RUST_MIN_STACK"] = "8388608"
    result = subprocess.run(args, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    if red:
        test, cause = red
        if not (result.returncode == 101 and test + " ... FAILED" in result.stdout
                and "test result: FAILED" in result.stdout and cause in result.stdout):
            raise AssertionError("Wrong mutation failure or unexpected success:\n" + result.stdout)
    elif not (result.returncode == 0 and all(test + " ... ok" in result.stdout for test in expected)):
        raise AssertionError("Real parser/replay contract did not execute green:\n" + result.stdout)


def green():
    run("lib", [HEADERS, TOTALS])
    run("integration", [ANSI, OXC])


green()
print("mutation: saved ANSI and explicitly synthetic Oxc CLI replays green, short/folded stdout/stderr and child exits preserved", flush=True)
original = SOURCE.read_text()
normalization = '    ANSI_SGR_RE.replace_all(bytes, &b""[..])'
if original.count(normalization) != 1:
    raise AssertionError("SGR normalization mutation anchor must be unique")
try:
    SOURCE.write_text(original.replace(normalization, "    std::borrow::Cow::Borrowed(bytes)", 1))
    run("lib", [], red=(HEADERS, "SGR styling must not hide diagnostic headers"))
    run("integration", [], red=(ANSI, "ansi-error diagnostic verdict regression"))
    print("mutation: bypassing SGR normalization makes the real parser and saved ANSI CLI replay red (exit101)", flush=True)
finally:
    SOURCE.write_text(original)

start = original.index("fn diagnostic_counts(")
end = original.index("\nfn detect_blocks(", start)
legacy = """fn diagnostic_counts(blocks: &[DiagnosticBlock]) -> (usize, usize) {
    let errors = blocks.iter().filter(|block| block.kind == BlockKind::Error).count();
    (errors, blocks.len().saturating_sub(errors))
}
"""
try:
    SOURCE.write_text(original[:start] + legacy + original[end:])
    run("lib", [], red=(TOTALS, "Oxc recap totals must replace duplicate rule/footer counts"))
    run("integration", [], red=(OXC, "oxc-warning diagnostic verdict regression"))
    print("mutation: restoring legacy block-only totals makes Oxc parser and actual CLI recap verdict red (exit101)", flush=True)
finally:
    SOURCE.write_text(original)
green()
if SOURCE.read_text() != original:
    raise AssertionError("Production source was not restored byte-for-byte")
print("mutation: restored production source; saved parser/replay contracts green, original exit statuses and diagnostic bytes intact", flush=True)

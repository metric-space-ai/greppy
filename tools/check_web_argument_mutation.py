#!/usr/bin/env python3
"""Prove malformed Web argv cannot be silently rewritten and dispatched."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/cli/src/lib.rs"
NAME = "web_parse_errors_never_rewrite_or_dispatch_another_operation"
COMMAND = ["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy",
           "--features", "ci-test-assets", "--test", "web_argument_contract",
           NAME, "--", "--test-threads=1"]

def run():
    result = subprocess.run(COMMAND, cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    print(result.stdout, flush=True)
    return result

fixed = '// dispatch a different operation against the current session.\n            if sub == "web" {'
original = SOURCE.read_text()
assert original.count(fixed) == 1, "Web parse guard must be unique"
assert run().returncode == 0, "fixed command contract must pass first"
try:
    SOURCE.write_text(original.replace(fixed, fixed.replace('if sub == "web"', 'if sub == "web" && false'), 1))
    red = run()
    assert (red.returncode == 101 and "test result: FAILED" in red.stdout
            and NAME + " ... FAILED" in red.stdout
            and "must refuse before Web dispatch" in red.stdout), red.stdout
    print("mutation: removing the Web argv guard fails the actual CLI dispatch contract", flush=True)
finally:
    SOURCE.write_text(original)
assert run().returncode == 0, "restored Web argv guard must pass"
subprocess.run(["git", "diff", "--exit-code", "--", str(SOURCE)], cwd=ROOT, check=True)
print("mutation: restored Web argument refusal green; no browser fixture needed", flush=True)

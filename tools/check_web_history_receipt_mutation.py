#!/usr/bin/env python3
"""Mutate the production history receipt; require incident-specific failures."""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--native", action="store_true")
args = parser.parse_args()
base = ["cargo", "test", "--locked", "--manifest-path", "crates/web-runtime/Cargo.toml",
        "--package", "web-runtime"]
if args.native:
    source = Path("crates/web-runtime/runtime/src/content_worker.rs")
    command = base + ["--test", "session-daemon", "native_history_waits_for_matching_traversal",
                      "--", "--test-threads=1", "--nocapture"]
    mutations = [("delegate.history_receipt.borrow().is_complete()", "true", 2,
                  ("goBack expected", "goForward expected"))]
    timeout = 1200
else:
    source = Path("crates/web-runtime/runtime/src/history_receipt.rs")
    command = base + ["--lib", "--no-default-features", "history_receipt",
                      "--", "--test-threads=1", "--nocapture"]
    mutations = [
        ("if expected == token {", "if true {", 1, ("foreign history receipt was accepted",)),
        ("is_some_and(|(_, complete)| *complete)", "is_some_and(|(_, _)| true)", 1,
         ("pending history was treated as complete",)),
        ("self.pending = None;", "/* omitted release */", 1, ("left: 2",)),
    ]
    timeout = 600
original = source.read_text()
try:
    for old, new, count, messages in mutations:
        assert original.count(old) == count, (old, original.count(old))
        source.write_text(original.replace(old, new))
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        output = result.stdout + result.stderr
        print(output, flush=True)
        assert result.returncode == 101 and any(message in output for message in messages), (
            "mutation did not reach its intended history assertion", result.returncode, messages)
        print("mutation: history receipt regression detected", flush=True)
finally:
    source.write_text(original)
subprocess.run(command, check=True, timeout=timeout)
subprocess.run(["git", "diff", "--exit-code", "--", str(source)], check=True)

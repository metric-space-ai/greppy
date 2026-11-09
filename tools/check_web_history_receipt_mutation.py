#!/usr/bin/env python3
"""Exercise native receipt loss, then bypass the real completion gate."""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--native", action="store_true")
args = parser.parse_args()
base = ["cargo", "test", "--locked", "--jobs", "2",
        "--manifest-path", "crates/web-runtime/Cargo.toml", "--package", "web-runtime"]
if args.native:
    source = Path("crates/web-runtime/runtime/src/content_worker.rs")
    positive = "native_history_waits_for_matching_traversal"
    negative = "native_history_rejects_missing_traversal_receipt"

    def invoke(name, *, ignored=False, red=None):
        command = base + ["--test", "session-daemon", name, "--", "--exact",
                          "--test-threads=1", "--nocapture"]
        if ignored:
            command.append("--ignored")
        result = subprocess.run(command, capture_output=True, text=True, timeout=1200)
        output = result.stdout + result.stderr
        if red is None:
            valid = result.returncode == 0 and name in output and "test result: ok. 1 passed;" in output
        else:
            valid = (result.returncode == 101 and name in output
                     and "test result: FAILED" in output and red in output)
        if not valid:
            raise AssertionError(output)
        print("mutation: " + name + (" detects missing completion gate" if red else " passed"), flush=True)

    original = source.read_text()
    callback = "if self.history_receipt.borrow_mut().complete(&traversal_id) {"
    gate = "delegate.history_receipt.borrow().is_complete()"
    assert original.count(callback) == 1
    assert original.count(gate) == 2
    held = original.replace(callback, callback.replace("if ", "if false && ", 1))
    try:
        # Actual Servo traverses, but its completion callback is deliberately lost.
        # The API must time out even if the visible destination has already changed.
        source.write_text(held)
        invoke(negative, ignored=True)
        # Keep exactly that native fault, then remove both production wait gates.
        source.write_text(held.replace(gate, "true"))
        invoke(negative, ignored=True,
               red="history traversal completed without a matching native receipt")
    finally:
        source.write_text(original)
    invoke(positive)
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
    original = source.read_text()
    try:
        for old, new, count, messages in mutations:
            assert original.count(old) == count, (old, original.count(old))
            source.write_text(original.replace(old, new))
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            output = result.stdout + result.stderr
            print(output, flush=True)
            assert result.returncode == 101 and any(message in output for message in messages), (
                "mutation did not reach its intended history assertion", result.returncode, messages)
            print("mutation: history receipt regression detected", flush=True)
    finally:
        source.write_text(original)
    subprocess.run(command, check=True, timeout=600)
subprocess.run(["git", "diff", "--exit-code", "--", str(source)], check=True)

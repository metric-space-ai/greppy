#!/usr/bin/env python3
"""Real native public-stop receipt, original false-success counter, exact restore."""
from pathlib import Path
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/cli/src/web/common.rs"
TEST = "runtime_stop_reports_observed_state_without_weakening_ownership"
metadata = subprocess.run(
    ["cargo", "metadata", "--locked", "--manifest-path", "crates/web-runtime/Cargo.toml",
     "--no-deps", "--format-version", "1"], cwd=ROOT, check=True,
    capture_output=True, text=True, timeout=60,
)
runtime = Path(json.loads(metadata.stdout)["target_directory"]) / "debug/web-runtime"
assert runtime.is_file(), "actual native runtime was not built; physical contract may not skip"
env = {**os.environ, "GREPPY_WEB_RUNTIME": str(runtime)}
command = ["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy", "--features",
           "ci-test-assets", "--test", "web_cli", TEST,
           "--", "--test-threads=1", "--nocapture"]


def run(red=False):
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=600)
    output = result.stdout + result.stderr
    print(output)
    if red:
        assert result.returncode == 101 and "unowned live runtime stop must not report success" in output
        print("mutation: original false-success stop receipt rejected by actual native control", flush=True)
    else:
        assert result.returncode == 0 and "RUNTIME_STOP_REAL_CONTROL_PASSED" in output
        print("mutation: real denied/owned/idempotent public-stop contract passed", flush=True)


original = SOURCE.read_bytes()
run()
old = "        if running {\n            let next = if live.is_some() || acknowledged {"
assert original.decode().count(old) == 1, "public-stop counter source target not unique"
try:
    SOURCE.write_text(original.decode().replace(old, old.replace("if running", "if false && running"), 1))
    run(red=True)
finally:
    SOURCE.write_bytes(original)
run()
assert SOURCE.read_bytes() == original
subprocess.run(["git", "diff", "--exit-code", "--", str(SOURCE)], cwd=ROOT, check=True)

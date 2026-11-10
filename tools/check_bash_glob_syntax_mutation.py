#!/usr/bin/env python3
"""Real native Bash / public atomic-write counterproof, never script execution."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/edit/src/txn.rs"
ENV = os.environ.copy()
ENV["RUST_MIN_STACK"] = "8388608"
FEATURES = ENV.get("GREPPY_BASH_CONTRACT_FEATURES", "ci-test-assets")
VALID = "complete_bash_absolute_glob_write_matches_native_syntax"


def run(red=False):
    result = subprocess.run(["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy", "--features", FEATURES,
                             "--test", "bash_glob_write", *([VALID] if red else []), "--", "--test-threads=1", "--nocapture"],
                            cwd=ROOT, env=ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    if red:
        assert result.returncode == 101 and VALID + " ... " in result.stdout and "valid complete Bash glob write rejected" in result.stdout and "BASH_GLOB_SYNTAX_VALID" in result.stdout and "missing `]]`" in result.stdout, result.stdout
    else:
        assert result.returncode == 0 and all(name + " ... " in result.stdout for name in [VALID, "broken_complete_bash_remains_atomic", "bash_recovery_is_dialect_bound_and_never_executes_script_or_startup_code"]), result.stdout


original = SOURCE.read_bytes()
try:
    run()
    print("mutation: trusted env-cleared absolute bash -n accepts complete fixture; real public write preserves exact source; native-invalid complete results remain atomically unwritten; Bash dialect and poisoned PATH/BASH_ENV/ENV/script nonexecution green", flush=True)
    text = original.decode()
    anchor = "if explicit_bash"
    assert text.count(anchor) == 1
    try:
        SOURCE.write_text(text.replace(anchor, "if false && explicit_bash", 1))
        run(red=True)
        print("mutation: disabling only exact Bash glob validation recovery restores original native-valid / Greppy Exit13 missing ]] failure", flush=True)
    finally:
        SOURCE.write_bytes(original)
    run()
    assert SOURCE.read_bytes() == original
    print("mutation: production source restored byte-for-byte; native/public-write, atomic invalid-source and nonexecution/dialect contracts green again", flush=True)
finally:
    SOURCE.write_bytes(original)

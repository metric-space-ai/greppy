#!/usr/bin/env python3
"""Exercise actual CLI Bash writes and reintroduce the released grammar defect."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "Cargo.toml"
LOCK = ROOT / "Cargo.lock"
NAME = "bash_single_equals_logical_write_is_exact_and_refusal_is_atomic"
FIXED = 'tree-sitter-bash = { path = "third_party/tree-sitter-bash", version = "=0.25.1" }'
LOCAL_LOCK = 'name = "tree-sitter-bash"\nversion = "0.25.1"\ndependencies = ['
OLD_LOCK = ('name = "tree-sitter-bash"\nversion = "0.25.1"\n'
            'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
            'checksum = "9e5ec769279cc91b561d3df0d8a5deb26b0ad40d183127f409494d6d8fc53062"\n'
            'dependencies = [')


def run(red=False):
    env = os.environ.copy()
    env["RUST_MIN_STACK"] = "8388608"
    result = subprocess.run(
        ["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy",
         "--features", "ci-test-assets", "--test", "cli_hardening", NAME,
         "--", "--test-threads=1"], cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900,
    )
    if red:
        if not (result.returncode == 101 and NAME + " ... FAILED" in result.stdout
                and "valid complete Bash dry-run must pass" in result.stdout
                and 'missing `]]`' in result.stdout and "left: 13" in result.stdout):
            raise AssertionError("Wrong failure or unexpected mutant success:\n" + result.stdout)
    elif not (result.returncode == 0 and NAME + " ... ok" in result.stdout):
        raise AssertionError("Actual CLI Bash contract did not pass:\n" + result.stdout)


run()
manifest = MANIFEST.read_text()
lock = LOCK.read_text()
if manifest.count(FIXED) != 1 or lock.count(LOCAL_LOCK) != 1:
    raise AssertionError("Grammar/lock mutation anchors must be unique")
try:
    MANIFEST.write_text(manifest.replace(FIXED, 'tree-sitter-bash = "=0.25.1"', 1))
    LOCK.write_text(lock.replace(LOCAL_LOCK, OLD_LOCK, 1))
    run(red=True)
    print("mutation: released Bash grammar makes real CLI valid dry-run red with exit13/missing ]]", flush=True)
finally:
    MANIFEST.write_text(manifest)
    LOCK.write_text(lock)
run()
print("mutation: restored generated grammar passes actual CLI exact write, dry-run and atomic invalid-Bash refusal", flush=True)

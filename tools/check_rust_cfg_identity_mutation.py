#!/usr/bin/env python3
"""First establish the real parser/store/read collision before changing identity."""
import os
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["RUST_MIN_STACK"] = "8388608"
r = subprocess.run(["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy", "--features", "ci-test-assets", "--test", "rust_cfg_identity", "rust_cfg_siblings_survive_parser_store_and_public_read", "--", "--test-threads=1"], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
assert r.returncode == 101 and "cfg sibling definitions collapsed: parser=2, store=1, read_exit=0" in r.stdout and "other_leaf()" in r.stdout, r.stdout
print("mutation: original cfg sibling collision proved by real parser=2 -> Store=1 -> successful public read of later not(unix) body; single-name source counterprobe passed", flush=True)

#!/usr/bin/env python3
"""Prove real C/C++ extraction and CLI read against the original alias omission."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/parser/src/extract.rs"
PARSER = "anonymous_compound_typedef_keeps_alias_span_members_and_tag_collisions"
CLI = "anonymous_compound_typedef_read_returns_complete_definition"


def run(kind, red=False):
    if kind == "parser":
        args = ["-p", "greppy-parser", "--lib", PARSER]
        test, cause = PARSER, "anonymous compound typedef alias missing"
    else:
        args = ["-p", "greppy", "--features", "ci-test-assets",
                "--test", "read_file_dispatch", CLI]
        test, cause = CLI, "anonymous compound typedef read failed"
    env = os.environ.copy()
    env["RUST_MIN_STACK"] = "8388608"
    result = subprocess.run(
        ["cargo", "test", "--locked", "--jobs", "2", *args,
         "--", "--test-threads=1"], cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600,
    )
    if red:
        valid = (result.returncode == 101 and test + " ... FAILED" in result.stdout
                 and cause in result.stdout)
    else:
        valid = result.returncode == 0 and test + " ... ok" in result.stdout
    if not valid:
        raise AssertionError(result.stdout)


def green():
    run("parser")
    run("cli")


original = SOURCE.read_bytes()
green()
print("mutation: real C/C++ parser preserves aliases, full spans, members and tagged collisions; CLI read returns full synthetic typedefs", flush=True)
text = original.decode()
anchor = "                c_cpp_anonymous_compound_typedef(source, file_path, node, result);"
assert text.count(anchor) == 1, "alias omission mutation anchor must be unique"
try:
    SOURCE.write_text(text.replace(anchor, "                // Mutant: retain original anonymous typedef omission.", 1))
    run("parser", red=True)
    run("cli", red=True)
    print("mutation: removing production anonymous-typedef extraction makes real parser and CLI read contracts red at the original omission", flush=True)
finally:
    SOURCE.write_bytes(original)
green()
assert SOURCE.read_bytes() == original, "production source not restored byte-for-byte"
print("mutation: byte-exact source restore; real parser and CLI read contracts green", flush=True)

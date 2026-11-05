#!/usr/bin/env python3
"""Prove both fresh and cached Go usage regressions fail with the original filter."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/parser/src/extract.rs"
FIXED = b"""                if is_definition_name(node)
                    && !node
                        .parent()
                        .is_some_and(|parent| parent.kind() == "qualified_type")
                {"""
MUTANT = b"                if is_definition_name(node) {"
CASES = [
    ("greppy-parser", "qualified_go_types_are_usages_not_definition_names",
     "qualified types must survive the definition-name filter"),
    ("greppy-indexer", "old_go_cache_repairs_private_and_overlay_without_invalidating_mixed_base",
     "source-validated cache repair must recover the missing Go usage"),
]


def run(package, exact=None):
    command = [
        "cargo", "test", "--locked", "--jobs", "2", "-p", package,
        "--test", "go_qualified_type_usage", "--",
    ]
    if exact:
        command += ["--exact", exact]
    command += ["--nocapture", "--test-threads=1"]
    return subprocess.run(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=600, check=False,
    )


def main():
    original = SOURCE.read_bytes()
    if original.count(FIXED) != 1:
        raise SystemExit("mutation target must match exactly once")
    try:
        SOURCE.write_bytes(original.replace(FIXED, MUTANT, 1))
        for package, exact, expected in CASES:
            red = run(package, exact)
            if (
                red.returncode != 101
                or expected not in red.stdout
                or "test result: FAILED" not in red.stdout
            ):
                print(red.stdout)
                raise SystemExit(f"mutation did not fail at {exact}")
            print(f"mutation: original filter turns {exact} red")
    finally:
        SOURCE.write_bytes(original)
    for package, _, _ in CASES:
        green = run(package)
        if green.returncode != 0:
            print(green.stdout)
            raise SystemExit(f"restored qualified-type contracts did not pass in {package}")
        print(f"mutation: restored fix turns {package} usage/cache contracts green")


if __name__ == "__main__":
    main()

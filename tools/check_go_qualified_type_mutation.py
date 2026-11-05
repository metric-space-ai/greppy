#!/usr/bin/env python3
"""Prove the Go qualified-type regression fails with the original filter."""
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
COMMAND = [
    "cargo", "test", "--locked", "--jobs", "2", "-p", "greppy-parser",
    "--test", "go_qualified_type_usage", "--", "--exact",
    "qualified_go_types_are_usages_not_definition_names", "--nocapture",
    "--test-threads=1",
]


def run():
    return subprocess.run(
        COMMAND, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=600, check=False,
    )


def main():
    original = SOURCE.read_bytes()
    if original.count(FIXED) != 1:
        raise SystemExit("mutation target must match exactly once")
    try:
        SOURCE.write_bytes(original.replace(FIXED, MUTANT, 1))
        red = run()
        if (
            red.returncode != 101
            or "qualified types must survive the definition-name filter" not in red.stdout
            or "test result: FAILED" not in red.stdout
        ):
            print(red.stdout)
            raise SystemExit("mutation did not fail at the qualified-type contract")
        print("mutation: original definition-name filter turns the contract red")
    finally:
        SOURCE.write_bytes(original)
    green = run()
    if green.returncode != 0:
        print(green.stdout)
        raise SystemExit("restored qualified-type fix did not pass")
    print("mutation: restored fix turns the same contract green")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prove both fresh and cached Go usage regressions fail with the original filter."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/parser/src/extract.rs"
FIXED = b"""                if is_definition_name(node)
                    && node
                        .parent()
                        .is_none_or(|parent| parent.kind() != "qualified_type")
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
    indexer = ROOT / "crates/indexer/src/lib.rs"
    original_indexer = indexer.read_bytes()
    layer_guard = b"legacy_js_schema.is_none() || uncertified_js_files.contains(path)"
    if original_indexer.count(layer_guard) != 1:
        raise SystemExit("layer mutation target must match exactly once")
    try:
        indexer.write_bytes(original_indexer.replace(
            layer_guard, b"legacy_js_schema.is_none()", 1,
        ))
        layer = run("greppy-indexer", "base_v4_does_not_certify_unrepaired_private_js_without_go")
        if (
            layer.returncode != 101
            or "Base-v4 must not certify unvalidated private JS facts" not in layer.stdout
            or "test result: FAILED" not in layer.stdout
        ):
            print(layer.stdout)
            raise SystemExit("layer mutation did not fail at the private JS contract")
        print("mutation: trusting Base-v4 for private JS turns the layer contract red")
    finally:
        indexer.write_bytes(original_indexer)
    for package, _, _ in CASES:
        green = run(package)
        if green.returncode != 0:
            print(green.stdout)
            raise SystemExit(f"restored qualified-type contracts did not pass in {package}")
        print(f"mutation: restored fix turns {package} usage/cache contracts green")


if __name__ == "__main__":
    main()

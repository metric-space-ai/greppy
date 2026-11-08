#!/usr/bin/env python3
"""Actual pre-fix CLI warm upgrades and restored-source caller counterproofs.

Run only in an isolated source checkout through the existing Linux build lane.
Synthetic cache vectors do not qualify a product inference backend.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LEGACY_BASE = "fd8d076ac1913a18e8270826022a6be6c8f3eb6c"
SPEC = ROOT / "crates/parser/src/spec.rs"
POLICY = ROOT / "crates/parser/src/lib.rs"
INDEXER = ROOT / "crates/indexer/src/lib.rs"
PRODUCTION = [SPEC, POLICY, INDEXER]
COLD = "cpp_typed_members_keep_owner_and_reject_shadowed_or_unknown_receivers"
AMBIGUOUS = "cpp_ambiguous_field_or_payload_owner_is_not_guessed"
WARM = ["cpp_member_warm_private_repair_preserves_vectors", "cpp_member_warm_base_repair_preserves_content_reuse"]
ENV = os.environ.copy()
ENV["RUST_MIN_STACK"] = "8388608"
FEATURES = ENV.get("GREPPY_CPP_CONTRACT_FEATURES", "ci-test-assets")


def command(args):
    return subprocess.run(args, cwd=ROOT, env=ENV, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)


def checked(args):
    result = command(args)
    assert result.returncode == 0, result.stdout
    return result.stdout


def run(warm=False, red=None, needle=None):
    names = WARM if warm else [COLD, AMBIGUOUS]
    result = command(["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy", "--features", FEATURES,
                      "--test", "cpp_typed_member_callers", *(["cpp_member_warm_"] if warm else []),
                      "--", "--test-threads=1", *(["--ignored"] if warm else [])])
    expected = " ... FAILED" if red else " ... ok"
    assert result.returncode == (101 if red else 0) and all(name + expected in result.stdout for name in names), result.stdout
    if red:
        assert red in result.stdout and (not needle or needle in result.stdout), result.stdout


if subprocess.run(["git", "cat-file", "-e", LEGACY_BASE + "^{commit}"], cwd=ROOT,
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
    checked(["git", "fetch", "--no-tags", "--depth=1", "origin", LEGACY_BASE])
original = {path: path.read_bytes() for path in PRODUCTION}
old = {path: subprocess.check_output(["git", "show", f"{LEGACY_BASE}:{path.relative_to(ROOT)}"], cwd=ROOT) for path in PRODUCTION}
metadata = json.loads(checked(["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"]))
target = Path(metadata["target_directory"])
legacy = target / "tmp" / "greppy-pre-cpp-member"
legacy.parent.mkdir(parents=True, exist_ok=True)
try:
    try:
        for path, data in old.items():
            path.write_bytes(data)
        assert all(path.read_bytes() == data for path, data in old.items())
        checked(["cargo", "build", "--locked", "--jobs", "2", "-p", "greppy", "--features", FEATURES, "--bin", "greppy"])
        shutil.copy2(target / "debug" / "greppy", legacy)
        print(f"mutation: actual immutable legacy CLI from {LEGACY_BASE}; binary SHA256={hashlib.sha256(legacy.read_bytes()).hexdigest()}; spec SHA256={hashlib.sha256(old[SPEC]).hexdigest()}", flush=True)
    finally:
        for path, data in original.items():
            path.write_bytes(data)
    run()
    run(warm=True)
    print("mutation: real parser/Store/public callers plus ambiguous/shadow negatives green; unchanged-source legacy private/Base normal reads repaired; stable nodes/vectors and actual content-key reuse green; no inference acceptance", flush=True)

    text = original[SPEC].decode()
    anchor = "if matches!(spec.imports, ImportStrategy::Cpp)"
    assert text.count(anchor) == 1
    try:
        SPEC.write_text(text.replace(anchor, "if false && matches!(spec.imports, ImportStrategy::Cpp)", 1))
        run(red="typed C++ member caller missing or falsely assigned", needle='"hits":[]')
        print("mutation: disabling only new C++ receiver provenance restores the original fresh, found-target empty caller result through the real stored public graph", flush=True)
    finally:
        SPEC.write_bytes(original[SPEC])

    text = original[SPEC].decode()
    anchor = "if names.contains(field) {"
    assert text.count(anchor) == 1
    try:
        SPEC.write_text(text.replace(anchor, "if false && names.contains(field) {", 1))
        run(red="typed C++ member caller missing or falsely assigned", needle="Pipeline::shadow")
        print("mutation: disabling only local/parameter shadow rejection creates a false caller; real graph negative contract red", flush=True)
    finally:
        SPEC.write_bytes(original[SPEC])

    text = original[POLICY].decode()
    anchor = "    expected == LEGACY_EXTRACTION_REVISION || revision == expected"
    assert text.count(anchor) == 1
    try:
        POLICY.write_text(text.replace(anchor, "    let _ = (expected, revision);\n    true", 1))
        run(warm=True, red="typed C++ member caller missing or falsely assigned", needle='"hits":[]')
        print("mutation: bypassing only the existing per-file revision comparison makes BOTH actual legacy private/Base normal reads retain the missing caller without source/HEAD changes", flush=True)
    finally:
        POLICY.write_bytes(original[POLICY])

    run()
    run(warm=True)
    assert all(path.read_bytes() == data for path, data in original.items())
    print("mutation: all production sources restored byte-for-byte; cold/negative and both warm cache-reuse contracts green again", flush=True)
finally:
    for path, data in original.items():
        path.write_bytes(data)
    legacy.unlink(missing_ok=True)

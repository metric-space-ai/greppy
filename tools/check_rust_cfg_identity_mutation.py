#!/usr/bin/env python3
"""Actual legacy CLI warm upgrade plus identity/revision source counterproofs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LEGACY_BASE = "62420447f0fa82de4f30bd9e7a4eb7cd1c528136"
EXTRACT = ROOT / "crates/parser/src/extract.rs"
POLICY = ROOT / "crates/parser/src/lib.rs"
INDEXER = ROOT / "crates/indexer/src/lib.rs"
COLD = "rust_cfg_siblings_survive_parser_store_and_public_read"
WARM = ["rust_cfg_warm_private_repair_same_source_through_normal_read", "rust_cfg_warm_base_repair_same_source_through_normal_read"]
ENV = os.environ.copy()
ENV["RUST_MIN_STACK"] = "8388608"


def command(args):
    return subprocess.run(args, cwd=ROOT, env=ENV, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)


def checked(args):
    result = command(args)
    assert result.returncode == 0, result.stdout
    return result.stdout


def read_sources(paths):
    return {path: path.read_bytes() if path.exists() else None for path in paths}


def write_sources(sources):
    for path, data in sources.items():
        if data is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def legacy_sources(paths):
    tracked = set(checked(["git", "ls-tree", "-r", "--name-only", LEGACY_BASE, "--", "crates"]).splitlines())
    return {path: subprocess.check_output(
        ["git", "show", f"{LEGACY_BASE}:{path.relative_to(ROOT)}"], cwd=ROOT)
        if path.relative_to(ROOT).as_posix() in tracked else None for path in paths}


def run(warm=False, red=None):
    names = WARM if warm else [COLD]
    result = command(["cargo", "test", "--locked", "--jobs", "2", "-p", "greppy", "--features", "ci-test-assets", "--test", "rust_cfg_identity", "rust_cfg_warm_" if warm else COLD, "--", "--test-threads=1", *(["--ignored"] if warm else [])])
    expected = " ... FAILED" if red else " ... ok"
    assert result.returncode == (101 if red else 0) and all(name + expected in result.stdout for name in names) and (not red or red in result.stdout), result.stdout


if subprocess.run(["git", "cat-file", "-e", LEGACY_BASE + "^{commit}"], cwd=ROOT,
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
    checked(["git", "fetch", "--no-tags", "--depth=1", "origin", LEGACY_BASE])
changed = checked(["git", "diff", "--no-renames", "--name-only", LEGACY_BASE, "--", "crates"]).splitlines()
production = [ROOT / name for name in changed if "/src/" in name and name.endswith(".rs")]
assert EXTRACT in production and POLICY in production
original = read_sources(production)
old = legacy_sources(production)
metadata = json.loads(checked(["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"]))
target = Path(metadata["target_directory"])
legacy = target / "tmp" / "greppy-pre-rust-cfg"
legacy.parent.mkdir(parents=True, exist_ok=True)
try:
    try:
        write_sources(old)
        assert read_sources(production) == old, "legacy production source snapshot mismatch"
        checked(["cargo", "build", "--locked", "--jobs", "2", "-p", "greppy", "--features", "ci-test-assets", "--bin", "greppy"])
        shutil.copy2(target / "debug" / "greppy", legacy)
        print(f"mutation: actual legacy CLI built from exact {LEGACY_BASE} production sources; binary SHA256={hashlib.sha256(legacy.read_bytes()).hexdigest()}; extractor SHA256={hashlib.sha256(old[EXTRACT]).hexdigest()}", flush=True)
    finally:
        write_sources(original)

    run()
    run(warm=True)
    print("mutation: parser/Store/public-read siblings and selectors, helper-edge owners, actual legacy private/Base warm publication, stable nodes/vectors and real fixture-vector content reuse green; no GPU/model claim", flush=True)

    text = original[EXTRACT].decode()
    anchor = "let free_function_qnames = rust_free_function_qnames(source, tree.root_node(), file_path);"
    assert text.count(anchor) == 1
    try:
        EXTRACT.write_text(text.replace(anchor, "let free_function_qnames = std::collections::HashMap::new();", 1))
        run(red="cfg sibling definitions collapsed: parser=2, store=1, read_exit=0")
        print("mutation: disabling only collision identity map restores real parser2/Store1/public-read later-body loss; renamed source control stays required", flush=True)
    finally:
        EXTRACT.write_bytes(original[EXTRACT])

    text = original[INDEXER].decode()
    anchor = 'if referrer_file.ends_with(".rs")\n            && local'
    assert text.count(anchor) == 1
    try:
        INDEXER.write_text(text.replace(anchor, 'if false && referrer_file.ends_with(".rs")\n            && local', 1))
        run(red="incoming ambiguous call guessed a platform")
        print("mutation: disabling only the same-file free-function ambiguity guard creates the false unique incoming edge; real graph/read contract red", flush=True)
    finally:
        INDEXER.write_bytes(original[INDEXER])

    text = original[POLICY].decode()
    anchor = "    expected == LEGACY_EXTRACTION_REVISION || revision == expected"
    assert text.count(anchor) == 1
    try:
        POLICY.write_text(text.replace(anchor, "    let _ = (expected, revision);\n    true", 1))
        run(warm=True, red="warm cfg revision did not repair collapsed identity")
        print("mutation: bypassing existing per-file revision comparison makes BOTH genuine legacy private and immutable-Base normal reads red without changing source/HEAD", flush=True)
    finally:
        POLICY.write_bytes(original[POLICY])

    run()
    run(warm=True)
    assert read_sources(production) == original
    print("mutation: all production sources restored byte-for-byte; cold parser/graph/read, both warm paths and stable vector reuse green again", flush=True)
finally:
    write_sources(original)
    legacy.unlink(missing_ok=True)

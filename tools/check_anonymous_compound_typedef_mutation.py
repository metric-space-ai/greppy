#!/usr/bin/env python3
"""Real legacy production CLI -> unchanged warm store -> normal read, with source counterproofs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LEGACY_BASE = "385773de86d97b541a1760c2135e62062ebbeab0"
EXTRACT = ROOT / "crates/parser/src/extract.rs"
POLICY = ROOT / "crates/parser/src/lib.rs"
VECTORS = ROOT / "crates/store/src/vector_embedding.rs"
PARSER = "anonymous_compound_typedef_keeps_alias_span_members_and_tag_collisions"
CLI = "anonymous_compound_typedef_read_returns_complete_definition"
PRIVATE = "extraction_revision_warm_private_read_preserves_other_languages_and_reuse"
OVERLAY = "extraction_revision_warm_immutable_base_read_uses_only_private_delta"
REUSE = "attached_base_content_reuse_is_exact_and_stays_hidden"
ENV = os.environ.copy()
ENV["RUST_MIN_STACK"] = "8388608"


def command(args, timeout=600):
    return subprocess.run(args, cwd=ROOT, env=ENV, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


def checked(args):
    result = command(args)
    if result.returncode:
        raise AssertionError(result.stdout)
    return result.stdout


def run(kind, red=None):
    cases = {
        "parser": (["-p", "greppy-parser", "--lib", PARSER], [PARSER]),
        "cli": (["-p", "greppy", "--features", "ci-test-assets", "--test", "read_file_dispatch", CLI], [CLI]),
        "warm": (["-p", "greppy", "--features", "ci-test-assets", "--test", "extraction_revision", "extraction_revision_warm_"], [PRIVATE, OVERLAY]),
        "reuse": (["-p", "greppy-store", "--lib", REUSE], [REUSE]),
    }
    args, names = cases[kind]
    tail = ["--", "--test-threads=1"] + (["--ignored"] if kind == "warm" else [])
    result = command(["cargo", "test", "--locked", "--jobs", "2", *args, *tail])
    if red:
        valid = (result.returncode == 101 and all(name + " ... FAILED" in result.stdout for name in names)
                 and red in result.stdout)
    else:
        valid = result.returncode == 0 and all(name + " ... ok" in result.stdout for name in names)
    if not valid:
        raise AssertionError(result.stdout)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Replace EVERY changed production Rust source by its exact immutable Main
# bytes, not a parser imitation or forged store. Tests/helpers are not compiled
# into `cargo build --bin greppy`. Restore all sources before the current build.
if subprocess.run(["git", "cat-file", "-e", LEGACY_BASE + "^{commit}"], cwd=ROOT,
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
    checked(["git", "fetch", "--no-tags", "--depth=1", "origin", LEGACY_BASE])
changed = checked(["git", "diff", "--name-only", LEGACY_BASE, "--", "crates"]).splitlines()
production = [ROOT / name for name in changed if "/src/" in name and name.endswith(".rs")]
assert EXTRACT in production and POLICY in production
original = {path: path.read_bytes() for path in production}
legacy_bytes = {path: subprocess.check_output(
    ["git", "show", f"{LEGACY_BASE}:{path.relative_to(ROOT)}"], cwd=ROOT) for path in production}
assert b"fn c_cpp_anonymous_compound_typedef" not in legacy_bytes[EXTRACT]
metadata = json.loads(checked(["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"]))
target = Path(metadata["target_directory"])
legacy = target / "tmp" / "greppy-pre-typedef"
legacy.parent.mkdir(parents=True, exist_ok=True)
try:
    try:
        for path, data in legacy_bytes.items():
            path.write_bytes(data)
        checked(["git", "diff", "--exit-code", LEGACY_BASE, "--", *[str(p.relative_to(ROOT)) for p in production]])
        checked(["cargo", "build", "--locked", "--jobs", "2", "-p", "greppy", "--features", "ci-test-assets", "--bin", "greppy"])
        shutil.copy2(target / "debug" / "greppy", legacy)
        print(f"mutation: actual legacy CLI built from exact {LEGACY_BASE} production sources; binary SHA256={digest(legacy)}; extractor SHA256={hashlib.sha256(legacy_bytes[EXTRACT]).hexdigest()}", flush=True)
    finally:
        for path, data in original.items():
            path.write_bytes(data)

    run("parser")
    run("cli")
    run("reuse")
    run("warm")
    print("mutation: exact legacy extractor seeds missing-symbol stores; unchanged sources/HEAD heal through normal read and private/immutable-Base snapshot paths; unchanged cached inputs reuse (fixture-vector contract, no GPU/model claim)", flush=True)

    text = original[EXTRACT].decode()
    anchor = "                c_cpp_anonymous_compound_typedef(source, file_path, node, result);"
    assert text.count(anchor) == 1
    try:
        EXTRACT.write_text(text.replace(anchor, "                // Mutant: original anonymous typedef omission.", 1))
        run("parser", "anonymous compound typedef alias missing")
        run("cli", "anonymous compound typedef read failed")
        print("mutation: removing actual typedef extraction makes parser and CLI read red at original omission", flush=True)
    finally:
        EXTRACT.write_bytes(original[EXTRACT])

    text = original[POLICY].decode()
    anchor = "    expected == LEGACY_EXTRACTION_REVISION || revision == expected"
    assert text.count(anchor) == 1
    try:
        POLICY.write_text(text.replace(anchor, "    let _ = (expected, revision);\n    true", 1))
        run("warm", "warm revision read did not heal the legacy omission")
        print("mutation: bypassing shared per-file revision comparison leaves actual legacy private/Base stores missing the symbol; normal warm reads red", flush=True)
    finally:
        POLICY.write_bytes(original[POLICY])

    text = original[VECTORS].decode()
    anchor = "let Some(identity) = self.overlay_reuse_identity() else {"
    assert text.count(anchor) == 1
    try:
        VECTORS.write_text(text.replace(anchor, "let Some(identity) = None::<&crate::BaseStoreIdentity> else {", 1))
        run("reuse", "attached Base content reuse missing")
        print("mutation: disabling only attached-Base content fallback makes exact hidden-input reuse red", flush=True)
    finally:
        VECTORS.write_bytes(original[VECTORS])

    run("parser")
    run("cli")
    run("reuse")
    run("warm")
    assert all(path.read_bytes() == data for path, data in original.items())
    print("mutation: every production source restored byte-for-byte; parser/CLI, true warm upgrade, exact Base-key/title/format guards and cache-reuse contracts green", flush=True)
finally:
    for path, data in original.items():
        path.write_bytes(data)
    legacy.unlink(missing_ok=True)

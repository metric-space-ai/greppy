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


# Reproduce each changed production source's exact immutable bytes OR absence.
# Later additions must not survive in the legacy build; removed legacy files
# must be restored too. Tests/helpers are not part of `cargo build --bin greppy`.
if subprocess.run(["git", "cat-file", "-e", LEGACY_BASE + "^{commit}"], cwd=ROOT,
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
    checked(["git", "fetch", "--no-tags", "--depth=1", "origin", LEGACY_BASE])
changed = checked(["git", "diff", "--no-renames", "--name-only", LEGACY_BASE, "--", "crates"]).splitlines()
production = [ROOT / name for name in changed if "/src/" in name and name.endswith(".rs")]
assert EXTRACT in production and POLICY in production
original = read_sources(production)
legacy_bytes = legacy_sources(production)
assert b"fn c_cpp_anonymous_compound_typedef" not in legacy_bytes[EXTRACT]
metadata = json.loads(checked(["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"]))
target = Path(metadata["target_directory"])
legacy = target / "tmp" / "greppy-pre-typedef"
legacy.parent.mkdir(parents=True, exist_ok=True)
try:
    try:
        write_sources(legacy_bytes)
        assert read_sources(production) == legacy_bytes, "legacy production source snapshot mismatch"
        checked(["cargo", "build", "--locked", "--jobs", "2", "-p", "greppy", "--features", "ci-test-assets", "--bin", "greppy"])
        shutil.copy2(target / "debug" / "greppy", legacy)
        print(f"mutation: actual legacy CLI built from exact {LEGACY_BASE} production sources; binary SHA256={digest(legacy)}; extractor SHA256={hashlib.sha256(legacy_bytes[EXTRACT]).hexdigest()}", flush=True)
    finally:
        write_sources(original)

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

    role = "    anonymous_aliases.contains(&(node.start_byte(), node.end_byte()))"
    assert text.count(role) == 1
    try:
        EXTRACT.write_text(text.replace(role, "    let _ = (node, anonymous_aliases);\n    false", 1))
        run("parser", "anonymous alias declarator became a reference")
        print("mutation: disabling shared anonymous-alias declaration role creates the false self-usage; actual parser contract red while real outer and tagged references stay required", flush=True)
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
    assert read_sources(production) == original
    print("mutation: every production source restored byte-for-byte; parser/CLI, true warm upgrade, exact Base-key/title/format guards and cache-reuse contracts green", flush=True)
finally:
    write_sources(original)
    legacy.unlink(missing_ok=True)

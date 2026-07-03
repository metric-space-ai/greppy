#!/usr/bin/env python3
"""Deterministic corpus generator for the agent-efficiency benchmark.

Produces several git-initialised repos of VARIED SIZE and LANGUAGE under
``corpus/`` that grepplus can index into a real cross-file call graph:

    rust_medium    Rust       ~80 files    full cross-file CALLS graph
    python_large   Python     ~420 files   full cross-file CALLS graph
    go_small       Go         ~16 files    full cross-file CALLS graph
    java_medium    Java       ~84 files    full cross-file CALLS graph
    js_small       JavaScript ~16 files    same-file CALLS + IMPORTS
    ts_large       TypeScript ~410 files   same-file CALLS + IMPORTS

The generator is fully deterministic: identical inputs produce byte-identical
files, so corpus + ground truth + benchmark are reproducible.  No randomness,
no timestamps in content.

Each repo is layered ``core -> service -> app`` so that:
  * functions call across files (real CALLS edges for the graph),
  * modules import each other (real IMPORTS edges),
  * a few "hub" symbols are called from many sites (good who-calls answers),
  * a small data-flow chain exists end to end (good research/trace answers).

The shapes are chosen to match what each language's grepplus extractor
actually resolves (verified empirically): Rust/Python/Go/Java resolve
cross-file CALLS; JS/TS resolve same-file CALLS plus cross-file IMPORTS.
"""
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = HERE / "corpus"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Normalise: exactly one trailing newline, deterministic content.
    path.write_text(text.rstrip("\n") + "\n", encoding="utf-8")


def git_init(repo: pathlib.Path) -> None:
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="corpus-gen",
        GIT_AUTHOR_EMAIL="corpus@example.invalid",
        GIT_COMMITTER_NAME="corpus-gen",
        GIT_COMMITTER_EMAIL="corpus@example.invalid",
        GIT_AUTHOR_DATE="2020-01-01T00:00:00Z",
        GIT_COMMITTER_DATE="2020-01-01T00:00:00Z",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-q", "-m", "generated corpus"],
        cwd=repo,
        check=True,
        env=env,
    )


# --------------------------------------------------------------------------
# RUST  (medium, ~80 files, full cross-file CALLS graph)
# --------------------------------------------------------------------------
def gen_rust(root: pathlib.Path, n_services: int) -> None:
    """Layered Rust crate.

    core/checksum.rs   : leaf hub  `compute_checksum` (called everywhere)
    core/clampval.rs   : leaf hub  `clamp_value`
    core/normalize.rs  : `normalize_record` -> calls compute_checksum + clamp_value
    service/svcN.rs    : `process_svcN` -> calls normalize_record (+ checksum)
    app/pipeline.rs    : `run_pipeline` -> calls every process_svcN
    """
    src = root / "src"
    # core leaves
    write(
        src / "core" / "checksum.rs",
        '''//! Checksum primitives shared across every service layer.

/// Fold a byte slice into a 64-bit rolling checksum. This is the single
/// most-called leaf in the crate; every record passes through it.
pub fn compute_checksum(bytes: &[u8]) -> u64 {
    let mut acc: u64 = 1469598103934665603;
    for b in bytes {
        acc ^= *b as u64;
        acc = acc.wrapping_mul(1099511628211);
    }
    acc
}

/// Combine two checksums into one (associative merge).
pub fn merge_checksums(a: u64, b: u64) -> u64 {
    compute_checksum(&[(a >> 8) as u8, b as u8]) ^ a ^ b
}
''',
    )
    write(
        src / "core" / "clampval.rs",
        '''//! Numeric clamping helpers used by normalisation.

/// Clamp `v` into the inclusive range `[lo, hi]`.
pub fn clamp_value(v: i64, lo: i64, hi: i64) -> i64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}
''',
    )
    write(
        src / "core" / "normalize.rs",
        '''//! Record normalisation: the junction between raw input and services.
use crate::core::checksum::compute_checksum;
use crate::core::clampval::clamp_value;

/// A normalised record: a clamped score plus a content checksum.
pub struct Record {
    pub score: i64,
    pub checksum: u64,
}

/// Normalise a raw `(score, payload)` pair into a [`Record`]. Calls both
/// core leaves, so it is the canonical mid-layer hub.
pub fn normalize_record(score: i64, payload: &[u8]) -> Record {
    let score = clamp_value(score, 0, 100);
    let checksum = compute_checksum(payload);
    Record { score, checksum }
}
''',
    )
    write(
        src / "core" / "mod.rs",
        "pub mod checksum;\npub mod clampval;\npub mod normalize;\n",
    )

    # services
    svc_mods = []
    for i in range(n_services):
        svc_mods.append(f"svc{i:02d}")
        write(
            src / "service" / f"svc{i:02d}.rs",
            f'''//! Service {i:02d}: validates and normalises one record kind.
use crate::core::checksum::compute_checksum;
use crate::core::normalize::{{normalize_record, Record}};

/// Process one payload through service {i:02d}. Calls `normalize_record`
/// (which fans out to the core leaves) and re-checksums the result.
pub fn process_svc{i:02d}(score: i64, payload: &[u8]) -> Record {{
    let rec = normalize_record(score, payload);
    let _double = compute_checksum(&rec.checksum.to_le_bytes());
    rec
}}
''',
        )
    write(
        src / "service" / "mod.rs",
        "".join(f"pub mod {m};\n" for m in svc_mods),
    )

    # app
    calls = "\n".join(
        f"    total ^= crate::service::svc{i:02d}::process_svc{i:02d}("
        f"i as i64, &[i as u8]).checksum;"
        for i in range(n_services)
    )
    write(
        src / "app" / "pipeline.rs",
        f'''//! Application pipeline: drives every service in turn.
use crate::core::checksum::merge_checksums;

/// Run the full pipeline across all services and fold their checksums.
/// This is the top-level entry point and calls every `process_svcNN`.
pub fn run_pipeline(n: usize) -> u64 {{
    let mut total: u64 = 0;
    for i in 0..n {{
{calls}
    }}
    merge_checksums(total, n as u64)
}}
''',
    )
    write(src / "app" / "mod.rs", "pub mod pipeline;\n")
    write(
        src / "lib.rs",
        "pub mod app;\npub mod core;\npub mod service;\n",
    )
    write(
        src / "main.rs",
        '''use probe_rust::app::pipeline::run_pipeline;

fn main() {
    println!("{}", run_pipeline(8));
}
''',
    )
    write(
        root / "Cargo.toml",
        '''[package]
name = "probe_rust"
version = "0.0.0"
edition = "2021"

[[bin]]
name = "probe_rust"
path = "src/main.rs"
''',
    )
    write(root / "README.md", "# rust_medium corpus repo (generated)\n")


# --------------------------------------------------------------------------
# PYTHON  (large, ~420 files, full cross-file CALLS graph)
# --------------------------------------------------------------------------
def gen_python(root: pathlib.Path, n_services: int) -> None:
    pkg = root / "app"
    write(
        pkg / "core" / "validate.py",
        '''"""Core validation primitives (leaf hub)."""


def validate_amount(amount):
    """Return True when ``amount`` is a finite, non-negative number."""
    return amount is not None and amount >= 0


def validate_currency(code):
    """Return True when ``code`` is a 3-letter currency code."""
    return isinstance(code, str) and len(code) == 3
''',
    )
    write(
        pkg / "core" / "money.py",
        '''"""Money arithmetic built on the validators (mid leaf)."""
from app.core.validate import validate_amount, validate_currency


def to_minor_units(amount, code):
    """Convert a major-unit amount into integer minor units."""
    if not validate_amount(amount):
        raise ValueError("bad amount")
    if not validate_currency(code):
        raise ValueError("bad currency")
    return int(round(amount * 100))
''',
    )
    write(
        pkg / "core" / "ledger.py",
        '''"""The ledger junction: every service posts through here."""
from app.core.money import to_minor_units


def post_entry(amount, code, account):
    """Normalise one ledger entry into minor units keyed by account."""
    minor = to_minor_units(amount, code)
    return {"account": account, "minor": minor, "code": code}
''',
    )
    write(pkg / "core" / "__init__.py", "")

    svc_names = []
    for i in range(n_services):
        name = f"svc{i:03d}"
        svc_names.append(name)
        write(
            pkg / "service" / f"{name}.py",
            f'''"""Service {i:03d}: posts entries for one account family."""
from app.core.ledger import post_entry
from app.core.validate import validate_currency


def process_{name}(amount, code):
    """Validate and post one entry through service {i:03d}."""
    if not validate_currency(code):
        return None
    return post_entry(amount, code, "acct-{i:03d}")
''',
        )
    write(pkg / "service" / "__init__.py", "")

    # app driver calls every service
    imports = "\n".join(
        f"from app.service.{n} import process_{n}" for n in svc_names
    )
    body = "\n".join(
        f"    rows.append(process_{n}(float(i), \"USD\"))" for i, n in enumerate(svc_names)
    )
    write(
        pkg / "pipeline.py",
        f'''"""Application pipeline: drives every service (top-level entry)."""
{imports}


def run_pipeline(i):
    """Run every service once and collect the posted ledger rows."""
    rows = []
{body}
    return [r for r in rows if r is not None]
''',
    )
    write(pkg / "__init__.py", "")
    write(
        root / "main.py",
        '''"""Entry script for the python_large corpus."""
from app.pipeline import run_pipeline


def main():
    rows = run_pipeline(1)
    print(len(rows))


if __name__ == "__main__":
    main()
''',
    )
    write(root / "README.md", "# python_large corpus repo (generated)\n")
    write(root / "pyproject.toml", '[project]\nname = "probe_python"\nversion = "0.0.0"\n')


# --------------------------------------------------------------------------
# GO  (small, ~16 files, full cross-file CALLS graph)
# --------------------------------------------------------------------------
def gen_go(root: pathlib.Path, n_services: int) -> None:
    write(
        root / "core" / "hash.go",
        '''package corpus

// ComputeHash folds a byte slice into a 32-bit hash. Leaf hub.
func ComputeHash(data []byte) uint32 {
	var h uint32 = 2166136261
	for _, b := range data {
		h ^= uint32(b)
		h *= 16777619
	}
	return h
}
''',
    )
    write(
        root / "core" / "clamp.go",
        '''package corpus

// ClampInt clamps v into [lo, hi]. Leaf hub.
func ClampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
''',
    )
    write(
        root / "core" / "normalize.go",
        '''package corpus

// Record is a normalised score plus content hash.
type Record struct {
	Score int
	Hash  uint32
}

// NormalizeRecord is the mid-layer junction: it calls both leaves.
func NormalizeRecord(score int, payload []byte) Record {
	return Record{Score: ClampInt(score, 0, 100), Hash: ComputeHash(payload)}
}
''',
    )
    for i in range(n_services):
        write(
            root / "service" / f"svc{i:02d}.go",
            f'''package corpus

// ProcessSvc{i:02d} runs one record through service {i:02d}.
func ProcessSvc{i:02d}(score int, payload []byte) Record {{
	rec := NormalizeRecord(score, payload)
	_ = ComputeHash(payload)
	return rec
}}
''',
        )
    calls = "\n".join(
        f"\ttotal ^= ProcessSvc{i:02d}(i, []byte{{byte(i)}}).Hash"
        for i in range(n_services)
    )
    write(
        root / "app" / "pipeline.go",
        f'''package corpus

// RunPipeline drives every service and folds their hashes. Top entry.
func RunPipeline(n int) uint32 {{
	var total uint32
	for i := 0; i < n; i++ {{
{calls}
	}}
	return total
}}
''',
    )
    write(
        root / "main.go",
        '''package main

import "fmt"

func main() {
	fmt.Println("corpus")
}
''',
    )
    write(root / "go.mod", "module probe_go\n\ngo 1.21\n")
    write(root / "README.md", "# go_small corpus repo (generated)\n")


# --------------------------------------------------------------------------
# JAVA  (medium, ~84 files, full cross-file CALLS graph)
# --------------------------------------------------------------------------
def gen_java(root: pathlib.Path, n_services: int) -> None:
    base = root / "src" / "main" / "java" / "corpus"
    write(
        base / "core" / "Checksum.java",
        '''package corpus.core;

/** Checksum primitives (leaf hub). */
public final class Checksum {
    private Checksum() {}

    /** Fold a byte array into a 64-bit checksum. */
    public static long computeChecksum(byte[] data) {
        long acc = 1469598103934665603L;
        for (byte b : data) {
            acc ^= (b & 0xff);
            acc *= 1099511628211L;
        }
        return acc;
    }
}
''',
    )
    write(
        base / "core" / "Clamp.java",
        '''package corpus.core;

/** Numeric clamping (leaf hub). */
public final class Clamp {
    private Clamp() {}

    /** Clamp v into [lo, hi]. */
    public static int clampValue(int v, int lo, int hi) {
        if (v < lo) {
            return lo;
        }
        if (v > hi) {
            return hi;
        }
        return v;
    }
}
''',
    )
    write(
        base / "core" / "Normalizer.java",
        '''package corpus.core;

/** Record normalisation junction: calls both core leaves. */
public final class Normalizer {
    public final int score;
    public final long checksum;

    private Normalizer(int score, long checksum) {
        this.score = score;
        this.checksum = checksum;
    }

    /** Normalise one record. */
    public static Normalizer normalizeRecord(int score, byte[] payload) {
        int s = Clamp.clampValue(score, 0, 100);
        long c = Checksum.computeChecksum(payload);
        return new Normalizer(s, c);
    }
}
''',
    )
    for i in range(n_services):
        write(
            base / "service" / f"Svc{i:02d}.java",
            f'''package corpus.service;

import corpus.core.Checksum;
import corpus.core.Normalizer;

/** Service {i:02d}. */
public final class Svc{i:02d} {{
    private Svc{i:02d}() {{}}

    /** Process one record through service {i:02d}. */
    public static Normalizer processSvc{i:02d}(int score, byte[] payload) {{
        Normalizer rec = Normalizer.normalizeRecord(score, payload);
        long ignore = Checksum.computeChecksum(payload);
        return rec;
    }}
}}
''',
        )
    calls = "\n".join(
        f"            total ^= corpus.service.Svc{i:02d}.processSvc{i:02d}"
        f"(i, new byte[]{{(byte) i}}).checksum;"
        for i in range(n_services)
    )
    write(
        base / "app" / "Pipeline.java",
        f'''package corpus.app;

/** Application pipeline: drives every service. Top-level entry. */
public final class Pipeline {{
    private Pipeline() {{}}

    /** Run the full pipeline and fold every service checksum. */
    public static long runPipeline(int n) {{
        long total = 0;
        for (int i = 0; i < n; i++) {{
{calls}
        }}
        return total;
    }}

    public static void main(String[] args) {{
        System.out.println(runPipeline(8));
    }}
}}
''',
    )
    write(root / "README.md", "# java_medium corpus repo (generated)\n")


# --------------------------------------------------------------------------
# JAVASCRIPT  (small, ~16 files; same-file CALLS + cross-file IMPORTS)
# --------------------------------------------------------------------------
def gen_js(root: pathlib.Path, n_services: int) -> None:
    src = root / "src"
    write(
        src / "core" / "checksum.js",
        '''// Checksum primitives. `mergeChecksums` calls `computeChecksum` in-file.
function computeChecksum(bytes) {
  let acc = 0x811c9dc5;
  for (const b of bytes) {
    acc ^= b;
    acc = Math.imul(acc, 0x01000193) >>> 0;
  }
  return acc;
}

function mergeChecksums(a, b) {
  // same-file call -> resolvable CALLS edge
  const seed = computeChecksum([a & 0xff, b & 0xff]);
  return (seed ^ a ^ b) >>> 0;
}

module.exports = { computeChecksum, mergeChecksums };
''',
    )
    write(
        src / "core" / "clamp.js",
        '''// Clamp helper. `clampScore` calls `clampValue` in-file.
function clampValue(v, lo, hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

function clampScore(v) {
  return clampValue(v, 0, 100);
}

module.exports = { clampValue, clampScore };
''',
    )
    write(
        src / "core" / "normalize.js",
        '''const { computeChecksum } = require('./checksum.js');
const { clampScore } = require('./clamp.js');

// `normalizeRecord` calls `buildRecord` in-file (resolvable CALLS edge).
function buildRecord(score, checksum) {
  return { score, checksum };
}

function normalizeRecord(score, payload) {
  const c = computeChecksum(payload);
  const s = clampScore(score);
  return buildRecord(s, c);
}

module.exports = { normalizeRecord, buildRecord };
''',
    )
    for i in range(n_services):
        write(
            src / "service" / f"svc{i:02d}.js",
            f'''const {{ normalizeRecord }} = require('../core/normalize.js');

// service {i:02d}; `processSvc{i:02d}` calls `tagSvc{i:02d}` in-file.
function tagSvc{i:02d}(rec) {{
  return {{ ...rec, svc: {i} }};
}}

function processSvc{i:02d}(score, payload) {{
  const rec = normalizeRecord(score, payload);
  return tagSvc{i:02d}(rec);
}}

module.exports = {{ processSvc{i:02d}, tagSvc{i:02d} }};
''',
        )
    requires = "\n".join(
        f"const {{ processSvc{i:02d} }} = require('./service/svc{i:02d}.js');"
        for i in range(n_services)
    )
    body = "\n".join(
        f"  rows.push(processSvc{i:02d}(i, [i & 0xff]));" for i in range(n_services)
    )
    write(
        src / "pipeline.js",
        f'''{requires}

// `runPipeline` calls `collect` in-file (resolvable CALLS edge).
function collect(rows) {{
  return rows.filter((r) => r);
}}

function runPipeline(i) {{
  const rows = [];
{body}
  return collect(rows);
}}

module.exports = {{ runPipeline, collect }};
''',
    )
    write(
        root / "index.js",
        '''const { runPipeline } = require('./src/pipeline.js');
console.log(runPipeline(1).length);
''',
    )
    write(
        root / "package.json",
        '{\n  "name": "probe_js",\n  "version": "0.0.0",\n  "private": true\n}\n',
    )
    write(root / "README.md", "# js_small corpus repo (generated)\n")


# --------------------------------------------------------------------------
# TYPESCRIPT  (large, ~410 files; same-file CALLS + cross-file IMPORTS)
# --------------------------------------------------------------------------
def gen_ts(root: pathlib.Path, n_services: int) -> None:
    src = root / "src"
    write(
        src / "core" / "validate.ts",
        '''// Core validators. `validateRecord` calls both validators in-file.
export function validateAmount(amount: number): boolean {
  return Number.isFinite(amount) && amount >= 0;
}

export function validateCode(code: string): boolean {
  return code.length === 3;
}

export function validateRecord(amount: number, code: string): boolean {
  return validateAmount(amount) && validateCode(code);
}
''',
    )
    write(
        src / "core" / "money.ts",
        '''import { validateRecord } from './validate';

// `toMinorUnits` calls `roundMinor` in-file (resolvable CALLS edge).
function roundMinor(amount: number): number {
  return Math.round(amount * 100);
}

export function toMinorUnits(amount: number, code: string): number {
  if (!validateRecord(amount, code)) {
    throw new Error('invalid');
  }
  return roundMinor(amount);
}
''',
    )
    write(
        src / "core" / "ledger.ts",
        '''import { toMinorUnits } from './money';

export interface Entry {
  account: string;
  minor: number;
}

// `postEntry` calls `makeEntry` in-file.
function makeEntry(account: string, minor: number): Entry {
  return { account, minor };
}

export function postEntry(amount: number, code: string, account: string): Entry {
  return makeEntry(account, toMinorUnits(amount, code));
}
''',
    )
    for i in range(n_services):
        write(
            src / "service" / f"svc{i:03d}.ts",
            f'''import {{ postEntry }} from '../core/ledger';

// service {i:03d}; `processSvc{i:03d}` calls `accountSvc{i:03d}` in-file.
function accountSvc{i:03d}(): string {{
  return 'acct-{i:03d}';
}}

export function processSvc{i:03d}(amount: number, code: string) {{
  return postEntry(amount, code, accountSvc{i:03d}());
}}
''',
        )
    imports = "\n".join(
        f"import {{ processSvc{i:03d} }} from './service/svc{i:03d}';"
        for i in range(n_services)
    )
    body = "\n".join(
        f"  rows.push(processSvc{i:03d}(i, 'USD'));" for i in range(n_services)
    )
    write(
        src / "pipeline.ts",
        f'''{imports}
import {{ Entry }} from './core/ledger';

// `runPipeline` calls `summarize` in-file.
function summarize(rows: Entry[]): number {{
  return rows.length;
}}

export function runPipeline(i: number): number {{
  const rows: Entry[] = [];
{body}
  return summarize(rows);
}}
''',
    )
    write(
        root / "index.ts",
        '''import { runPipeline } from './src/pipeline';
console.log(runPipeline(1));
''',
    )
    write(
        root / "package.json",
        '{\n  "name": "probe_ts",\n  "version": "0.0.0",\n  "private": true\n}\n',
    )
    write(root / "README.md", "# ts_large corpus repo (generated)\n")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
# (repo dir, language label, size label, generator, n_services)
REPOS = [
    ("rust_medium", "rust", "medium", gen_rust, 72),
    ("python_large", "python", "large", gen_python, 415),
    ("go_small", "go", "small", gen_go, 9),
    ("java_medium", "java", "medium", gen_java, 76),
    ("js_small", "javascript", "small", gen_js, 8),
    ("ts_large", "typescript", "large", gen_ts, 405),
]


def exclude_from_outer_repo() -> None:
    """Keep the generated corpus out of the OUTER grepplus repo via
    ``.git/info/exclude`` (local, untracked) rather than a tracked
    ``.gitignore corpus/`` entry.

    Why not ``.gitignore``: the ``ignore`` crate that powers grepplus's file
    walker reads parent-directory gitignores, so a ``corpus/`` pattern in a
    committed ``.gitignore`` makes grepplus SKIP every file inside the corpus
    repos when indexing them — silently breaking the benchmark. ``.git/info/
    exclude`` is honored by ``git`` (so the embedded repos don't pollute the
    outer status / create broken gitlinks) but is NOT applied across the nested
    repo boundary by the walker, because each corpus repo has its own ``.git``
    which is the ceiling for ignore traversal. Verified empirically.
    """
    # locate the outer repo's .git directory (walk up from CORPUS)
    cur = CORPUS.resolve()
    git_dir = None
    for parent in [cur, *cur.parents]:
        cand = parent / ".git"
        if cand.is_dir():
            git_dir = cand
            break
    if git_dir is None:
        return
    info = git_dir / "info"
    info.mkdir(parents=True, exist_ok=True)
    exclude = info / "exclude"
    rel = CORPUS.resolve().relative_to(git_dir.parent)
    line = f"{rel.as_posix()}/"
    existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    if line not in existing.splitlines():
        with exclude.open("a", encoding="utf-8") as fh:
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            fh.write(f"# generated agent-efficiency corpus (see gen_corpus.sh)\n{line}\n")


def main() -> int:
    if CORPUS.exists():
        shutil.rmtree(CORPUS)
    CORPUS.mkdir(parents=True)
    for name, lang, size, fn, n in REPOS:
        repo = CORPUS / name
        repo.mkdir(parents=True)
        fn(repo, n)
        git_init(repo)
        nfiles = sum(1 for _ in repo.rglob("*") if _.is_file() and ".git" not in _.parts)
        print(f"  {name:14s} {lang:11s} {size:7s} {nfiles:4d} files")
    exclude_from_outer_repo()
    print(f"corpus written to {CORPUS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

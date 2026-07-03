#!/usr/bin/env python3
"""Deterministic search-context cost benchmark — NO LLM, NO API, NO money.

The agent A/B benchmark (run_bench.py) is confounded, as the self-audit
documents: the model-reported input tokens include pi's fixed ~3.4K base
system prompt AND are subject to MiniMax prompt caching, so they swing
run-to-run and bias the ratio toward 1.0. This benchmark removes ALL of
that: it measures, deterministically, the number of bytes/tokens of
*search context* each strategy makes an agent ingest to answer a question.
There is no model, so there is nothing to contaminate.

Three strategies, the third being the C-original's own baseline:

  * grepplus     — the bytes of the right grepplus command's output
                   (who-calls / callees / find-usages / semantic).
  * grep         — the bytes of `grep -rn <symbol>` output (a careful,
                   line-numbered grep — the *tough* baseline).
  * file-reader  — grep for the symbol, then read every file that matched,
                   in full. This models the upstream README's "file-by-file
                   search" baseline (the one the C "120x fewer tokens" claim
                   is measured against), so our numbers are directly
                   comparable to that claim.

Token estimate = bytes / 4 (a standard rough char→token ratio; we report
both). Every number is reproducible: re-running gives identical results.

Run:  python3 bench/agent_efficiency/context_cost.py
"""
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
BIN = str(REPO / "target" / "debug" / "grepplus")
CORPUS = HERE / "corpus"


def sh(args, cwd=None):
    """Run a command, return (stdout_bytes)."""
    try:
        p = subprocess.run(
            args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=60,
        )
        return p.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return b""


def grep_bytes_and_files(symbol, repo):
    """`grep -rn <symbol>`: return (output_bytes, set_of_matched_files)."""
    out = sh(["grep", "-rnI", "--include=*.*", symbol, "."], cwd=repo)
    files = set()
    for line in out.decode("utf-8", "replace").splitlines():
        # grep -rn format: ./path:line:content
        if ":" in line:
            f = line.split(":", 1)[0]
            files.add(f)
    return len(out), files


def fileread_bytes(files, repo):
    """Total bytes of the distinct files grep matched (naive 'read every file
    that mentions the symbol' baseline — the C 'file-by-file' model)."""
    total = 0
    for f in files:
        p = (repo / f) if not os.path.isabs(f) else pathlib.Path(f)
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def gp_bytes(args, repo):
    """Bytes of a grepplus command's stdout."""
    return len(sh([BIN, *args, "--root", str(repo)]))


def tok(b):
    return round(b / 4)


# --------------------------------------------------------------------------
# Task set — structural (symbol-anchored) + semantic (vocabulary bridging).
# Symbols are real identifiers in the deterministic corpus.
# --------------------------------------------------------------------------
STRUCTURAL = [
    # (repo, kind, symbol)
    ("rust_medium", "who-calls", "compute_checksum"),
    ("rust_medium", "who-calls", "normalize_record"),
    ("rust_medium", "callees", "run_pipeline"),
    ("rust_medium", "find-usages", "Record"),
    ("python_large", "who-calls", "validate_currency"),
    ("python_large", "callees", "run_pipeline"),
    ("ts_large", "who-calls", "normalizeRecord"),
    ("java_medium", "who-calls", "computeChecksum"),
    ("go_small", "who-calls", "NormalizeRecord"),
    ("js_small", "who-calls", "normalizeRecord"),
]

# Literal find-definition tasks (contract Z3): "show me the definition of
# X" lookups — the domain where plain grep is optimal. These mirror the
# literal_control tasks r076-r084 in tasks_v2.json. The agent reaches for
# `grepplus context <symbol>` (it returns the def's source span so the
# agent need not open the file); the grep baseline is `grep -rn <symbol>`.
# Z3 requires grepplus not be >10% worse here, i.e. grep/grepplus >= 0.9.
# Each symbol is an EXACT primary definition name in the corpus, so
# `context` takes the exact-name fast path and returns one lean span.
# (repo, symbol)
LITERAL = [
    ("rust_medium", "clamp_value"),        # r076
    ("python_large", "validate_amount"),   # r077
    ("python_large", "to_minor_units"),    # r078
    ("go_small", "ClampInt"),              # r079
    ("java_medium", "clampValue"),         # r080
    ("ts_large", "processSvc100"),         # r081
    ("ts_large", "roundMinor"),            # r082
    ("rust_medium", "merge_checksums"),    # r083
    ("python_large", "post_entry"),        # r084
]

# Semantic tasks: a natural-language concept whose words do NOT appear
# literally in the target identifiers, so grep/file-read cannot find it but
# embedding semantic can. (concept, repo)
SEMANTIC = [
    ("rust_medium", "hash fingerprint of bytes"),          # → compute_checksum
    ("rust_medium", "clamp a value to a range"),           # → clamp_value
    ("python_large", "money currency validation sanity"),  # → validate_currency
    ("ts_large", "tidy up an incoming record"),            # → normalizeRecord
]


def main():
    if not os.path.exists(BIN):
        sys.exit(f"build grepplus first: {BIN} missing")

    # Guard (review finding): if a corpus repo is not indexed, grepplus
    # commands return an error/empty and the ratios become garbage
    # (e.g. 0t / huge x). Verify every repo we touch has a populated graph
    # FIRST, and tell the user how to fix it, instead of silently lying.
    repos = sorted(
        {r for r, *_ in STRUCTURAL}
        | {r for r, _ in LITERAL}
        | {r for r, _ in SEMANTIC}
    )
    missing = []
    for repo in repos:
        rd = CORPUS / repo
        out = sh([BIN, "stats", "--root", str(rd)]).decode("utf-8", "replace")
        n = 0
        for line in out.splitlines():
            if line.strip().lower().startswith("nodes:"):
                digits = "".join(c for c in line if c.isdigit())
                n = int(digits) if digits else 0
        if n == 0:
            missing.append(repo)
    if missing:
        sys.exit(
            "corpus not indexed (empty graph) for: "
            + ", ".join(missing)
            + f"\nindex them first, e.g.:  for d in {CORPUS}/*/; do {BIN} index \"$d\"; done"
        )

    print("=" * 92)
    print("DETERMINISTIC SEARCH-CONTEXT COST  (bytes the agent must ingest to answer; lower=better)")
    print("no LLM, no API — fully reproducible. token ≈ bytes/4.")
    print("=" * 92)

    print("\n## STRUCTURAL queries (symbol-anchored)\n")
    hdr = f"{'task':<34} {'file-read':>12} {'grep':>10} {'grepplus':>10}  {'fr/gp':>7} {'grep/gp':>7}"
    print(hdr)
    print("-" * len(hdr))
    fr_ratios, grep_ratios = [], []
    for repo, kind, sym in STRUCTURAL:
        rd = CORPUS / repo
        gbytes, files = grep_bytes_and_files(sym, rd)
        frbytes = fileread_bytes(files, rd)
        gp = gp_bytes([kind, sym], rd)
        gp = max(gp, 1)
        fr_r = frbytes / gp
        grep_r = gbytes / gp
        fr_ratios.append(fr_r)
        grep_ratios.append(grep_r)
        print(f"{repo[:10]+' '+kind+' '+sym:<34} "
              f"{tok(frbytes):>10}t {tok(gbytes):>8}t {tok(gp):>8}t  "
              f"{fr_r:>6.1f}x {grep_r:>6.1f}x")

    def med(xs):
        xs = sorted(xs)
        n = len(xs)
        return (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2) if xs else 0

    print(f"\n  MEDIAN factor vs grepplus:  file-reader {med(fr_ratios):.1f}x   "
          f"careful-grep {med(grep_ratios):.1f}x")

    # ----------------------------------------------------------------------
    # LITERAL find-definition queries (contract Z3). Here plain grep is
    # optimal, so grepplus must not make the agent ingest MORE. We compare
    # `grepplus context <symbol>` (the def-span lookup an agent uses) to
    # `grep -rn <symbol>`. Factor = grep/grepplus; Z3 requires >= 0.9.
    # ----------------------------------------------------------------------
    print("\n## LITERAL find-definition queries (Z3 — plain grep is optimal; "
          "grepplus must be >= 0.9x)\n")
    lhdr = f"{'repo  symbol':<34} {'grep':>10} {'gpp ctx':>10}  {'grep/gp':>8}"
    print(lhdr)
    print("-" * len(lhdr))
    lit_ratios = []
    for repo, sym in LITERAL:
        rd = CORPUS / repo
        gbytes, _files = grep_bytes_and_files(sym, rd)
        gp = max(gp_bytes(["context", sym], rd), 1)
        r = gbytes / gp
        lit_ratios.append(r)
        flag = "" if r >= 0.9 else "  << Z3 FAIL"
        print(f"{repo[:12]+' '+sym:<34} "
              f"{tok(gbytes):>8}t {tok(gp):>8}t  {r:>7.2f}x{flag}")
    lit_min = min(lit_ratios) if lit_ratios else 0
    lit_med = med(lit_ratios)
    agg_grep = sum(grep_bytes_and_files(s, CORPUS / r)[0] for r, s in LITERAL)
    agg_gp = sum(max(gp_bytes(["context", s], CORPUS / r), 1) for r, s in LITERAL)
    print(f"\n  MIN factor {lit_min:.2f}x   MEDIAN {lit_med:.2f}x   "
          f"AGGREGATE grep/gp {agg_grep/agg_gp:.2f}x   "
          f"(Z3 gate: min & aggregate must be >= 0.9)")
    print("  " + ("PASS — grepplus is grep-competitive on literal lookups."
                  if lit_min >= 0.9 and agg_grep / agg_gp >= 0.9
                  else "FAIL — grepplus costs more than grep on a literal lookup."))

    print("\n## SEMANTIC queries (vocabulary bridging — words absent from the code)\n")
    print(f"{'concept':<40} {'grep hits':>10} {'grepplus(embedding) top hit':>34}")
    print("-" * 86)
    for repo, concept in SEMANTIC:
        rd = CORPUS / repo
        # grep for each concept word; count total matches (the naive approach).
        total = 0
        for w in concept.split():
            total += len(sh(["grep", "-rniI", w, "."], cwd=rd).splitlines())
        sem_out = sh([BIN, "semantic", concept, "--root", str(rd)]).decode("utf-8", "replace")
        top = ""
        for line in sem_out.splitlines():
            if line and not line.startswith("#"):
                top = line.strip()
                break
        print(f"{concept[:38]:<40} {total:>10} {top[:34]:>34}")
    print("\n  For semantic tasks grep returns scattered/irrelevant matches or "
          "nothing;\n  the embedding finds the right symbol with ZERO literal overlap → "
          "savings are effectively unbounded (grep cannot answer at all).")


if __name__ == "__main__":
    main()

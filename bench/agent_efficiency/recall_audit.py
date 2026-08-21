#!/usr/bin/env python3
"""LLM-free recall audit: does the graph answer match rg evidence?

Development instrument (NOT a benchmark arm — only real Pi/MiniMax runs
count for published numbers). For each repo it samples symbols straight
from the store's nodes table, runs `who-calls`, and checks two invariants:

  1. RESOLUTION: every sampled node's name resolves as a graph symbol
     (guards against O8-class window/cap bugs — the store demonstrably
     has the node, so "not a graph symbol" is always a product bug).
  2. RECALL FLOOR: when rg finds call-shaped references (`name(`) in other
     files, who-calls should list at least one caller OR honestly disclose
     textual candidates. Zero callers + zero disclosure + rg evidence
     = a silent recall hole worth investigating.

Usage:
    python3 recall_audit.py --root realcorpus/django [--sample 200] [--seed 7]
    python3 recall_audit.py --all          # the 6 real repos
Exit code 1 when invariant 1 fails anywhere (hard bug), else 0.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import sqlite3
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
BIN = pathlib.Path(os.environ.get("GREPPY_BENCH_BIN")
                   or HERE.parents[1] / "target" / "release" / "greppy")
CORPUS_HOME = pathlib.Path(os.environ.get("GREPPY_BENCH_CORPUS_HOME") or HERE)
REALCORPUS = CORPUS_HOME / "realcorpus"
REAL = ["serde", "flask", "gson", "zod", "tokio", "django"]
PRIMARY = {"Function", "Method", "Struct", "Class", "Enum", "Trait", "Interface"}


def store_db_for(root: pathlib.Path) -> pathlib.Path | None:
    """Locate the store graph.db whose projects.root_path == root."""
    base = os.environ.get("GREPPY_STORE_DIR")
    candidates = []
    if base:
        candidates.append(pathlib.Path(base))
    home = pathlib.Path.home()
    candidates += [
        home / ".local/share/greppy",
        home / "Library/Application Support/greppy",
        home / ".cache/greppy",
    ]
    want = str(root.resolve())
    for c in candidates:
        if not c.is_dir():
            continue
        for db in c.glob("*/graph.db"):
            try:
                con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                roots = [r[0] for r in con.execute("SELECT root_path FROM projects")]
                con.close()
            except sqlite3.Error:
                continue
            if any(os.path.realpath(r) == os.path.realpath(want) for r in roots):
                return db
    return None


def sample_symbols(db: pathlib.Path, n: int, seed: int) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT label, name, qualified_name, file_path FROM nodes "
        "WHERE label IN ({}) AND name != '' AND name NOT LIKE '\\_%' ESCAPE '\\'"
        .format(",".join("?" * len(PRIMARY))), sorted(PRIMARY)).fetchall()
    con.close()
    rng = random.Random(seed)
    rng.shuffle(rows)
    seen: set[str] = set()
    out = []
    for label, name, qname, fp in rows:
        if name in seen:
            continue
        seen.add(name)
        out.append({"label": label, "name": name, "qname": qname, "file": fp})
        if len(out) >= n:
            break
    return out


def who_calls(root: pathlib.Path, name: str) -> str:
    p = subprocess.run(
        [str(BIN), "who-calls", name, "--root", str(root)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, text=True, timeout=120,
    )
    return p.stdout


def rg_call_evidence(root: pathlib.Path, name: str, def_file: str) -> int:
    """Files (excluding the defining file) containing a call-shaped `name(`."""
    p = subprocess.run(
        ["rg", "-l", "--fixed-strings", f"{name}(", str(root)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL, text=True,
    )
    files = [f for f in p.stdout.splitlines() if f and not f.endswith(def_file)]
    return len(files)


def audit_repo(root: pathlib.Path, n: int, seed: int) -> dict:
    db = store_db_for(root)
    if db is None:
        return {"root": str(root), "error": "no store found — index first"}
    syms = sample_symbols(db, n, seed)
    res = {"root": str(root), "sampled": len(syms), "not_graph_symbol": [],
           "silent_recall_holes": [], "with_callers": 0, "disclosed_textual": 0,
           "no_evidence": 0}
    res["dotted_name_unresolvable"] = []
    for s in syms:
        out = who_calls(root, s["name"])
        if "is not a graph symbol" in out or "symbol not found" in out:
            # A node whose NAME contains a separator ('.'/'::', e.g. TOML
            # keys like `tool.flit.module`) is parsed as an Owner.member
            # query and can never resolve to itself — a known lookup-
            # semantics edge, tracked separately from O8-class cap bugs.
            if "." in s["name"] or "::" in s["name"]:
                res["dotted_name_unresolvable"].append(s["name"])
            else:
                res["not_graph_symbol"].append(s["name"])
            continue
        has_callers = bool(re.search(r"— [1-9]\d* caller", out))
        disclosed = "name-match candidates" in out
        if has_callers:
            res["with_callers"] += 1
        elif disclosed:
            res["disclosed_textual"] += 1
        else:
            ev = rg_call_evidence(root, s["name"], s["file"])
            if ev > 0:
                res["silent_recall_holes"].append(
                    {"name": s["name"], "label": s["label"],
                     "rg_files_with_call": ev, "qname": s["qname"]})
            else:
                res["no_evidence"] += 1
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=pathlib.Path)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    roots = ([REALCORPUS / r for r in REAL] if args.all
             else [args.root] if args.root else None)
    if not roots:
        ap.error("--root or --all required")

    hard_fail = False
    reports = []
    for root in roots:
        r = audit_repo(root, args.sample, args.seed)
        reports.append(r)
        if r.get("error"):
            print(f"[audit] {root}: {r['error']}", file=sys.stderr)
            continue
        ngs = r["not_graph_symbol"]
        holes = r["silent_recall_holes"]
        print(f"[audit] {root.name}: sampled={r['sampled']} callers={r['with_callers']} "
              f"textual={r['disclosed_textual']} no-evidence={r['no_evidence']} "
              f"NOT_GRAPH_SYMBOL={len(ngs)} silent_holes={len(holes)}")
        for nm in ngs[:5]:
            print(f"    HARD BUG resolution: {nm} (store has the node, query missed it)")
        for h in holes[:5]:
            print(f"    hole: {h['name']} ({h['label']}) — rg sees calls in "
                  f"{h['rg_files_with_call']} other file(s), graph+disclosure empty")
        if ngs:
            hard_fail = True
    if args.json:
        print(json.dumps(reports, indent=1))
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())

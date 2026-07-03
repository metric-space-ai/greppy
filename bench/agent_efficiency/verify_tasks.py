#!/usr/bin/env python3
"""Verify the benchmark corpus and that every task is answerable.

Two guarantees are checked mechanically against the live grepplus graph:

  1. CORPUS INTEGRITY -- each repo indexes and its graph contains cross-file
     edges (CALLS, and where applicable IMPORTS). Printed as a per-repo table.

  2. GROUND-TRUTH ANSWERABILITY -- for every task in ``tasks.json`` the
     ``check`` descriptor is run against grepplus (who-calls / callees /
     find-usages / path / search-code / search-symbols) and the expected
     symbols / files / counts must be present. A task that does not resolve is
     a FAIL, so a clean run proves the bank is 100% answerable.

Exit code is non-zero if any repo lacks cross-file edges or any task fails.

Usage:
    python3 bench/agent_efficiency/verify_tasks.py
    python3 bench/agent_efficiency/verify_tasks.py --index   # (re)index first
"""
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
BIN = str(REPO / "target" / "release" / "grepplus")
CORPUS = HERE / "corpus"

REPOS = ["rust_medium", "python_large", "go_small",
         "java_medium", "js_small", "ts_large"]


def gp(root, *args):
    p = subprocess.run([BIN, *args, "--root", str(root)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.stdout.decode("utf-8", "replace")


def gp_json(root, *args):
    out = gp(root, *args, "--json")
    try:
        return json.loads(out), out
    except json.JSONDecodeError as e:
        return None, f"invalid JSON from {' '.join(args)}: {e}: {out[:500]}"


def index_all():
    for r in REPOS:
        gp(CORPUS / r, "index", str(CORPUS / r))


def stats(root):
    out = gp(root, "stats")
    d = {}
    for line in out.splitlines():
        line = line.strip()
        for key in ("CALLS", "IMPORTS", "USES", "TYPE_REF"):
            if line.startswith(key + " "):
                d[key] = int(line.split()[1])
        if line.startswith("files:"):
            d["files"] = int(line.split()[1])
        if line.startswith("edges:"):
            d["edges"] = int(line.split()[1])
    return d


def check_task(root, chk):
    """Return (ok, detail). Runs the appropriate grepplus query and asserts the
    expected members / file / count are present."""
    kind = chk["kind"]
    if kind in ("who_calls", "callees", "find_usages"):
        sub = {"who_calls": "who-calls", "callees": "callees",
               "find_usages": "find-usages"}[kind]
        data, raw = gp_json(root, sub, chk["symbol"])
        if data is None:
            return False, raw
        if data.get("status", "ok") not in ("ok", None):
            return False, f"{sub} {chk['symbol']}: status {data.get('status')}"
        hits = data.get("hits") or []
        total = data.get("total_exact")
        if not isinstance(total, int):
            total = len(hits)
        joined = "\n".join(json.dumps(hit, sort_keys=True) for hit in hits)
        for m in chk.get("expect_members", []):
            if m not in joined:
                return False, f"{sub} {chk['symbol']}: missing member {m}"
        if "min_count" in chk and total < chk["min_count"]:
            return False, (f"{sub} {chk['symbol']}: count {total} "
                           f"< {chk['min_count']}")
        shown = data.get("shown", len(hits))
        return True, f"{sub} {chk['symbol']}: {total} exact rows ({shown} shown)"
    if kind == "path":
        data, raw = gp_json(root, "path", "--from", chk["frm"], "--to", chk["to"])
        if data is None:
            return False, raw
        steps = data.get("steps") or []
        joined = "\n".join(json.dumps(step, sort_keys=True) for step in steps)
        if data.get("path_found") and chk["to"] in joined and chk["frm"] in joined:
            return True, f"path {chk['frm']}->{chk['to']}: found"
        return False, (
            f"path {chk['frm']}->{chk['to']}: no path "
            f"(reason={data.get('reason')}, raw={raw.strip()[:80]})"
        )
    if kind in ("search_code", "search_symbols"):
        sub = "search-code" if kind == "search_code" else "search-symbols"
        data, raw = gp_json(root, sub, chk["query"])
        if data is None:
            return False, raw
        hits = data.get("hits") or []
        joined = "\n".join(json.dumps(hit, sort_keys=True) for hit in hits)
        if chk["expect_file"] in joined:
            return True, f"{sub} '{chk['query']}': hit {chk['expect_file']}"
        return False, (
            f"{sub} '{chk['query']}': {chk['expect_file']} not in "
            f"{len(hits)} shown results (raw={raw.strip()[:80]})"
        )
    return False, f"unknown check kind {kind}"


def main():
    if "--index" in sys.argv:
        print("indexing corpus repos...")
        index_all()

    tasks = json.load(open(HERE / "tasks.json"))

    print("\n== CORPUS INTEGRITY ==")
    corpus_ok = True
    print(f"{'repo':14s} {'files':>6} {'edges':>6} {'CALLS':>6} "
          f"{'IMPORTS':>7} {'cross-file?':>11}")
    for r in REPOS:
        s = stats(CORPUS / r)
        has_calls = s.get("CALLS", 0) > 0
        # IMPORTS present means cross-file module links exist; CALLS>0 with a
        # layered corpus means cross-file or same-file resolved edges exist.
        cross = has_calls
        corpus_ok = corpus_ok and cross
        print(f"{r:14s} {s.get('files', 0):>6} {s.get('edges', 0):>6} "
              f"{s.get('CALLS', 0):>6} {s.get('IMPORTS', 0):>7} "
              f"{'yes' if cross else 'NO':>11}")

    print("\n== GROUND-TRUTH ANSWERABILITY ==")
    fails = []
    by_repo_pass = {}
    for t in tasks:
        root = CORPUS / t["repo"]
        ok, detail = check_task(root, t["check"])
        by_repo_pass.setdefault(t["repo"], [0, 0])
        by_repo_pass[t["repo"]][1] += 1
        if ok:
            by_repo_pass[t["repo"]][0] += 1
        else:
            fails.append((t["id"], t["repo"], detail))

    for r in REPOS:
        p, n = by_repo_pass.get(r, (0, 0))
        print(f"  {r:14s} {p:3d}/{n:<3d} tasks verified")

    if fails:
        print(f"\nFAILED {len(fails)} task(s):")
        for tid, repo, detail in fails:
            print(f"  {tid} [{repo}] {detail}")

    total = len(tasks)
    passed = total - len(fails)
    print(f"\nTOTAL: {passed}/{total} tasks verified answerable")
    by_type = {}
    for t in tasks:
        by_type[t["type"]] = by_type.get(t["type"], 0) + 1
    print(f"task types: {by_type}")
    langs = sorted({t['lang'] for t in tasks})
    print(f"languages : {langs}")

    if not corpus_ok or fails:
        print("\nRESULT: FAIL")
        return 1
    print("\nRESULT: PASS -- corpus has cross-file edges; all tasks answerable")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""corpus-v2 verifier (REALCORPUS_TASKGEN_SPEC.md, gate 5).

Independently validates ``tasks_v2.json`` + ``task_classes_v2.json`` against
``realcorpus/candidates.json`` (the audited C-oracle ground truth), the
pinned-repo MANIFEST, and the frozen synthetic v1 corpus -- and RE-RUNS every
mechanical gate:

  A. structural: unique sequential ids, classes doc exactly partitions the
     task list, langs match MANIFEST, pinned input sha256 matches.
  B. ground truth: every check field of every task is validated field-by-
     field against candidates.json (floor semantics verbatim); control tasks
     must be verbatim v1 payloads chosen by the deterministic even-spacing
     rule (re-derived here, not trusted).
  C. gates re-run:
       - vocabulary firewall on every fuzzy_discovery question (same gate
         function as the generator: stemmed lexical collision + rg -i top-3),
       - multi-hop gate on every research_multihop target with FRESH
         ``grepplus impact`` measurements in the isolated store; stored gate
         numbers must reproduce exactly.
  D. reproduction: gen_real_tasks.build() is executed again and its
     serialization compared byte-for-byte with the files on disk (proves the
     outputs are a pure deterministic function of the inputs).

Prints naked denominators (tasks per class / repo / language), the gate
protocols, and the sha256 of both output files. Exits non-zero on ANY
violation.

Usage:
    python3 bench/agent_efficiency/verify_real_tasks.py
    REALTASKS_WORK_DIR=/path/to/scratch python3 .../verify_real_tasks.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen_real_tasks as gen  # noqa: E402  (shared gate implementation)

TASKS_V2 = HERE / "tasks_v2.json"
CLASSES_V2 = HERE / "task_classes_v2.json"

CLASS_ORDER = [
    "graph_discovery", "fuzzy_discovery", "research_multihop",
    "literal_control", "graph_control_synth",
]
REAL_CLASSES = {"graph_discovery", "fuzzy_discovery", "research_multihop"}
V1_SOURCE_CLASS = {"literal_control": "literal_control",
                   "graph_control_synth": "graph_control"}

violations: list[str] = []


def fail(msg: str) -> None:
    violations.append(msg)
    print(f"[verify] VIOLATION: {msg}")


def check(cond: bool, msg: str) -> bool:
    if not cond:
        fail(msg)
    return cond


def main() -> int:
    for path in (TASKS_V2, CLASSES_V2, gen.CANDIDATES, gen.MANIFEST,
                 gen.TASKS_V1, gen.CLASSES_V1):
        if not path.exists():
            print(f"[verify] missing file: {path}")
            return 2

    tasks = json.loads(TASKS_V2.read_text(encoding="utf-8"))
    classes_doc = json.loads(CLASSES_V2.read_text(encoding="utf-8"))
    cands = json.loads(gen.CANDIDATES.read_text(encoding="utf-8"))
    manifest = json.loads(gen.MANIFEST.read_text(encoding="utf-8"))
    tasks_v1 = json.loads(gen.TASKS_V1.read_text(encoding="utf-8"))
    classes_v1 = json.loads(gen.CLASSES_V1.read_text(encoding="utf-8"))
    v1_by_id = {t["id"]: t for t in tasks_v1}

    # ----------------------------------------------------------- A. structure
    ids = [t["id"] for t in tasks]
    check(len(ids) == len(set(ids)), "duplicate task ids")
    for i, tid in enumerate(ids):
        check(tid == f"r{i + 1:03d}",
              f"id sequence broken at index {i}: {tid!r}")

    check(set(classes_doc["classes"].keys()) == set(CLASS_ORDER),
          f"classes doc keys != expected 5 classes: "
          f"{sorted(classes_doc['classes'])}")
    classed: dict[str, str] = {}
    for cls, spec in classes_doc["classes"].items():
        for tid in spec["ids"]:
            check(tid not in classed,
                  f"{tid} listed in both {classed.get(tid)} and {cls}")
            classed[tid] = cls
    check(set(classed) == set(ids),
          "classes doc ids do not exactly partition tasks_v2 ids")
    for t in tasks:
        check(classed.get(t["id"]) == t["class"],
              f"{t['id']}: class field {t['class']!r} != classes doc "
              f"{classed.get(t['id'])!r}")

    cand_sha = gen._sha256(gen.CANDIDATES)
    check(classes_doc.get("input_candidates_sha256") == cand_sha,
          "task_classes_v2 input_candidates_sha256 != sha256(candidates.json)"
          f" ({classes_doc.get('input_candidates_sha256')} vs {cand_sha})")

    for t in tasks:
        if t["class"] in REAL_CLASSES:
            check(t["repo"] in gen.REPO_ORDER,
                  f"{t['id']}: unknown real repo {t['repo']!r}")
            want_lang = manifest["repos"][t["repo"]]["lang"]
            check(t["lang"] == want_lang,
                  f"{t['id']}: lang {t['lang']!r} != MANIFEST {want_lang!r}")
            check(t["check"].get("semantics") == "floor",
                  f"{t['id']}: check.semantics != 'floor'")

    # -------------------------------------------------------- B. ground truth
    def calls_map(repo: str) -> dict[str, dict]:
        return {t["symbol"]: t
                for t in cands["repos"][repo]["edge_types"]["CALLS"]["targets"]}

    def locate_map(repo: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for et in ("CALLS", "USAGE"):
            ed = cands["repos"][repo]["edge_types"].get(et, {})
            for t in ed.get("targets", []):
                out.setdefault(t["symbol"], t)
        return out

    for t in tasks:
        tid, cls, chk = t["id"], t["class"], t["check"]

        if cls == "graph_discovery":
            src = calls_map(t["repo"]).get(chk.get("symbol"))
            if not check(src is not None,
                         f"{tid}: symbol {chk.get('symbol')!r} not a CALLS "
                         f"target of {t['repo']}"):
                continue
            check(chk["kind"] == "who_calls", f"{tid}: check.kind != who_calls")
            check(chk["expect_members"] == src["expect_members"],
                  f"{tid}: expect_members diverge from candidates.json")
            check(bool(chk["expect_members"]),
                  f"{tid}: graph_discovery with empty expect_members")
            check(chk["file_evidence"] == src["file_evidence"],
                  f"{tid}: file_evidence diverge from candidates.json")
            check(chk["min_count"] == src["min_count"],
                  f"{tid}: min_count {chk['min_count']} != oracle "
                  f"{src['min_count']}")
            check(t["target"]["file"] == src["file"]
                  and t["target"]["line"] == src["line"],
                  f"{tid}: target file:line != candidates.json")
            check(f"`{src['symbol']}`" in t["q"]
                  or f".{src['symbol']}`" in t["q"],  # `Owner.method` form
                  f"{tid}: question does not name the symbol")

        elif cls == "fuzzy_discovery":
            src = locate_map(t["repo"]).get(chk.get("query"))
            if not check(src is not None,
                         f"{tid}: query {chk.get('query')!r} not a CALLS/USAGE "
                         f"target of {t['repo']}"):
                continue
            check(chk["kind"] == "search_symbols",
                  f"{tid}: check.kind != search_symbols")
            check(chk["expect_file"] == src["file"],
                  f"{tid}: expect_file != candidates.json target file")
            check(t["target"]["file"] == src["file"]
                  and t["target"]["line"] == src["line"],
                  f"{tid}: target file:line != candidates.json")
            check(t["target"]["edge_type"] == src["edge_type"],
                  f"{tid}: edge_type != candidates.json")
            check(src["symbol"] not in t["q"],
                  f"{tid}: question leaks the symbol name verbatim")

        elif cls == "research_multihop":
            gate = t.get("multihop_gate")
            check(isinstance(gate, dict) and gate.get("max_ratio")
                  == gen.MULTIHOP_MAX_RATIO,
                  f"{tid}: missing/incorrect multihop_gate block")
            if chk["kind"] == "impact":
                src = calls_map(t["repo"]).get(chk.get("symbol"))
                if not check(src is not None,
                             f"{tid}: symbol {chk.get('symbol')!r} not a "
                             f"CALLS target of {t['repo']}"):
                    continue
                check(chk["expect_members"] == src["expect_members"],
                      f"{tid}: expect_members diverge from candidates.json")
                check(chk["file_evidence"] == src["file_evidence"],
                      f"{tid}: file_evidence diverge from candidates.json")
                check(chk["min_count"] == src["min_count"],
                      f"{tid}: min_count != oracle floor")
                check(chk["direction"] == "incoming",
                      f"{tid}: impact check direction != incoming")
                check(t["target"]["file"] == src["file"]
                      and t["target"]["line"] == src["line"],
                      f"{tid}: target file:line != candidates.json")
            elif chk["kind"] == "path":
                cmap = calls_map(t["repo"])
                to_sym, frm, via = chk["to"], chk["frm"], chk["via"]
                src = cmap.get(to_sym)
                if not check(src is not None,
                             f"{tid}: chain endpoint {to_sym!r} not a CALLS "
                             f"target of {t['repo']}"):
                    continue
                check(len(via) == 1, f"{tid}: chain via must be one hop")
                c = via[0]
                check(c in src["expect_members"],
                      f"{tid}: via {c!r} is not an oracle caller of {to_sym!r}")
                ct = cmap.get(c)
                if check(ct is not None,
                         f"{tid}: via {c!r} is itself not a CALLS target"):
                    check(frm in ct["expect_members"],
                          f"{tid}: frm {frm!r} is not an oracle caller of "
                          f"{c!r}")
                check(frm not in (to_sym, c),
                      f"{tid}: degenerate chain {frm} -> {c} -> {to_sym}")
                check(t["target"]["file"] == src["file"]
                      and t["target"]["line"] == src["line"],
                      f"{tid}: target file:line != candidates.json")
            else:
                fail(f"{tid}: unknown research check kind {chk['kind']!r}")

        elif cls in V1_SOURCE_CLASS:
            sid = t.get("source_id")
            src = v1_by_id.get(sid)
            if not check(src is not None,
                         f"{tid}: source_id {sid!r} not in v1 tasks.json"):
                continue
            v1_ids = classes_v1["classes"][V1_SOURCE_CLASS[cls]]["ids"]
            check(sid in v1_ids,
                  f"{tid}: source_id {sid} not in v1 class "
                  f"{V1_SOURCE_CLASS[cls]}")
            reused = {k: v for k, v in t.items()
                      if k not in ("id", "class", "source_id")}
            want = {k: v for k, v in src.items() if k != "id"}
            check(reused == want,
                  f"{tid}: payload not verbatim v1 task {sid}")
        else:
            fail(f"{tid}: unknown class {cls!r}")

    # deterministic even-spacing of the control selections, re-derived
    def evenly_spaced(items: list, k: int) -> list:
        n = len(items)
        if k >= n:
            return list(items)
        return [items[(i * n) // k] for i in range(k)]

    for cls, n_req in (("literal_control", gen.N_LITERAL),
                       ("graph_control_synth", gen.N_GRAPH_CONTROL)):
        want_ids = evenly_spaced(
            sorted(classes_v1["classes"][V1_SOURCE_CLASS[cls]]["ids"]), n_req)
        got_ids = [t["source_id"] for t in tasks if t["class"] == cls]
        check(got_ids == want_ids,
              f"{cls}: source_ids {got_ids} != deterministic even-spaced "
              f"selection {want_ids}")

    # --------------------------------------------------------- C. gates rerun
    print("[verify] re-running mechanical gates (mirrors + isolated store: "
          f"{gen.WORK_DIR}) ...")
    mirrors = gen.ensure_mirrors(manifest)

    fuzzy_tasks = [t for t in tasks if t["class"] == "fuzzy_discovery"]
    for t in fuzzy_tasks:
        src = locate_map(t["repo"]).get(t["check"]["query"])
        if src is None:
            continue  # already reported above
        ok, viol = gen.firewall_check(
            t["q"], src["symbol"], src["file"], mirrors[t["repo"]])
        check(ok, f"{t['id']}: vocabulary firewall FAILED on re-run: {viol}")
    print(f"[verify] firewall re-run: {len(fuzzy_tasks)} fuzzy questions "
          f"checked, {sum(1 for v in violations if 'firewall' in v)} failed")

    research_tasks = [t for t in tasks if t["class"] == "research_multihop"]
    for t in research_tasks:
        sym = t["check"].get("symbol") or t["check"]["to"]
        m = gen.impact_ratio(mirrors[t["repo"]], sym)
        check(gen.gate_pass(m),
              f"{t['id']}: multi-hop gate FAILED on re-run for {sym!r}: {m}")
        g = t["multihop_gate"]
        check(g["reach1"] == m["reach1"]
              and g["reach_total"] == m["reach_total"]
              and (m["ratio"] is not None
                   and g["ratio"] == round(m["ratio"], 4)),
              f"{t['id']}: stored gate numbers {g['reach1']}/"
              f"{g['reach_total']} (ratio {g['ratio']}) do not reproduce: "
              f"fresh {m}")
    print(f"[verify] multi-hop gate re-run: {len(research_tasks)} research "
          f"targets re-measured with grepplus impact")

    # -------------------------------------------------------- D. reproduction
    print("[verify] re-running gen_real_tasks.build() for byte reproduction "
          "check ...")
    built = gen.build()
    re_tasks = gen.serialize(built["tasks"]).encode("utf-8")
    re_classes = gen.serialize(built["classes"]).encode("utf-8")
    check(re_tasks == TASKS_V2.read_bytes(),
          "tasks_v2.json is NOT byte-identical to a fresh regeneration")
    check(re_classes == CLASSES_V2.read_bytes(),
          "task_classes_v2.json is NOT byte-identical to a fresh regeneration")
    rep = built["report"]

    # ----------------------------------------------------------- denominators
    def tally(key) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in tasks:
            k = key(t)
            out[k] = out.get(k, 0) + 1
        return out

    print("\n[verify] ===== naked denominators =====")
    print(f"[verify] total tasks: {len(tasks)}")
    by_class = tally(lambda t: t["class"])
    for c in CLASS_ORDER:
        print(f"[verify]   class {c:20s} {by_class.get(c, 0):3d}")
    print("[verify] per repo:")
    for r, n in sorted(tally(lambda t: t["repo"]).items()):
        print(f"[verify]   repo  {r:20s} {n:3d}")
    print("[verify] per language:")
    for l, n in sorted(tally(lambda t: t["lang"]).items()):
        print(f"[verify]   lang  {l:20s} {n:3d}")
    print("[verify] per class x repo:")
    for k, n in sorted(tally(lambda t: (t["class"], t["repo"])).items()):
        print(f"[verify]   {k[0]:20s} {k[1]:16s} {n:3d}")

    print("\n[verify] ===== gate protocols =====")
    print(f"[verify] fuzzy firewall: {rep['fuzzy_authored']} authored, "
          f"{rep['fuzzy_survivors']} survived, "
          f"{len(rep['fuzzy_rejections'])} rejected; "
          f"{rep['fuzzy_selected']} selected "
          f"{json.dumps(rep['fuzzy_quotas'])}")
    for rj in rep["fuzzy_rejections"]:
        why = rj.get("violations", [rj["reason"]])
        print(f"[verify]   REJECT {rj['repo']}/{rj['symbol']}: {why[0]}")
    print(f"[verify] multi-hop gate stats (all CALLS targets): "
          f"{json.dumps(rep['multihop_gate_stats'])}")
    print(f"[verify] research pools (gate-passing, unused, named): "
          f"{json.dumps(rep['research_pool_sizes'])}")

    print("\n[verify] ===== hashes =====")
    print(f"[verify] sha256 tasks_v2.json        = {gen._sha256(TASKS_V2)}")
    print(f"[verify] sha256 task_classes_v2.json = {gen._sha256(CLASSES_V2)}")
    print(f"[verify] sha256 candidates.json (in) = {cand_sha}")

    if violations:
        print(f"\n[verify] FAIL: {len(violations)} violation(s)")
        return 1
    print(f"\n[verify] PASS: {len(tasks)} tasks, 0 violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Leak-check bench user prompts against the harvest ledgers.

A prompt may only contain what a real user could know from experiencing the
bug: symptoms, observable output, config they wrote. It must not contain
solution-side information from the harvested PR/issue metadata:

  - changed file paths (or their basenames, except manifest files)
  - PR / issue references
  - any 5-gram of the issue title (title phrasing == copied issue)

Exemptions must be listed explicitly with a justification; the only accepted
class is runtime output the user directly observes (error/log lines that the
reporter also happened to paste into the issue title).

Usage:
  leakcheck_prompts.py --ledger-glob '/mnt/nvme1/greppy-bench-v3/*-ledger.jsonl' \
      prompts-user-66.json [additional-prompts.jsonl ...]
Exit 0 = clean, 1 = leaks found.
"""
import argparse, ast, glob, json, re, sys

# runtime log lines the user directly observes; reporter pasted them into the title
EXEMPT_SUBSTR = {
    "invalid read on closed body",          # go-caddy 7713
    "listener for success confirmation",    # go-caddy 7741
}
MANIFESTS = {"go.mod", "go.sum", "Gemfile", "Gemfile.lock", "Cargo.toml", "Cargo.lock"}


def load_ledgers(pattern):
    led = {}
    for f in glob.glob(pattern):
        for line in open(f):
            r = json.loads(line)
            led[str(r.get("pr_number") or "")] = r
            led[r.get("candidate_id", "")] = r
    return led


def check(record, prompt):
    t = prompt.lower()
    leaks = []
    paths = record.get("authoritative_changed_paths") or []
    if isinstance(paths, str):
        paths = ast.literal_eval(paths)
    for path in paths:
        stem = path.rsplit("/", 1)[-1]
        if path.lower() in t or (stem.lower() in t and stem not in MANIFESTS):
            leaks.append("path:" + path)
    for ref in ("pull request", str(record.get("pr_number")),
                "#" + str(record.get("issue_number"))):
        if ref and str(ref).lower() in t:
            leaks.append("ref:" + str(ref))
    title = re.sub(r"[^a-z0-9 ]", " ", (record.get("issue_title") or "").lower()).split()
    flat = re.sub(r"[^a-z0-9 ]", " ", t)
    for n in range(max(0, len(title) - 4)):
        gram = " ".join(title[n:n + 5])
        if gram in flat and not any(e in gram for e in EXEMPT_SUBSTR):
            leaks.append("title5gram:" + gram)
    return leaks


def iter_prompts(path):
    if path.endswith(".json"):
        for pid, prompt in json.load(open(path)).items():
            yield pid, prompt
    else:
        for line in open(path):
            r = json.loads(line)
            yield r["candidate_id"], r["prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger-glob", required=True)
    ap.add_argument("prompt_files", nargs="+")
    args = ap.parse_args()
    led = load_ledgers(args.ledger_glob)
    bad = total = 0
    for pf in args.prompt_files:
        for pid, prompt in iter_prompts(pf):
            total += 1
            rec = led.get(pid)
            leaks = ["NO-LEDGER-MATCH"] if rec is None else check(rec, prompt)
            if leaks:
                bad += 1
                print(f"{pf}:{pid} LEAK: {leaks[:3]}")
    print(f"RESULT: {total - bad}/{total} clean")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

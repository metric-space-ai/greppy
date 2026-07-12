#!/usr/bin/env python3
"""Rebuild SFT rows from all week*_raw.jsonl using the CURRENT student prompt.

Raw files are the source of truth (function + teacher summary); the student
prompt format is applied here, at training time. Writes to stdout.
Usage: python3 rebuild_sft.py > /tmp/all_sft.jsonl
"""
import glob, hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prod_gen import product_prompt, PROD_DIR

n = 0
for f in sorted(glob.glob(os.path.join(PROD_DIR, "week*_raw.jsonl"))):
    for l in open(f):
        r = json.loads(l)
        if not r.get("summary"):
            continue
        h = hashlib.sha1(r["source"].encode()).hexdigest()
        if h in globals().setdefault("_seen", set()):
            continue
        _seen.add(h)
        print(json.dumps({"prompt": product_prompt(r["source"]),
                          "completion": "\n".join(r["summary"]) + "<|im_end|>",
                          "lang": r["lang"], "repo": r["repo"]}))
        n += 1
print(f"{n} sft rows", file=sys.stderr)

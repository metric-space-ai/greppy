#!/usr/bin/env python3
"""Production SFT-data generation for the weekly nano checkpoints.

Sources: The Vault function parquet shards (per language, small or full split),
downloaded to ~/nano-summary-pilot/vault/. Language-balanced, repo-level
train/holdout split, provenance kept, docstring-strip augmentation, resumable,
drop-only validation (one fresh resample, no corrective repair).

Usage:
  source ~/.config/greppy/minimax.env
  python3 prod_gen.py --week 1 --target 150000 [--split small|full] [--dry-run 50]

Outputs (append-safe, resumable):
  prod/week<N>_raw.jsonl      audit: function + summary + provenance + tokens
  prod/week<N>_sft.jsonl      training rows (exact product prompt format)
  prod/holdout_functions.jsonl  frozen eval pool from held-out repos (never trained)
"""
import argparse, hashlib, json, os, random, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

class AdaptiveLimiter:
    """AIMD concurrency: +1 permit per RAMP_OK clean calls (up to hi),
    halve on rate limit (down to lo) with a cool-down pause."""

    def __init__(self, start=4, lo=2, hi=4, ramp_ok=20, cooldown=20):
        self.limit, self.lo, self.hi = start, lo, hi
        self.ramp_ok, self.cooldown = ramp_ok, cooldown
        self.active = 0
        self.ok_streak = 0
        self.paused_until = 0.0
        self.cv = threading.Condition()

    def __enter__(self):
        with self.cv:
            while self.active >= self.limit or time.time() < self.paused_until:
                self.cv.wait(timeout=1.0)
            self.active += 1
        return self

    def __exit__(self, *exc):
        with self.cv:
            self.active -= 1
            self.cv.notify_all()

    def on_success(self):
        with self.cv:
            self.ok_streak += 1
            if self.ok_streak >= self.ramp_ok and self.limit < self.hi:
                self.limit += 1
                self.ok_streak = 0
                print(f"[limiter] up -> {self.limit}", flush=True)
                self.cv.notify_all()

    def on_rate_limit(self):
        with self.cv:
            new = max(self.lo, self.limit // 2)
            if new != self.limit:
                print(f"[limiter] RATE LIMIT: {self.limit} -> {new}, cooling {self.cooldown}s", flush=True)
            self.limit = new
            self.ok_streak = 0
            self.paused_until = time.time() + self.cooldown

LIMITER = AdaptiveLimiter()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_run import PROMPTS, validate, call_m3

import pyarrow.parquet as pq

LANGS = ["python", "java", "javascript", "php", "c", "c_sharp", "cpp", "go", "ruby", "rust"]
LANG_LABEL = {"c_sharp": "csharp"}
BASE_URL = "https://huggingface.co/datasets/Fsoft-AIC/the-vault-function/resolve/main/data/train/{split}/{lang}-{part:05d}-of-{parts:05d}.parquet"
VAULT_DIR = os.path.expanduser("~/nano-summary-pilot/vault")
PROD_DIR = os.path.expanduser("~/nano-summary-pilot/prod")
HOLDOUT_MOD = 20          # hash(repo) % 20 == 0 -> holdout repo (never trained)
DOCSTRIP_FRAC = 0.3       # fraction of documented functions trained docstring-free
MAX_SRC_CHARS = 6000
MIN_SRC_CHARS = 80

def _remote_shards(lang, split):
    """List actual {lang}-*.parquet files under data/train/<split> via the HF tree API."""
    api = (f"https://huggingface.co/api/datasets/Fsoft-AIC/the-vault-function/"
           f"tree/main/data/train/{split}")
    with urllib.request.urlopen(urllib.request.Request(api), timeout=60) as r:
        entries = json.load(r)
    names = [os.path.basename(e["path"]) for e in entries
             if e.get("path", "").endswith(".parquet")
             and os.path.basename(e["path"]).startswith(f"{lang}-")]
    return sorted(names)

def shard_paths(lang, split):
    return sorted(
        os.path.join(VAULT_DIR, p) for p in os.listdir(VAULT_DIR)
        if p.startswith(f"{split}-{lang}-") and p.endswith(".parquet")
    ) if os.path.isdir(VAULT_DIR) else []

def download_lang(lang, split, max_shards=None):
    os.makedirs(VAULT_DIR, exist_ok=True)
    local = shard_paths(lang, split)
    if local:
        return
    remote = _remote_shards(lang, split)
    if not remote:
        raise RuntimeError(f"no shards found for {lang}/{split}")
    if max_shards:
        remote = remote[:max_shards]
    for name in remote:
        url = (f"https://huggingface.co/datasets/Fsoft-AIC/the-vault-function/"
               f"resolve/main/data/train/{split}/{name}")
        dst = os.path.join(VAULT_DIR, f"{split}-{lang}-{name.split('-', 1)[1]}"
                           if not name.startswith(f"{split}-") else name)
        # normalize local name to {split}-{lang}-<rest>
        dst = os.path.join(VAULT_DIR, f"{split}-{name}")
        if not os.path.exists(dst):
            print(f"downloading {url}", flush=True)
            urllib.request.urlretrieve(url, dst + ".tmp")
            os.rename(dst + ".tmp", dst)

def is_holdout(repo):
    return int(hashlib.sha1(repo.encode()).hexdigest(), 16) % HOLDOUT_MOD == 0

def iter_functions(lang, split, want, rng, done_keys, holdout_sink):
    """Yield up to `want` train candidates, stratified by length bucket."""
    cols = ["repo", "path", "code", "license", "hexsha", "original_docstring"]
    buckets = {"s": [], "m": [], "l": []}
    per = want // 3 + 1
    for shard in shard_paths(lang, split):
        pf = pq.ParquetFile(shard)
        avail = [c for c in cols if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=4096, columns=avail):
            for r in batch.to_pylist():
                code = r.get("code") or ""
                if not (MIN_SRC_CHARS <= len(code) <= MAX_SRC_CHARS):
                    continue
                key = hashlib.sha1(code.encode()).hexdigest()
                if key in done_keys:
                    continue
                if is_holdout(r["repo"]):
                    holdout_sink(lang, r)
                    continue
                nl = code.count("\n") + 1
                b = "s" if nl < 12 else "m" if nl <= 40 else "l"
                if len(buckets[b]) < per * 40:  # reservoir headroom
                    buckets[b].append(r)
            if all(len(v) >= per * 4 for v in buckets.values()):
                break
        if all(len(v) >= per * 4 for v in buckets.values()):
            break
    out = []
    for b in buckets.values():
        rng.shuffle(b)
        out += b[:per]
    rng.shuffle(out)
    return out[:want]

def strip_docstring(code, doc):
    if doc and doc.strip() and doc in code:
        return code.replace(doc, "").strip()
    return code

def product_prompt(source):
    # student prompt v3 ("qwen35-brief-v3"): minimal task tag, no think block
    return "<|im_start|>user\nbrief:\n" + source.strip() + "<|im_end|>\n<|im_start|>assistant\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--target", type=int, default=150000)
    ap.add_argument("--split", default="small", choices=["small", "full"])
    ap.add_argument("--dry-run", type=int, default=0)
    ap.add_argument("--workers", type=int, default=64)  # pool cap; effective concurrency = LIMITER
    args = ap.parse_args()

    rng = random.Random(1000 + args.week)
    os.makedirs(PROD_DIR, exist_ok=True)
    raw_path = os.path.join(PROD_DIR, f"week{args.week}_raw.jsonl")
    sft_path = os.path.join(PROD_DIR, f"week{args.week}_sft.jsonl")
    holdout_path = os.path.join(PROD_DIR, "holdout_functions.jsonl")

    done_keys = set()
    for wk_file in os.listdir(PROD_DIR):
        if wk_file.endswith("_raw.jsonl"):
            for l in open(os.path.join(PROD_DIR, wk_file)):
                done_keys.add(hashlib.sha1(json.loads(l)["source"].encode()).hexdigest())
    print(f"{len(done_keys)} functions already used in previous tranches", flush=True)

    target = args.dry_run or args.target
    per_lang = target // len(LANGS)
    holdout_seen = set()
    if os.path.exists(holdout_path):
        for l in open(holdout_path):
            holdout_seen.add(hashlib.sha1(json.loads(l)["source"].encode()).hexdigest())
    holdout_f = open(holdout_path, "a")
    holdout_quota = {}

    def holdout_sink(lang, r):
        # keep a bounded holdout pool per language, frozen once written
        if holdout_quota.get(lang, 0) >= 500:
            return
        key = hashlib.sha1((r.get("code") or "").encode()).hexdigest()
        if key in holdout_seen:
            holdout_quota[lang] = holdout_quota.get(lang, 0) + 1
            return
        holdout_seen.add(key)
        holdout_quota[lang] = holdout_quota.get(lang, 0) + 1
        holdout_f.write(json.dumps({"lang": LANG_LABEL.get(lang, lang), "repo": r["repo"],
                                    "path": r["path"], "source": r["code"],
                                    "license": r.get("license"), "hexsha": r.get("hexsha")}) + "\n")

    jobs = []
    for lang in LANGS:
        download_lang(lang, args.split, max_shards=(1 if args.split == "full" else None))
        cands = iter_functions(lang, args.split, per_lang, rng, done_keys, holdout_sink)
        label = LANG_LABEL.get(lang, lang)
        for r in cands:
            src = r["code"]
            doc = r.get("original_docstring")
            stripped = False
            if doc and rng.random() < DOCSTRIP_FRAC:
                s2 = strip_docstring(src, doc)
                if len(s2) >= MIN_SRC_CHARS:
                    src, stripped = s2, True
            jobs.append({"lang": label, "repo": r["repo"], "path": r["path"], "source": src,
                         "license": r.get("license"), "hexsha": r.get("hexsha"),
                         "docstring_stripped": stripped})
        print(f"{lang}: {len(cands)} candidates", flush=True)
    holdout_f.flush()
    rng.shuffle(jobs)

    def gen_one(fn):
        prompt = PROMPTS["K"]
        for k in ("lang", "path", "source"):
            prompt = prompt.replace("{" + k + "}", fn[k])
        errs, reply = ["never-ran"], ""
        attempts = 0
        while attempts < 2:  # one fresh resample, then drop (rate limits do not count)
            with LIMITER:
                reply, usage = call_m3(prompt)
            if reply.startswith("<<RATELIMIT"):
                LIMITER.on_rate_limit()
                continue
            LIMITER.on_success()
            attempts += 1
            errs = validate(reply, fn["source"], max_lines=2)
            if not errs:
                lines = [l.strip() for l in reply.splitlines() if l.strip()]
                return {**fn, "summary": lines, "tokens": usage.get("total_tokens")}
        return {**fn, "summary": None, "dropped": True, "last_errors": errs}

    raw_f, sft_f = open(raw_path, "a"), open(sft_path, "a")
    kept = dropped = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(gen_one, jobs)):
            raw_f.write(json.dumps(r) + "\n")
            if r.get("summary"):
                kept += 1
                sft_f.write(json.dumps({"prompt": product_prompt(r["source"]),
                                        "completion": "\n".join(r["summary"]) + "<|im_end|>",
                                        "lang": r["lang"], "repo": r["repo"]}) + "\n")
            else:
                dropped += 1
            if i % 200 == 0:
                raw_f.flush(); sft_f.flush()
                print(f"{i}/{len(jobs)} kept={kept} dropped={dropped}", flush=True)
    print(f"DONE week{args.week}: kept={kept} dropped={dropped} ({dropped*100.0/max(1,kept+dropped):.2f}%)", flush=True)

if __name__ == "__main__":
    main()

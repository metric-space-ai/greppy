# The release model changed upstream, and only the digest guard noticed

Found 2026-08-02 when CI went red on `2a21a8e` after being fully green on
`e89e231` — with nothing between the two commits but bench Python and markdown.

## What happened

```
fetch_model_assets: digest mismatch for Qwen3.5-0.8B-MTP-Q4_K_M.gguf
  got  b36838d6969d415e08e7f91ab4aa069dcc260ec0801ea1d00bb5dab234181200
  want 97c034cb9408ed2b2241171343dc7d46a08acb9cfa38683473d045180986eff4
```

It looked like a flaky download. It was not:

- The rerun produced **the same** `got` digest. A truncated transfer would give
  a different one each time.
- `x-linked-size` upstream is `529297152` — **byte-identical in size** to the
  local file. Same size, different hash is the signature of a re-upload, not
  corruption.
- ubuntu and macOS passed the same step in the same run because they printed
  `ok` without a `fetch` line: their `actions/cache` was warm, so they never
  downloaded. Windows missed the cache and fetched fresh.

The HF history explains it:

```
080231e8daee  2026-07-25  Qwen3.5-0.8B MTP Q4_K_M — 944k-row function-purpose finetune
e9b68576b80e  2026-07-19  Upload Qwen3.5-0.8B-MTP-Q4_K_M.gguf
```

On 25 July the navigation-hint model was replaced by the 944k-row finetune.
Verified per revision:

```
e9b68576b80e -> 97c034cb…   the pinned bytes, what 0.3.0 was built and benchmarked with
080231e8daee -> b36838d6…   the finetune
main         -> b36838d6…
```

Only that one file changed. Both tokenizers and the embedding gguf are
identical across every revision.

## The mechanism defect

`crates/cli/assets/MODEL_ASSETS.json` carried `"revision": "main"` — a **moving
branch**. Every release fetch resolved whatever `main` pointed at, and the
sha256 was the only thing standing between an upstream re-upload and a release
built on unreviewed weights. The guard worked, which is the good news; that it
was the *only* line of defence is the defect.

Worse, the failure is cache-shaped: as long as any runner's cache stays warm,
the change is invisible. It surfaced only because one platform missed its cache.

## Fixed

Each asset now carries its own immutable 40-character commit, and
`fetch_model_assets.sh` uses it (`.assets[i].revision // $REV`). All four
verified to resolve to their pinned digests, and a fresh download at a pinned
revision was proven to match.

The pins deliberately preserve **exactly what 0.3.0 was built and measured
with** — no silent model change.

## Open, and it is the owner's call

Should 0.3.0 ship the 19 July model or the 25 July 944k-row finetune?

- Every measurement so far — the 115-task navigation bench, the 33-case smoke
  gate — used the 19 July model. Switching invalidates that comparison.
- The finetune is presumably the better hint model; that is exactly the kind of
  improvement a release should carry.

Whichever is chosen, it must be chosen: the pin now makes it a decision instead
of whatever `main` happens to serve on the day CI's cache goes cold.

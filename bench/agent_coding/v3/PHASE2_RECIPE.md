# Phase 2, proven end to end: the first ready adapter

`cpp-fmt` reached `ready` on 2026-07-31. Everything below is the exact,
reproducible path; the remaining 23 repositories are throughput, not design.

```
status: {'ready': 1, 'pending': 23}
cpp-fmt: ready
```

## The five steps

1. **Image, built and pushed to a local registry.** A locally built image has
   an image ID but no registry digest, and the manifest requires `@sha256:` in
   the reference. Run a registry once
   (`docker run -d --restart=always -p 127.0.0.1:5000:5000 --name registry registry:2`),
   then tag and push; `docker inspect --format '{{index .RepoDigests 0}}'`
   yields the digest-pinned reference the manifest accepts.

2. **Mirror.** `git clone --mirror <url> /mnt/nvme1/greppy-bench-v3/mirrors/<id>.git`.

3. **Metadata.** `adapters.cli --config <cfg> metadata --repository-id <id>
   --repository-url <url> --merged-after 2026-05-01T00:00:00Z
   --merged-before 2026-07-15T23:59:59Z --all-merged-prs --output <candidates>`
   with `GITHUB_TOKEN` set. Every merged PR in the window is captured; the
   technical exclusions are applied later, and each exclusion lands in the
   ledger with its enumerated cause.

4. **Validation.** `adapters.cli --config <cfg> validate --repository-id <id>
   --mirror … --metadata … --scratch … --repetitions 2 --required-passing 1
   --offline --runner-image-id <digest> --output <ledger>`. Requires
   `GREPPY_ADAPTER_IMAGE_ID` bound to the same digest — the adapter refuses
   otherwise, which is correct.

5. **Manifest.** `build_manifest` with ALL 24 `--config` flags (it validates
   the full registry), `--image <ref@sha256:…>`, `--image-id`, and
   `--smoke-ledger <id>=<path>` for every repository whose ledger exists.
   Rows without a bound ledger stay `pending`; that is the intended default.

The manifest builder re-runs each adapter's probe locally, so it must run on
the machine that holds the toolchains and the image — gpu3, with the full
PATH from `TOOLCHAINS_gpu3.md`.

## What the ready row is bound to

```
parent_baseline        pass
parent_plus_test       fail      failure_mode: build
gold_plus_test         pass
merged_plus_test       pass
clean_room_repetitions 2
offline                true
runner_image_digest    sha256:a4a9ca3e…
changed_source         ['include/fmt/compile.h']
changed_tests          ['test/compile-test.cc']
merge_provenance       target_parent_verified, merged_result_tree_verified,
                       pr_delta_no_target_drift  (all true)
```

Note `failure_mode: build` — this candidate is exactly the case that the
pre-fix harness discarded as "budget inexecutable"
(`FINDING_compiled_languages.md`). Without that fix cpp-fmt would have yielded
nothing and the first ready adapter would not exist.

## The cost, measured rather than guessed

fmt is a small C++ repository: 21 merged PRs in the window, 4 admissible,
1 validated, roughly 25 minutes of wall time for the full two-repetition
sweep over all candidates. Repositories like kubernetes, opencv, rails and the
TypeScript compiler have one to two orders of magnitude more candidates and far
heavier builds. Plan Phase 4 in machine-days and run repositories in parallel
per toolchain profile, not sequentially.

# A profile image is not enough: version match and warmed caches

Measured 2026-07-31 while taking `rust-clap` through the Phase 2 recipe that
had just produced the first ready adapter (`cpp-fmt`).

## What happened

```
rust-clap: 19 candidates
  15  no linked issue
   2  no derivable independent behavior tests
   2  parent PASS_TO_PASS baseline failed
   0  validated
```

"Parent baseline failed" means the repository's own tests do not pass at the
parent commit inside the image — before any patch is involved. Two distinct
causes, and the first hid the second:

**1. Toolchain version.** The image carried Rust 1.83; clap at the frozen
commit requires `edition2024`, stabilized in a later toolchain:

> The package requires the Cargo feature called `edition2024`, but that
> feature is not stabilized in this version of Cargo (1.83.0)

**2. Offline dependency resolution.** With a current toolchain (cargo 1.97)
the build gets further and then fails:

> required by package `clap_bench v0.0.0` … offline mode (via `--offline`)
> can sometimes cause surprising resolution failures

The validation runs `--network none` by design. Without a pre-populated
dependency cache, every repository whose build resolves dependencies —
cargo, pip, pnpm/npm, maven/gradle, bundler — fails its parent baseline and
its candidates are silently recorded as unusable.

## Why cpp-fmt did not show this

fmt is header-only C++ with a vendored test framework: no version-sensitive
edition feature and no dependency resolution at all. It is the friendliest
repository in the registry, and it passed on the first attempt. **One green
adapter does not validate the image strategy** — it validated the harness.

## What Phase 2 therefore requires per repository

- a toolchain **version** matched to the pinned commit's requirement, not just
  the right language. Pin it in the image and record it in the adapter row;
  a floating `:1` tag will silently drift and invalidate sealed results.
- a **warmed dependency cache** baked into the image or mounted read-only:
  `cargo fetch` / `pip download` / `pnpm fetch` / `mvn dependency:go-offline`
  / `bundle package`, executed once with network, then reused offline.
- proof that the parent baseline passes offline **before** any candidate is
  judged. A repository whose baseline cannot pass offline must be reported as
  an adapter defect, never as a stream of "budget inexecutable" candidates —
  otherwise the corpus silently loses exactly the dependency-heavy projects,
  which are the large unfamiliar repositories the benchmark exists to measure.

This is the single largest remaining chunk of Phase 2 and it is per
repository, not per profile.

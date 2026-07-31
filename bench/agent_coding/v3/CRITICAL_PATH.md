# The real critical path, measured 2026-07-31

The plan says Phase 2 (adapters) and Phase 3 (metadata harvest) can be
prepared in parallel. The implementation does not allow it, and it is right
not to: `pipeline preflight` refuses with

```
missing_adapters: 24
missing_mirrors:  24
ok: false
```

The metadata stage harvests THROUGH the repository adapters, so no candidate
can be captured before its adapter exists and its mirror is present. Phases 2
and 3 are therefore sequential, not parallel. Everything downstream (4 seal,
5 smoke, 6 full run) waits behind that.

## What is done

- Phase 1 complete: no task-class quotas, no patch-size or cross-file gates,
  no navigation-pressure admission, adoption demoted to diagnostics; HMAC
  sampling with a sealed secret and a published commitment; candidate ledger
  with enumerated exclusion reasons; 60 tests green without PYTHONPATH.
- `freeze.json` exists: window 2026-05-01 .. 2026-07-15, sealed selection
  secret in `~/.config/greppy-bench-v3/selection_secret.bin` (0600, never
  committed), commitment `sha256` recorded in the freeze, registry / contract
  / manifest hashed, agent pi 0.80.2, budgets declared.
- gpu3 can host it: docker 27.2 without sudo, NVMe 1.8T free, NAS 12T free,
  all nine toolchain profiles installed, 24/24 adapters pass the tool probe.

## What Phase 2 actually requires (the long pole)

Per repository: a digest-pinned image, an offline dependency cache, verified
repository-native setup/rebuild/test argv arrays, and a two-repetition
clean-room smoke ledger bound to the adapter proof and image id. Twelve of the
24 additionally need their test invocation reviewed by hand first
(`adapters/COMMAND_VERIFICATION.md`): kubernetes, prometheus, zod, vite, the
TypeScript compiler, kafka, opencv, eslint, webpack, node, rails, discourse.

These are not small builds — kubernetes, opencv, rails, discourse and the
TypeScript compiler each carry substantial toolchains and test suites. This is
machine-days, not machine-hours, and it is the honest reason the corpus cannot
exist today.

## Model declaration and contamination

The contract already handles what I initially thought was missing:

```
candidate_pr_merged_at_or_after                     2026-05-01
declared_model_training_cutoff_must_be_on_or_before 2026-04-01
minimum_days_after_declared_cutoff                  30
unknown_or_later_model_cutoff_policy                report as sealed-corpus,
                                                    never as contamination-free
```

`freeze.json` currently declares the evaluation model as MiniMax-M3 with
revision `declared-unknown-cutoff`. Under the policy above that is permitted,
but it means the eventual result may be published as a sealed-corpus result
and must NOT be described as contamination-free. If a dated cutoff can be
obtained from the provider, replacing that revision string before the harvest
removes the caveat; doing it afterwards does not.

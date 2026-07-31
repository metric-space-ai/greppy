# Greppy 0.3.0 sealed coding corpus (V3)

V3 is the release corpus for the end-to-end claim, not a collection of
microbenchmarks for individual Greppy commands. The agent receives a real issue
and a large unfamiliar parent tree, changes code, and is graded mechanically.
Greppy use is voluntary. Capability tags are added only after the run.

## Fixed design

- 144 tasks from 24 repositories and eight graph-certified languages.
- Exactly six tasks plus at least two sealed reserves per repository (4.17% of
  released tasks); exactly 18 released tasks per language (12.5%).
- Real merged PRs that close real issues; issue title and body are the verbatim
  prompt.
- The PR's production delta is sealed gold. Its behavior tests become hidden
  `FAIL_TO_PASS`; a scoped pre-existing set is retained as `PASS_TO_PASS`.
- The agent sees a one-commit parent snapshot. Hidden tests are applied only
  after the agent stops; gold and post-parent Git objects never enter its
  filesystem or environment.
- Same model, budget, normal tools, hardware and grader in both arms. The
  treatment is the exact shipped `AGENTS.md`, loaded at run time and hashed.

The exact repository/language registry is
[`repository_registry.json`](repository_registry.json). The admission,
selection, contamination, validation, execution and storage rules are in
[`corpus_contract.json`](corpus_contract.json). Both files are preregistration
inputs and must be hashed into the sealed corpus manifest.

## Why V2 is not the release corpus

The existing material is useful for developing the harness, but not for the
general 0.3.0 claim:

- 40 serious and 90 medium candidates were harvested from only six repos.
- Validation yielded 41 runnable tasks: 25 Hugo, 7 Zod, 4 Flask, 4 Serde and 1
  Tokio; Gson yielded none.
- That makes Hugo 61% of the bank. It is validation-survivor bias, not a
  meaningful repository distribution.
- The adapter handover says 18 repos are configured, while the current `REPOS`
  mapping in `swe_bench_adapter.py` contains 17. V3 therefore treats every
  repository adapter as an explicit preflight item instead of trusting that
  statement.

V3 keeps the SWE-bench-like mechanics that are already right. It replaces the
repo selection, holdout, isolation and quota policy.

## Harvest and selection sequence

1. Before harvest, freeze and hash the registry, contract, temporal window,
   model, agent, budgets, exact selection algorithm, and selection-secret
   commitment. Escrow the secret itself.
2. Harvest every merged PR in the window for all 24 repositories. There is no
   target candidate count and no metadata prefilter by patch shape or task class.
3. Preserve one candidate-ledger row per PR. Apply only the enumerated technical
   exclusions, including linked-issue/provenance, substantive code or config,
   independent behavior tests, reproducible parent-fail/gold-pass/PASS_TO_PASS,
   offline execution, leak, denylist, and registered-budget checks.
4. On gpu3, keep one trusted builder clone per repository on configured NVMe.
   Fetch missing objects into that clone and create disposable validation
   worktrees. Do not make a full clone per candidate.
5. Measure repository scale at each parent. Validate adapter/toolchain setup,
   then prove baseline-pass, hidden `FAIL_TO_PASS` parent-fail/gold-pass and
   `PASS_TO_PASS` on two clean runs. JDK/Maven deployment or version failure is
   a hard preflight failure, never a reason to drop Java.
6. For each repository, compute `HMAC-SHA256(secret, repo_id + NUL +
   candidate_id)`, sort ascending, and validate in that order until six tasks
   plus at least two reserves pass. Replacement is only the same repository's
   next rank; task class, changed files/lines, cross-file shape, and expected
   Greppy behavior cannot affect selection.
7. Export deterministic parent trees into new one-commit repositories. Store
   immutable snapshots, tests, gold, reserves, the candidate ledger, the sealed
   selection secret, and the signed corpus manifest on configured NAS; agents
   cannot mount or read it.
8. Run exactly three complete paired trajectories and manually read all six traces.
   Fix and rerun the smoke after any prompt, binary, harness, adapter or limit
   change. Sign the reviewed smoke evidence. Only then start the 144-task run.
9. Report correctness, gross cost, cost per solved task, tokens and time.
   Report prompt overhead separately, and give both mean and median with `n`.
   Tool choices, reads and edit loops remain diagnostic.
10. Redact and checksum active NVMe artifacts, atomically publish immutable
    evidence to NAS, verify it, then apply the NVMe retention policy. Publish
    the task corpus and selection secret only after all preregistered runs end.

## Storage configuration on gpu3

No mountpoint is embedded in the corpus. Deployment provides two distinct
absolute roots:

```text
GREPPY_BENCH_NVME_ROOT  active clones, worktrees, stores, raw runs, build caches
GREPPY_BENCH_NAS_ROOT   immutable snapshots, hidden tests, gold, evidence, archive
```

The productive runner must refuse implicit fallback paths. Agent processes get
only their NVMe worktree/store and explicit read-only dependencies; they do not
get builder clones or NAS access.

## gpu3 preflight

Run the fail-closed preflight before validation, the three-trajectory smoke and
the full run:

```bash
export GREPPY_BENCH_NVME_ROOT=/configured/nvme/root
export GREPPY_BENCH_NAS_ROOT=/configured/nas/root
export JAVA_HOME=/configured/jdk17
export PATH=/configured/maven/bin:$JAVA_HOME/bin:$PATH

python3 bench/agent_coding/v3/preflight_gpu3.py \
  --config bench/agent_coding/v3/preflight.example.json \
  --report /configured/nvme/root/preflight.json
```

The paths above are placeholders; neither the corpus nor the script assumes a
mountpoint. `JAVA_HOME` is prepended for the Java probe and Maven is resolved
from `PATH`, so gpu3's local JDK 17 and Maven installations work without being
encoded in a task or manifest.

The example uses `execution_mode: container`, which matches gpu3. In that mode
the host needs only `git`, `rg`, Docker 27.2+, `pi`, and the exact Greppy 0.3.0
binary (an absolute `command_overrides.greppy` is allowed). Every language
toolchain is instead proven inside the digest-pinned image registered for each
repo adapter. `execution_mode: host` remains available and requires every
language tool directly on the host, including JDK/Maven through `JAVA_HOME` and
`PATH`.

The preflight exits `0` only when all checks pass and `2` otherwise. In both
cases it emits a JSON report. It also verifies distinct writable NVMe/NAS
devices with configured free-space floors and an executable, proof-bound
adapter probe for every one of the 24 registry repos. Its required
`network_policy` check also runs the Docker isolation audit described in
[`NETWORK_ISOLATION.md`](NETWORK_ISOLATION.md), including provider-via-proxy and
negative GitHub/DNS/direct-socket probes.
[`adapters.example.json`](adapters.example.json) is intentionally all `pending`,
so using the template without real adapters fails rather than silently
shrinking the corpus.

The productive runner does not accept an unsigned report by itself. Operations
must sign a small attestation that hashes the report and binds the exact runner,
gate/pricing contracts, Greppy binary, shipped `AGENTS.md`, provider extension,
model, agent image, network proof and byte hashes of every read-only dependency
tree. A full 144-task invocation additionally requires signed evidence for
exactly three reviewed paired smoke trajectories (six arm traces) with those
same bindings. A `--task` subset is archived as `smoke_only_subset`; it never
produces a release decision. See
[`OPERATIONS_EVIDENCE.md`](OPERATIONS_EVIDENCE.md) for the formats and signing
flow.

## Release-blocking invariants

- The final corpus is exactly 24 repos × 6 tasks, eight languages, plus at least
  two sealed reserves per repository.
- A failed candidate is replaced only by the same repository's next HMAC rank;
  no repository, class, or patch-shape backfill is allowed.
- `git rev-list --all --count` is exactly one inside every agent workspace;
  there are no remotes, alternates, hidden tests or gold-derived IDs.
- Agent network egress is denied.
- Both arms expose the same normal tool schemas and a working `rg` baseline.
- Source-open accounting includes Greppy source output (`read`, `read-smart`,
  `read-file`, `--code`) as well as built-in and shell reads.
- Any zero-denominator rate is `N/A`, never a pass.
- Missing Maven or a missing V3 repo adapter blocks the run.
- Any post-seal change creates a new identity and requires a complete rerun.

The preregistered release decision additionally requires all 144 tasks, paired
correctness 95% lower bound at least −5 percentage points,
repository-clustered ITT gross provider-cost ratio 95% upper bound at most
0.80, and valid signed operational evidence. Greppy adoption is reported only
as a diagnostic and can never gate release. Missing or invalid arms and every
zero denominator fail, and a failed full-run gate exits nonzero after preserving
the immutable archive. Token and wall-time results include paired bootstrap
intervals.

Pi's current post-hoc JSON cannot observe the workspace immediately before and
after an individual failed Greppy edit. Therefore transactionality is explicitly
reported as unobservable and is not a V3 release gate; searching logs for words
such as “partial state” is not accepted as evidence. Making it a gate requires a
real per-tool interceptor that records `git diff` and `git status` hashes on both
sides of every failed edit. The provider-key process boundary is documented
separately in [`PROVIDER_CREDENTIAL_BOUNDARY.md`](PROVIDER_CREDENTIAL_BOUNDARY.md).
Because agent shell children can currently read the provider key and make
unattributed calls, subset smokes are marked cost-invalid and the full runner
is hard-blocked until a separately attestable credential broker exists.

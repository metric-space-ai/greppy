# V3 harvest and sealing pipeline

This directory builds the private 0.3.0 end-to-end coding corpus. It preserves
the method already established by `../swe_bench_adapter.py`: a task comes from
a merged PR that closes a real issue, starts at the exact target-branch parent
before the merged result, uses the issue title and body verbatim, and is kept
only after hidden `FAIL_TO_PASS` and `PASS_TO_PASS` tests prove the change.

The corpus contract and `repository_registry.json` are preregistration inputs.
The pipeline does not alter or backfill their 24 repositories, 8 languages,
144 tasks, six tasks per repository, or at least two sealed reserves per
repository. There are no task-class or patch-shape slots.

## Storage

No mount point is assumed. Configure both roots explicitly:

```sh
export GREPPY_BENCH_NVME_ROOT=/resolved/local-nvme/greppy-bench
export GREPPY_BENCH_NAS_ROOT=/resolved/nas/greppy-bench
```

Alternatively, pass `--storage-config` with:

```json
{"nvme":{"root":"/absolute/nvme/path"},"nas":{"root":"/absolute/nas/path"}}
```

The roots must be distinct and non-nested. Preflight also rejects them when
they resolve to the same device. Mirrors, validation worktrees and mutable
scratch live on NVMe. Frozen snapshots, sealed tests, gold and evidence live
under one atomically published NAS release directory.

## Reproducible stages

Each of the 24 repositories needs an executable V3 adapter. Language
similarity is not enough: the adapter owns authoritative merged-PR/linked-issue
metadata and repository-native test result extraction. The adapter manifest is
private, versioned, and has this shape:

```json
{
  "schema_version":"greppy.agent-coding-v3.adapter-manifest.1",
  "adapters": [
    {
      "repository_id":"rust-ripgrep",
      "status":"ready",
      "toolchain_profile":"rust-cargo",
      "image":"registry.example/adapter@sha256:<64 lowercase hex>",
      "image_id":"sha256:<64 lowercase hex>",
      "proof_sha256":"<64 lowercase hex>",
      "commands": {
        "probe":["python3","adapters/ripgrep.py","probe"],
        "metadata":["python3","adapters/ripgrep.py","metadata"],
        "validation":["python3","adapters/ripgrep.py","validate"]
      }
    }
  ]
}
```

The commands are argv arrays and never run through a shell. Before metadata
harvest, `freeze.json` must contain a canonical `frozen_spec` and its SHA-256,
binding the registry, contract, adapter manifest, model, agent, budgets, exact
selection algorithm, and `sha256(selection_secret)`. The secret remains sealed
during harvest and is stored with the sealed corpus for later verification.

```sh
python3 -m bench.agent_coding.v3.pipeline preflight \
  --registry bench/agent_coding/v3/repository_registry.json \
  --adapter-manifest /nas-private/adapters.json

python3 -m bench.agent_coding.v3.pipeline metadata \
  --registry bench/agent_coding/v3/repository_registry.json \
  --freeze /nas-private/freeze.json \
  --adapter-manifest /nas-private/adapters.json \
  --metadata-env-file /run/private/github-api.env

python3 -m bench.agent_coding.v3.pipeline validate \
  --registry bench/agent_coding/v3/repository_registry.json \
  --freeze /nas-private/freeze.json \
  --adapter-manifest /nas-private/adapters.json \
  --selection-secret-file /nas-private/selection.key

python3 -m bench.agent_coding.v3.pipeline seal \
  --registry bench/agent_coding/v3/repository_registry.json \
  --contract bench/agent_coding/v3/corpus_contract.json \
  --freeze /nas-private/freeze.json \
  --candidates "$GREPPY_BENCH_NVME_ROOT/scratch/agent-coding-v3/<freeze>/validate/all.jsonl" \
  --adapter-manifest /nas-private/adapters.json \
  --denylist /nas-private/swe-and-prior-corpora.json \
  --stage-manifest "$GREPPY_BENCH_NVME_ROOT/scratch/agent-coding-v3/<freeze>/metadata/stage-manifest.json" \
  --stage-manifest "$GREPPY_BENCH_NVME_ROOT/scratch/agent-coding-v3/<freeze>/validate/stage-manifest.json" \
  --selection-secret-file /nas-private/selection.key
```

`metadata` passes the frozen creation/merge window plus `--all-merged-prs` to
every adapter. The adapter emits exactly one row for every merged PR in that
window; it has no target count and performs no patch-shape prefilter. `validate`
preserves every row, records one enumerated technical exclusion when a candidate
cannot proceed, and otherwise receives the local trusted mirror, isolated NVMe
worktree, two clean-room repetitions and an explicit offline flag. Both stages
publish per-repository JSONL atomically and then a deterministic combined
ledger. A missing adapter, mirror, toolchain, output, or malformed ledger is a
hard failure. Maven 3.9.9 and JDK 17 are already installed on gpu3's NVMe, but
must be exposed through the deployment `PATH`/`JAVA_HOME` and pass preflight;
Java/Maven can never be silently dropped because the default shell missed them.

Metadata and validation never execute adapter commands on the host. Before
each role, the pipeline resolves the manifest's digest-pinned image and proves
its local Docker image ID. Metadata runs read-only in that image with an
explicit bridge network and only its private output directory mounted.
Validation runs the same image with `--network none`; it sees only the trusted
mirror and frozen metadata ledger read-only plus its repository-specific
scratch and output directories read-write. The stage manifest records the
observed image IDs, network mode and output hashes. This prevents a container
preflight from certifying code that the productive stage later replaces with a
host implementation.
The authenticated GitHub credential is supplied only through an explicit
Docker env-file; its contents are neither placed in argv nor copied into stage
manifests or evidence.
Seal accepts only the exact validation-stage `all.jsonl` and requires both
stage manifests. Their image IDs, network modes, adapter-manifest hash and
combined-ledger hashes are copied into immutable corpus commitments; a locally
constructed replacement ledger cannot bypass the container proof.

## Validated-row contract

The sealing input binds authoritative metadata to locally re-extracted diffs.
Important fields are:

- repository registry ID, PR and linked issue numbers;
- PR creation/merge timestamps, exact merged result and exact first parent;
- merge strategy and positive provenance checks for target parent, merged tree
  and absence of target-branch drift;
- issue title/body verbatim; task class, file counts and changed-line counts are
  retained only as post-hoc diagnostics;
- exact changed source/test path lists from the adapter classifier, without any
  minimum or target based on their count;
- parent-bound repository-scale evidence meeting 200 eligible files and 25,000
  eligible LOC; task patch size and shape do not affect admission;
- proof states: parent baseline pass, parent+hidden-test fail,
  gold+hidden-test pass, complete merged result pass, two repetitions, offline;
- nonempty `fail_to_pass` and `pass_to_pass` test IDs;
- SHA-256 values for test, gold and full patches, test command, runner image,
  log hash, and validation timestamp.

The sealer re-resolves the commit and parent from the trusted local mirror,
re-splits and re-hashes every patch, and rejects any mismatch. For every
repository it computes `HMAC-SHA256(secret, repo_id + NUL + candidate_id)`, sorts
ascending, and walks that order until six tasks plus at least two reserves pass.
A failed candidate can be replaced only by that repository's next rank. There is
no cross-repository or task-class backfill.

The sealed candidate ledger has one row per harvested PR, including repository,
candidate ID, HMAC rank position, admission/exclusion decision, one enumerated
technical exclusion reason, validation outcome, final task/reserve/neither
disposition, and post-hoc shape/class diagnostics. This makes validation
survivor bias directly auditable.

Legacy V2 inputs are always loaded as local denylists. An explicit sealed
denylist must additionally attest coverage of SWE-bench and all earlier Greppy
corpora. Exact solution, issue, gold and grader-test matches receive the single
`denylisted` exclusion reason. Within-corpus issue-title, production-path and
production-diff similarities are recorded post hoc as diagnostics only; they
cannot admit, exclude, replace, or reweight a candidate.

The sealed manifest and evidence bind the raw corpus contract, registry,
freeze and adapter manifest; the canonical validated-candidate ledger; the
selection-secret commitment and algorithms; adapter proof hashes, toolchain
profiles, validation image digests, and every denylist input. Changing any of
them produces a different corpus identity.

## Agent isolation

The public/controller side contains only the natural issue prompt and a
deterministic tar of the parent source tree. The tar contains no `.git`, refs,
remotes, descendants, PR number, task ID, grader test or selection metadata.
The harness imports it into a fresh one-commit Git repository at a random
workspace path; the opaque controller ID must not enter the prompt, cwd, or
environment.

The runner must prove `git rev-list --all --count == 1` and an empty remote
list before every measured trajectory. Task IDs remain opaque 128-bit values;
no PR, parent, solution or test hash is encoded in them. Completed runtime
artifacts are checksummed and moved into the immutable NAS evidence namespace.

The test and gold patches are named only by a 128-bit opaque HMAC identifier
inside the sealed NAS directory. NAS and trusted NVMe mirrors are not mounted
in the agent sandbox. After the agent exits, the grader starts again from a
fresh parent snapshot, applies the captured agent patch, then the sealed test
patch, and evaluates `FAIL_TO_PASS` plus `PASS_TO_PASS`. Hidden tests are never
present during the measured agent phase.

Before any full 144-task run, execute three real complete paired trajectories
through snapshot import, both agent arms and sealed grading, and inspect all six
raw traces. Their smoke manifest is a required run input; any adapter, prompt,
binary, image, isolation or budget change invalidates it and requires a new
three-task smoke.

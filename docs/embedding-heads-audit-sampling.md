# Blind audit sampling and queue routing

`audit_sampling.py` freezes audit selection from a source roster containing only
identity, captured-content hashes, domain, family, length class, split, stage and
example IDs bound to their exact sanitized-input hashes. Labels, model predictions
and outcomes are rejected from the sampling roster.

Every pilot and final-test example requires both M3 and Grok. During broad data
preparation, Grok reviews a deterministic pseudorandom sample of complete sources
within each domain/family/length/split stratum. The sample contains ceil(10% of
sources) per stratum; all examples belonging to a selected output or episode are
included. Rare strata therefore have a larger sampling fraction. The exact
inclusion probability is recorded; estimates must use it and cluster by source,
not treat correlated spans as independent samples.

Conflict and uncertainty IDs add a separate targeted-review flag. They never
change the random cohort. Overlap retains both flags, so targeted cases cannot
silently enter the random quality estimate. The plan hash binds the entire source
population and sanitized input identities. Source additions require a new version;
old plans and random-selection hashes remain preserved. Seed and source population
must be committed before labels are available; deterministic hashing alone cannot
prove that this preregistration happened or prevent a caller from cherry-picking
seeds. Source completeness and stratum correctness require independent review.

```sh
python3 audit_sampling.py --sources sources.json --seed 17 --out audit-plan.json
python3 teacher_queue.py --db queue.sqlite enqueue-audit examples.jsonl \
  --sources sources.json --plan audit-plan.json
```

Routing verifies the plan from the roster, checks every supplied example against
its sanitized hash and queues all M3 work plus required Grok work. Inputs can be
sharded. Each job contains one existing bounded example so expanding the targeted
cohort cannot change already completed jobs' batch identities. A later invalid
row stops the shard; previously committed valid jobs remain resumable. Provider
workers retain existing quota, authentication, retry and lease behavior.

`admission.py --audit-plan audit-plan.json --source-roster sources.json` uses this
verified selection. Unsampled broad examples still require completed M3 labels,
complete/privacy-admitted capture and a matching independent evidence receipt.
Selected examples require Grok. Existing disagreements remain held, and M3
uncertainty always requires escalation and adjudication. Missing planned examples
are reported; a partial shard cannot claim complete population review. Without an
audit plan, the existing conservative requirement for both teachers remains.

A receipt is not self-authenticating: its evidence artifact must be independently
verified by the caller. No source is admitted just because teachers agree or a
plan names it. This change does not resolve existing Web conflicts, admit the
historical archives, open a final test or activate any model.

Validation covers source-level sampling, separate domains and rare strata,
order-independent selection, unchanged random cohorts during problem escalation,
rejection of predictions/duplicates/final-stage evasion/tampering, exact redacted
input binding, provider routing, completed-job reuse, missing population coverage
and mandatory independent evidence for unsampled examples. These are synthetic
pipeline tests, not a measured teacher-quality estimate.

# Shared native head inputs (experimental)

The Rust module `greppy_embed_native::head_input` prepares source-addressable
inputs for all three heads. The `head_inputs` example uses that same module for
offline native feature extraction. This contract is not yet wired into live
bash-smart or Web selection and does not enable a production model.

## Source and target contract

A source binds an ID, UTF-8 text and its SHA-256. Candidate target and context
spans use half-open offsets into the original UTF-8 bytes. Source verification
runs once; an immutable borrow protects subsequent preparation. Invalid hashes,
UTF-8 boundaries, empty targets, overlapping context, ambiguous identity and
unbounded conditioning are rejected.

Severity inputs contain no task conditioning. Both rankers require an explicit
task. Web additionally requires an observation ID and positive goal version.
The last action may be unknown (null); a dispatch status is preserved as
observed and is never interpreted as task success. Missing record fields stay
missing. Goal version and action context bind prepared identities separately
from prompt content, allowing content embeddings to be reused while invalidating
goal-dependent scores.

Every prompt has the classification prefix followed by a JSON object containing
the exact target text, explicit context texts, head, task and last action.
Source IDs and offsets remain metadata, outside the semantic prompt.
The input contract binds its version and limits. Each prepared row also binds
source, candidate, original target, actual target part, context used/omitted,
conditioning, exact prompt bytes and native token count.

Targets exceeding the configured byte/token budget are split at UTF-8 boundaries
into consecutive parts. Context is included as whole spans and dropped explicitly
when necessary. Task and target text are never silently truncated. A candidate
that cannot fit the part budget fails as a whole. The caller must preserve the
deterministic output on failure. Large-target parts need independent, exact-span
annotation; parent/block labels must not be projected onto them.

`log_spans` traverses every physical line, including CRLF and an unterminated
tail. This establishes addressability of late causes; efficient all-output
selection and its latency budget remain separate implementation work.

## Exact native tokenization

`PromptTokenizer::encode_prompts_exact` checks full raw token IDs against the
actual padded native batch and rejects token or byte pre-limit truncation.
Raw counts explicitly disable truncation and padding inherited from a serialized
tokenizer. The existing bounded generic retrieval API retains its behavior.
`EmbeddingGemma::embed_prompts_exact` feeds the verified batch through the same
CPU, Metal or CUDA forward path as ordinary embedding inference.

The extractor accepts JSONL requests with `source`, `candidates` and `limits`:

```sh
head_inputs /assets/tokenizer.json prepare < inputs.jsonl
head_inputs /assets/tokenizer.json cuda /assets/model.gguf < inputs.jsonl
```

Use explicit `cpu`, `metal` or `cuda`; automatic backend selection is refused.
Native feature rows bind tokenizer, executable and model checksums and the actual
backend. Each vector must have 768 finite values. Output is incremental; a
nonzero process exit invalidates the extraction as a complete job. Callers must
not admit a partial stream. There is no automatic teacher-to-training admission
bridge.

## Representation and verification boundary

This JSON target/context prompt is a new experimental representation, not the
unchanged R5 block input. It requires new embeddings and reviewed exact-span
labels. It must not be presented as the rubric-only ablation. R5 remains a
separate baseline. Encoder weights remain frozen; no encoder training occurs.

`head_input_contract` tests all-byte target coverage, a 100001-line source,
UTF-8/CRLF boundaries, context separation, goal-dependent identity, invalid
inputs and fail-closed budgets. `head_tokenizer_exact` deliberately serializes
truncation/padding and tests overflow and the byte pre-limit against known tokens.

`tools/embedding_heads/native_input_probe.py` runs only on GPU3. It captures
complete stdout/stderr and exit codes from two controlled GCC cases (warning
with exit 0, error with exit 1), embeds explicit targets with the frozen model,
and verifies preparation/inference identity and repeated native vectors.
It also covers two log goals, a synthetic Web record with goal-version change,
and an oversized Unicode target. The 100001-line fixture embeds only its explicit
tail target. The Web fixture is synthetic, not browser evidence. Process timing
includes cold startup/hashing and does not establish warm head latency.

These checks do not supply teacher labels, broad-corpus admission, calibrated
heads, backend acceptance, or successful agent workflows. Production eligibility
is explicitly false in every extracted row and report.

## GPU3 evidence, 2026-09-05

The native probe prepared 19 parts from eight candidates and embedded eight parts
from seven candidates on both CPU and CUDA. Within each backend the repeated
vectors and complete output artifacts were identical. CPU/CUDA inputs were
identical, but vectors differed: maximum absolute component difference
0.009082430973649025; minimum cosine similarity 0.9970154203726503.
This small probe does not justify sharing calibration or thresholds.

The executable SHA-256 was
`4c92ddfbe9e3aa411e597e64b5c48b5066e024e18209456e2769f46ed9ce2faf`.
It used the baseline-pinned model and tokenizer assets. Reports and complete
controlled input/output artifacts are durable under
`/Users/michaelwelsch/.local/state/greppy-heads/2026-09-05/native-input-probe-{cpu,cuda}-v1/`;
the comparison is `native-input-cpu-cuda-v1.json` alongside them.

Python annotation target construction now follows the same LF boundary rule as
Rust: lone CR, vertical tab, form feed, NEL and Unicode paragraph/line separators
remain inside their physical LF-delimited source line. Their bytes are unchanged.
Changed target boundaries change source-record IDs and require new annotations;
existing labels must not be transferred by position.

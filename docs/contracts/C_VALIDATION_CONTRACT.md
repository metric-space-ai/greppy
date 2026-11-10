# Draft: configured native C validation v1

**Owner sign-off: outstanding. This proposal must not be merged or installed as a registered authority before that sign-off.**

This is the proposed versioned extension to the registered edit contract. The rationale is that raw tree-sitter C diagnostics can reject complete compiler-valid source containing preprocessing alternatives and declaration macros. The extension uses the caller's explicitly declared compiler configuration. It does not turn an arbitrary validator's exit status into a syntax certificate.

## Surface and default

`greppy write PATH [NEW] --c-context FILE` and `greppy patch [DIFF] --c-context FILE` select this authority. FILE is relative to the selected workspace root. The context schema is `c-validation-context.v1.schema.json`. Other verbs do not accept this option. Missing/unsupported coverage is a refusal, never an ignored option. This first version supports Unix GCC/Clang; other platforms refuse validation. It does not certify C-input deletion or C++.

Without the option, existing edit validation is unchanged. The existing `--allow-syntax-errors` route is unchanged and cannot be combined with native authority. `--verify` remains a separate **post-publication** check: its failure returns17 while the edit remains applied. A native pre-publication failure never reaches that phase.

## Proposed binding semantics

The complete proposed transaction and explicitly declared project inputs are staged privately with their workspace-relative layout. Every required C translation unit is checked using the explicitly selected local GCC/Clang frontend, configuration and arguments. The native check runs even if the raw tree has no errors. A successful check authorizes publication only for the checked source bytes, context and captured input snapshot. It does not certify inactive preprocessor variants, other translation units, linking or project behavior.

Raw tree-sitter C errors/missing nodes remain in `raw_parser_observations`; they are not relabeled as zero or hidden. Their increase is not, by itself, a veto when this explicitly selected native authority succeeds. Default parser guarantees are not reinterpreted.

The frontend executable is resolved, hashed and queried for its declared family/version. Its identity, effective arguments, configuration hash, staged input hashes and compiler-produced dependency hashes are reported. The caller selects and trusts that local frontend; this is not toolchain authenticity attestation. Compiler search/override environment variables are cleared and LC_ALL is C. No tool, dependency, configuration, compiler or macro branch is silently chosen or downloaded.

Preprocessing dependency discovery precedes the complete syntax check. Missing headers, absent tools, unsupported arguments or unrepresented input paths are unavailable validation, not a claim of malformed source. Unknown execution/driver failures do not allow publication. The syntax-check dependency set and input hashes must remain stable. Context, compiler and live input preconditions are rechecked before publication; existing CAS/create-vs-replace/journal behavior remains authoritative.

A successful native receipt means the validation passed, **not** that publication happened. Publication status is reported separately. Dry runs validate staged bytes but do not publish. Already-as-sent calls with an explicit context still validate that context.

## States and receipts

`greppy.c-validation-result.v1` is the new versioned `c_validation` subrecord. It is on the applied edit record for success and in `error.c_validation` for a refused validation. Its schema is `c-validation-result.v1.schema.json`. The surrounding default edit-record schema and existing fields are retained; this registered extension explicitly names the different authority and its narrower scope.

| Native state | Meaning | Exit | Publishes |
| --- | --- | --- | --- |
| passed | Complete represented TU check succeeded for the recorded snapshot |0 after successful publish |Only after publisher succeeds |
| rejected | The selected compiler rejected the proposal |14 |No |
| unavailable | Tool/input/context/driver/observation unavailable |14 |No |
| timed_out | The existing bounded verifier stopped the owned process tree |14 |No |
| stale | Validated proposal/input/identity binding changed |12 |No |
| invalid context specification | Malformed/ambiguous context or invalid request |20 |No |

Unsupported verbs with either `--c-context FILE` or `--c-context=FILE`, and conflicting native/bypass flags, refuse with exit64. The requested authority is never removed by guessed-option recovery. A compiler rejection is not advertised as proof that every emitted diagnostic is a source-syntax error.

## First-version supported boundary

- The configuration is explicitly named. Translation-unit entries are unique, project-relative `.c` paths with represented project-relative working directories.
- Every changed `.c`/`.h` input must actually occur in a declared unit's compiler dependency manifest. Unlisted units and unconsumed changed headers refuse validation.
- Project inputs are explicit regular, nonsymlink files. Proposals may provide new headers in the same patch. Noncanonical path aliases, parent traversal, unrepresented live-workspace escapes and source/configuration self-editing are refused.
- The stage is bounded to2048 files/64MiB. The context and dependency manifest are each at most1MiB. Compiler capture is bounded to1MiB per stream. The transaction uses the existing verifier timeout with a context cap of60s; context settings cannot silently raise it.
- Argument support is explicit: one `{source}`, one `-fsyntax-only`, C-standard/diagnostic/freestanding/target/CPU options and represented relative include paths. Shell commands, response files, additional source operands, output/link/preprocess-only driver modes and unsupported options refuse validation.
- Dependency hashes for the selected execution are retained; this does not assert hermetic reproducibility of all toolchain/platform state.

## Acceptance before Owner sign-off and landing

The draft must show actual CLI publication of the complete upstream FatFs source with LF and CRLF bytes unchanged, the complete compiler-invalid trailing function refused without mutation, actual missing compiler/header/context coverage refusals, changed-input staleness and a focused mutation of the real native publish decision. Existing default and post-publication verification contracts remain unchanged. The original upstream copyright/licence is retained in the original archive and any source fixture.

No unsigned prompt change is part of this proposal.

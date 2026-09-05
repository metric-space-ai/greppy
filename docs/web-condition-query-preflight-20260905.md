# Condition-query correction: functional evidence

Date: 2026-09-05. This is a CLI correctness/diagnostics change, not a passed agent-efficiency comparison.

`web wait` and `web assert` now reject unknown query kinds before session lookup or polling. In S05, `time=500ms` had silently become repeated empty matches until the ten-second timeout. Help now explains exact whole-element text matching, regex partial matching and the fact that durations are not conditions.

The shared query parser preserves bare CSS containing attribute operators and general-sibling combinators. Its JavaScript resolver no longer performs an unconditional `querySelectorAll("*")` before direct CSS, XPath, ID and tag queries. This removes that specific redundant traversal in the generated source; it does not establish a fully incremental native index or an end-to-end engine-cost reduction.

Condition validation preserves the existing JavaScript regex dialect. The older find/extract validator's Rust-regex restriction is not extended to conditions; its dialect mismatch remains a separately reported issue. Native malformed regex errors stop the wait immediately, although their verbose internal error and incorrect retry/doctor guidance remain open diagnostics findings.

Validation:

- 13 Library tests pass, including CSS operator preservation and invalid-query rejection.
- Three CLI integration tests pass: unknown kinds for wait/assert and invalid operators fail before session lookup; valid CSS and a JavaScript lookahead reach normal session resolution.
- Real runtime: the old CLI returns matched=0/exit13 for `label~input`; the new CLI returns one match. A chain checking that sibling selector, `input[type=number]` and `text~/Basic(?= browser)/` completes all three steps successfully.
- A real-session `time=500ms` returns exit30 immediately; a malformed native regex returns engine_error/exit34 without a wait timeout. These are technical probes, not paired agent time/token measurements.
- The tested CLI SHA256 remains `50bd2254b5b13803d37a4fccca8f408c0ecd5e1a849edb657552fde4c3759784` after the native probes. Runtime SHA256 is the unchanged `57318ead7505fdf2aa7e62a89c511bc207a9c9c9848e50e11421fc208678d399`.

Two test-harness mistakes are retained: an initial `--bin` filter ran zero tests and provided no validation; the first integration test expected the symbolic NO_SESSION code in human output instead of the actual actionable text. Neither is a product failure. A later traversal-counter probe was refused before its first step because its idle session exceeded the 120-second wall-time budget; it yields no traversal measurement.

Exact source hashes and tool results: `/Users/michaelwelsch/.local/state/greppy-web-study/cli-condition-preflight-20260905-01/`. `tests-terminal.json` records the successful test set; `native-probes.json` and `native-terminal.json` retain the browser results and limitations.

S06 deliberately continues to use the old S05 CLI and runtime so its onboarding-only condition is not mixed with this product change. No installed application, frozen baseline, study prompt or previous result has been replaced by this correction. Input/output token acceptance remains unproven.

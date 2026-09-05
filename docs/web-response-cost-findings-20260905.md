# Browser response-cost findings — 2026-09-05

Greppy has not passed the efficiency requirement. This analysis separates provider-reported output-token components across 30 already completed development trials. It reads exported usage counters and visible tool-call records, not private reasoning text. It adds no new browser trials and does not overwrite earlier results.

| Development block | Pairs | Total output C/A | Reasoning output C/A | Other output C/A | Model responses C/A |
| --- | ---: | ---: | ---: | ---: | ---: |
| Series 03 text | 5 | +80.58% | +17.75% | +125.23% | 0.00% |
| Series 03 checkbox | 5 | +67.17% | 0.00% | +109.15% | +14.29% |
| Series 04 dialog | 5 | +65.68% | +18.45% | +83.70% | +60.00% |

Each percentage is the median of paired C/A changes for that metric. Component medians cannot be added to reconstruct total medians. “Other output” is reported output minus reported reasoning output; it includes generated tool-call syntax and visible prose. It is not a tokenizer estimate and cannot assign costs to individual command strings. Model response count differs from tool-call count because a response may generate a final answer or several calls.

The text block has seven model responses in both marginal medians, yet substantially more generated output for C. Its visible C1 commands repeatedly construct a full shell invocation, long wrapper path, session ID and process-handle reporting. Those long paths and mandatory separate session creation were coordinator-imposed and were removed in Series 04; they must not be assigned to the browser engine. C3/C5 additionally hit the previously reported session-flag grammar error and abandon their chains after help calls. That remains the concrete E3 diagnostics/grammar finding, rather than a new conclusion from aggregate counts.

Series 04 still has eight C model responses versus five A in the marginal medians. All five C traces contain three separate observations and a shell URL-quoting retry. The supported E1 change is to return usable page state with open/actions so an agent can make its next decision from that result. Additional observation must remain available for a deliberately different scope or a genuinely unresolved state. The quoting failures occur in zsh before Greppy runs; they are not native browser failures. Both kinds of overhead stay in total cost.

This evidence prioritizes reliable action feedback and executable, consistent command grammar before model training. It does not establish that reasoning effort or trust caused any failure, that all non-reasoning output is tool syntax, or that a particular fix will produce a predicted token saving. E1 must first retain references for the same DOM node and reject replaced nodes. Then a new paired condition can test whether agents actually stop issuing redundant observations and whether input and output tokens both fall.

Reproduction: `python3 -B bench/web_study/response_costs.py SERIES --cases text checkbox --output NEW_JSON` (Series 03) or `--cases dialog` (Series 04). The output refuses overwrite. Seven targeted accounting tests passed, including missing and inconsistent component telemetry. Derived artifacts:

- `/Users/michaelwelsch/.local/state/greppy-web-study/basic-series-20260905-03/response-cost-components-20260905.json`
- `/Users/michaelwelsch/.local/state/greppy-web-study/basic-series-20260905-04/response-cost-components-20260905.json`

Every selected trial is retained: 20/20 selected planned trials from Series 03 and 10/10 from Series 04. The remaining address trials in the broader Series 03 plan are still explicitly unrun. This is descriptive attribution, not a causal ablation, full time acceptance, actual Greppy-agent arm B, or Office acceptance.

## Transport audit before the next condition

All five Series 04 C participants forward only `text(r.output)` from `tools.exec_command`, repeating the shell wrapper and explicit yield/output limits per action. This discards `exit_code` and any `session_id`. C1 and C5 issue `web status` after a blank open response: those are browser status calls, **not host-process polls**. C1's quoted open is request `call_KIJoUWQv3Rj3CtbeMECFmuK7`, source line 22, response 24; its status is `call_6KtR9Tf4sLFhbzgPPnQBhcQQ`, lines 29/32. C5's quoted open is `call_n0vtAArKcXt8mC3feA7dWniT`; status is `call_Fog8OF9sx3EX92QK5zfqR1wK`. The exported records do not contain the inner exec result, so they cannot prove that either historical open yielded a process handle.

A separate harmless transport probe reproduces the mechanism: a two-second child returned `output: ""` plus `session_id: 70014` after the one-second yield. Polling that same handle returned exit 0 and `transport-ready`. Evidence: `/Users/michaelwelsch/.local/state/greppy-web-study/transport-probe-20260905-01.json`. This is a host transport failure mode, not a native Greppy defect or a browser performance measurement. New C dispatch instructions must retain the complete exec result (for example `text(await tools.exec_command({cmd: COMMAND}))`), poll a returned handle to completion, and preserve exit codes. Shell quoting remains necessary. This condition change must be declared; old errors and total costs remain in the report.

C1's default `web status` response additionally emits detailed coverage, receipt and inventory fields (1,477 bytes for the exported response JSON, not tokens). This is a separate reported Greppy output/usability finding: normal readiness should be concise, with diagnostic detail still available explicitly. Do not confuse a healthier transport with an engine optimization. Compare native E1 in the corrected transport condition before attributing savings.

The current opt-in chain renderer is another unmeasured integration risk: `run_chain` calls printing `dispatch_inner` for every step before the compact step marker. It does not consolidate automatic E1 page snapshots. Candidate verification must inspect both single-action feedback and a known action chain, retaining requested observations, partial progress and errors. No gain is inferred from shortening only the step markers.

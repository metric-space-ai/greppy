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

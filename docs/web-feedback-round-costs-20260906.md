# Measured model rounds around repeated submissions

The S08 public calls are now joined to the provider-usage timelines by exact call
ID. The audit verifies each annotated ID/command, reconciles all ten timeline
sums, keeps complete generations intact and records input-source hashes. It does
not inspect private reasoning. Cached input is included in input totals.

| Trial | Repeat/poll generations | Actual input | Actual output | Share of trial input/output |
|---|---:|---:|---:|---|
| A3 | 2 | 125244 | 194 | 14.36% / 9.35% |
| C2 | 3 | 124285 | 265 | 25.49% / 25.05% |
| C3 | 3 | 126131 | 300 | 25.79% / 30.33% |
| C4 | 1 | 46954 | 191 | 7.73% / 11.55% |
| C5 | 2 | 84367 | 122 | 18.96% / 13.56% |

A1/A2/A4/A5/C1 contain no annotated repeated submission rounds. A4's failure to
submit stays in the full series and is not scored as a fast success. C1 has other
costly mistakes; zero here does not mean zero overhead. Normal outcome checks,
reloads, initial submission and successful stale-ref recovery are not counted as
avoidable actions. These are measured associated generations, NOT amounts that
may be subtracted from the benchmark or promised as savings after a fix.

This narrows the engineering priority: eliminate intermediate-state ambiguity
around asynchronous effects, then measure whether repeat mutations disappear.
The correctly operated C5 transport still incurs this loop. The standard arm is
not immune: A3's click, second click and Return continue to show the dialog with
Quantity 3; a later diagnostic reports enabled=false and HTML value attribute 1.
A subsequent setValue/observation shows the completed correct reservation. The
attribute 1 is not evidence that the input property was 1: fill can change a live
value without changing its HTML attribute. No fixture-data corruption is inferred.

Source review confirms the reservation submit handler awaits send(), which awaits
fetch plus refresh(), before closing its dialog. The table row buttons are
replaced on refresh. Merely dispatching a click does not wait for those promises.
The fixture has no explicit in-flight state on the Confirm button. Do not invent
one in the tool, reinterpret the later isEnabled result as an earlier disabled
state, or weaken stale-reference rejection. No frozen fixture was changed.

The implementation requirement is an action with an explicit expected condition,
validated before mutation, tied to the action's resolved session/tab and evaluated
by the runtime. Preserve the action receipt and expose expectation held/timeout
separately. A failed expectation must retain the already executed action and its
current state, never imply rollback or automatically repeat it. A chain stops at
that failure. Malformed, stale, unsupported or ambiguous conditions must never
prove success. Post-navigation conditions must not silently inspect another tab.
This is still open: current CLI action TargetOpts has no expectation; separate
web wait --native exists and shares a validated condition compiler. A two-RPC
CLI wrapper alone would not establish the planned runtime-bundling gate.

The audit ran successfully against all ten frozen timelines:
`bench/web_study/basic_fixture/feedback_round_costs.py`.
Artifact: `/Users/michaelwelsch/.local/state/greppy-web-study/modal-feedback-20260906/round-costs.json`.
It is manual, explicit public-call annotation, not an exhaustive automatic failure
classifier. A fresh controlled ablation and broader task coverage remain required.

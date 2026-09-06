# Next-series onboarding condition

`browser_plugin_coordinated_v4` keeps the exact v3 business goals, disposable
reservation authorization, browser APIs, batching permission and short final
answer. Both arms receive one additional shared paragraph: do not message other
tasks; report remaining failure in the final response, with the coordinator
responsible for trace capture and forwarding bug reports.

This addresses an observed S07 participant deviation: unsolicited completion
messages to other tasks added model responses after the browser task. It is an
explicit prospective harness condition, not a Greppy product optimization, and
its effect must not be attributed to the browser. All actual participant tokens
and recovery work still count; no communication cost is subtracted afterward.
Older v1/v2/v3 prompts remain unchanged. Existing frozen plans are not rewritten.

Validation:17 pytest cases passed across test_onboarding.py and test_dispatch.py.
The new tests cover identical shared scope for both arms and every fixture case,
absence of task-specific selectors/answers, unchanged v3 text, the same Luna/medium
settings with no history fork, and immutable exact dispatch preparation without
claiming delivery. An initial local test fixture omitted `repeat`; this harness
mistake was corrected before the passing run. A prior unittest invocation ran
zero tests and is not counted as validation.

S08 was prepared after the matching CLI/runtime passed compact native feedback
and all 12 native wait/reference checks. Its frozen condition is
`browser_plugin_native_wait_v5`: v4 unchanged for A, with generic native wait and
action-plus-wait capability documentation for C. It gives no fixture selectors,
answers or task-specific wait predicate. This instruction cost remains counted.
All 19 onboarding/dispatch tests passed before participant dispatch.

Five serial A/C pairs now run with frozen CLI f6ef88 and runtime4a7070 on port
51084. Data: `/Users/michaelwelsch/.local/state/greppy-web-study/table-series-20260906-08`.
Root spawn requests and returned task paths are visible in this conversation;
opaque rollout assignment text prevents exact readback auditing. Tokens and independent
oracle results are required; this development series is not final acceptance.

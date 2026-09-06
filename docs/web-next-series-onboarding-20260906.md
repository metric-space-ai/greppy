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

The next series remains unprepared until the new CLI/runtime hashes and compact
feedback preflight are valid. No participant was started for this condition yet.

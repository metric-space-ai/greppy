# Untrusted error text invalidates a healthy CLI session

The preserved CLI 0.4.0 (SHA256
1d53d0a71c443f0f7fdff0ae569b89e08cd50af1f5a85308e308325124110281)
and runtime d8f437c493d7780f39bb8868da9227ce8600dadaa7a7e393936954cf1877277b
reproduce a second context-loss path despite passing the earlier foreign-session
refusal regression.

A disposable page has select#choice with one value `known` and the ordinary
option label `session was not found`. Opening and observing work. Selecting
`missing` correctly returns OPTION_NOT_FOUND/exit 34 and leaves selection alone.
The error's fenced choice data includes the label. The next implicit observation
returns NO_SESSION/30. Explicit observation with the original own session ID
still succeeds, with value `known`.

`common::is_missing_session` treats the substrings `session` and `was not found`
in error.message as lifecycle evidence, even though the typed error is
OPTION_NOT_FOUND and the matched text is page data. This causes the CLI to discard
its healthy association and can force an agent into avoidable recovery. No token
cost or event frequency is inferred from this single functional reproduction.

Probe command and all individual argv/stdout/stderr are preserved under
`/Users/michaelwelsch/.local/state/greppy-web-study/error-text-context-20260906-01`.
Root Exec2390 exited 1 at the context-preservation assertion. Both executable
hashes were unchanged, explicit own-state recovery succeeded, runtime stop
returned running=false, and the HTTP thread ended.

The reusable regression is `bench/web_study/basic_fixture/session_error_text_probe.py`.
It is deliberately red on this candidate; no failing result was replaced with a
pass. The exact bug report and requested regression fix were sent solely to
01a02118-0d61-7e10-a9d4-be496fa34879. Source repair remains with that task.

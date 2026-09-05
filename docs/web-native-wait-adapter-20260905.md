# Experimental native wait adapter

The local CLI source now accepts `web wait QUERY --native`. The default retains
its existing backend until native correctness and empirical comparisons pass.
An explicitly supplied `--interval` conflicts with `--native`; it is never
silently ignored.

The adapter validates the same query/URL/title conditions as the existing CLI,
then wraps their `{holds, detail}` result into a strictly checked Boolean.
`--absent` negates only an actual Boolean. A malformed result, stale reference,
unsupported method or runtime error does not trigger another backend.

The native path sends one `web.wait` request with explicit session/tab identity
and timeout_ms. Its request deadline uses that timeout. The socket has a five
second delivery margin for the engine's deadline response; this does not extend
the engine condition budget. Unsupported monotonic durations are rejected
before dispatch. Runtime TIMEOUT retains CLI exit 13 and held=false, while other
typed errors retain their codes, exit status and recovery. Successful native
page state and unknown fields are preserved.

Validation on 2026-09-06: rustfmt and diff check pass. All seven expectation
unit tests passed in the compiled CLI library, including the three new native
adapter cases. The same linked test binary then passed all 79 web tests,
including scoped typed-error recovery and compact output. Evidence and source
hashes: `/Users/michaelwelsch/.local/state/greppy-web-study/native-wait-unit-20260906-01`.
The CLI executable was linked and preserved as CLI82f3 on 2026-09-06. Its real
native probe passed nine checks, including delayed DOM, timeout recovery and
explicit inactive tabs. The stale-ref diagnostic and full-navigation waiter
still failed, reproducing the separately reported runtime defects. Evidence:
`/Users/michaelwelsch/.local/state/greppy-web-study/wait-native-20260906-01`.
There is no native adapter acceptance or measured efficiency improvement yet.
The completed acceptance probe must cover
later content, absence, stale refs, explicit inactive tabs, timeout budgets and
continued usability of other sessions before agent trials use this switch.

Separately, isolation/descriptor probe 03 now passes all 14 checks against
CLI SHA256 1d53d0a71c443f0f7fdff0ae569b89e08cd50af1f5a85308e308325124110281
and unchanged S07 runtime 3ee1f2e17a697b29fac40791028228d085b819af9c4d2a2042b34e877b3cdfb8.
Both own implicit sessions survive rejected foreign-session access. Both
runtimes report running=false after cleanup; the HTTP thread stopped. Evidence:
`/Users/michaelwelsch/.local/state/greppy-web-study/isolation-descriptor-20260905-03`.
This proves the scoped context fix, not the new Wait or rebuilt native Inspect.

The native waiter is not yet purely event-driven. A source-level scheduler
probe executed its actual JavaScript (SHA256
4400484dc1f1ea8bd729a81e0536f1d6bb563127809deddbb6cbf712e9469a11):
60 synthetic animation frames with an unchanged false DOM predicate produced
61 evaluations including installation; the following mutation succeeded on
execution 62 and canceled the pending frame. Evidence:
`/Users/michaelwelsch/.local/state/greppy-web-study/wait-schedule-source-probe-01.json`.
This counts evaluations in a controlled scheduler, not native CPU or real frame
cadence. The ContentWorker also paints while WebView::animating. Therefore a
single CLI/RPC request must not be equated with zero repeated predicate work.

The source probe is reproducible with `wait_schedule_probe.cjs`. General
JavaScript conditions may require frame sampling; blindly deleting rAF would
break conditions whose state changes without a DOM mutation. A restricted,
validated condition contract is needed before using mutation-only wakeups.
This is a separate optimization candidate; the frozen native correctness build
was not changed by this audit.

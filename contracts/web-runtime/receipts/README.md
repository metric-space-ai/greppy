# Web-runtime oracle receipts

Candidate (Greppy + Servo) fixtures are exercised by `session-daemon`,
`phase1-spike`, and `compat-core`. Behavior comparison against
`playwright@1.62.1` + pinned Chromium is a CI oracle job: it is not run
in this repository's default test profile because that reference stack is
a prohibited production runtime dependency and is not installed in the
default developer environment.

Until that job produces per-fixture `behavior: passing` rows, inventory
behavior stays `unverified` even when schema is `implemented` and source
tests pass.

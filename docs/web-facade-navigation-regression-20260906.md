# Native navigation state contradicts the Playwright facade

A real preserved-runtime probe confirms two failed contracts after page-driven
full-document navigation. The engine renders the destination, but the in-flight
native waiter times out and `page.url()` continues returning the previous URL.
This is a correctness and efficiency finding, not an agent token comparison.

The corrected probe is `bench/web_study/basic_fixture/facade_navigation_wait_probe.py`.
Run from `/Users/michaelwelsch/greppy-worktrees/web-efficiency`:

```sh
greppy bash-smart -- /usr/bin/python3 bench/web_study/basic_fixture/facade_navigation_wait_probe.py \
  --cli /Volumes/tmp/dev-artifacts/greppy/web-efficiency/cli-candidate-inspect-154d1a775f41/greppy \
  --runtime /Volumes/tmp/dev-artifacts/greppy/native-select-wait-preserved.oMr7VT/web-runtime \
  --scratch /Volumes/tmp/dev-artifacts/greppy/web-efficiency/facade-wait-20260906-03 \
  --evidence /Users/michaelwelsch/.local/state/greppy-web-study/facade-wait-20260906-03
```

Use fresh create-only output directories for repetition. CLI SHA256:
`154d1a775f4156c6d33742ac309e063a49563a57e16cfb6a92c97a20a8082471`.
Runtime SHA256:
`d8f437c493d7780f39bb8868da9227ce8600dadaa7a7e393936954cf1877277b`.
Both were unchanged after the run. Raw calls and terminal.json are in the
specified evidence directory. Own runtime stopped and HTTP thread ended.

The whole sequence uses one page inside one `web pw` call, explicitly navigated
to the local fixture. First, delayed DOM insertion is detected successfully.
Next a normal button click schedules `location.href='/landed'` two seconds later.
The waiter uses the correct signature:

```javascript
await page.waitForFunction(
  () => location.pathname === '/landed', undefined, { timeout: 5000 }
);
```

After its timeout, independent native reads on that same page report:

| Read | Observed value |
|---|---|
| `page.evaluate(() => location.href)` | `http://127.0.0.1:49260/landed` |
| `page.title()` | `Landed` |
| `page.locator('#landed').count()` | `1` |
| `page.url()` | `http://127.0.0.1:49260/` |
| Wait error | `TimeoutError: timeout: waitForFunction` |

The diagnostic snippet catches the wait error solely to capture these reads.
Its successful return is not interpreted as successful waiting: the outer
verifier exits **1** and records passed=false. The native CLI adapter is not
accepted by this facade probe. The old CLI polling control separately passed
waiting across the same kind of full-document navigation.

Possible causes to investigate are a waiter stored in the old window and a
facade URL cache lacking engine navigation updates. These are source-informed
hypotheses, not the proof. The contradictory native reads and timeout are the
proof. Both findings were reported to the fixed bug task with a request for
separate regressions and a verified candidate callback.

Earlier attempts are retained: 01 incorrectly expected `web pw` to inherit the
CLI's current page; its wrapper actually creates a new context/page. The public
help does not make that transition explicit. Attempt 02 navigated explicitly,
but supplied timeout options as the second argument rather than the third,
thus leaving the default timeout active. Attempt 03 corrects both harness
assumptions and still reproduces the defects. No timing conclusion is drawn
from any attempt, especially under the shared host's other build work.

The current-page/Fassade transition also remains an interface gap against the
planned shared explicit session/tab contract. It is separate from the two
reproduced native navigation defects and must not be disguised by reopening
real application state in a future agent benchmark.

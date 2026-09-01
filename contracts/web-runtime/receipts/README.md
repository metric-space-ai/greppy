# Web-runtime receipts

Production runtime MUST NOT depend on Node.js, Playwright, or Chromium.
Oracle comparison jobs invoke a pinned Playwright + Chromium stack only as
an external reference.

## Oracle receipts (`oracle-*.json`)

- `oracle-setcontent.json` — `page.setContent` + `title` / numeric `evaluate` /
  `locator.innerText` compared with playwright@1.62.1 + chromium-1234
  (Chrome for Testing 151.0.7922.34). This is not a full-surface claim.
- `oracle-dialog.json` — native `alert` return value and dialog type/message.
  Chromium delivers the live Dialog during `page.on('dialog')`; the candidate
  records Servo SimpleDialogs and exposes them via `waitForEvent('dialog')`
  after evaluate. Confirm/prompt are candidate-only in the same fixture.
- `oracle-fill.json` — `locator.fill` of a text input value.
- `oracle-console.json` — `console.log('hello-console')` text and type `log`.
  Chromium delivers ConsoleMessage during the log; the candidate records
  Servo `show_console_message` and flushes after evaluate.
- `oracle-content.json` — Page.content markers + Locator.count/innerHTML/textContent.
- `oracle-skip.json` — recorded skip used when the reference stack is absent.
  Never treat skip as `behavior:passing`.

Inventory `behavior` stays `unverified` except for the exact symbols covered
by a named **oracle** receipt.

## Source receipts (`source-*.json`)

Named Greppy fixtures that executed symbols (`source:passing`). These are not
Chromium comparisons and MUST NOT set `behavior:passing` or raise
`coverage_level` above `unverified`.

- `source-spa-hydrate.json` / `source-wait-for-function-value.json` — Page.waitForFunction
- `source-object-disposed.json` — closed/stale handle `object_disposed`
- `source-keyboard.json` — Keyboard type/press/down/up/insertText
- `source-actionability.json` — locator actionability waits
- `source-frames-fail-closed.json` — frames / nested fail-closed
- `source-evaluate-serialization.json` — evaluate special-value serialization

Schema `implemented` is not Playwright compatibility.

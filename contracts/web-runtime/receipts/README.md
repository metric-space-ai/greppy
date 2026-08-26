# Web-runtime oracle receipts

Production runtime MUST NOT depend on Node.js, Playwright, or Chromium.
Oracle comparison jobs invoke a pinned Playwright + Chromium stack only as
an external reference.

Receipts in this directory:

- `oracle-setcontent.json` — `page.setContent` + `title` / numeric `evaluate` /
  `locator.innerText` compared with playwright@1.62.1 + chromium-1234
  (Chrome for Testing 151.0.7922.34). This is not a full-surface claim.
- `oracle-dialog.json` — native `alert` return value and dialog type/message.
  Chromium delivers the live Dialog during `page.on('dialog')`; the candidate
  records Servo SimpleDialogs and exposes them via `waitForEvent('dialog')`
  after evaluate. Confirm/prompt are candidate-only in the same fixture.
- `oracle-fill.json` — `locator.fill` of a text input value.
- `oracle-skip.json` — recorded skip used when the reference stack is absent.

Inventory `behavior` stays `unverified` except for the exact symbols covered
by a named receipt. Schema `implemented` is not Playwright compatibility.

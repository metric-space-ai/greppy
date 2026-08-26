import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
page.setDefaultTimeout(800);
await page.setContent(`<!DOCTYPE html><html><body>
<button id="go">Go</button>
<button id="off" disabled>Off</button>
<button id="hid" style="display:none">Hid</button>
</body></html>`);
await page.locator("#go").click();
let disabledFailed = false;
const t0 = Date.now();
try {
  await page.locator("#off").click();
} catch (error) {
  disabledFailed = String(error.message).includes("timed out") || String(error.message).includes("actionable");
}
if (!disabledFailed) throw new Error("disabled click must not be treated as actionable");
if (Date.now() - t0 > 5_000) throw new Error("disabled click ignored page timeout");
let hiddenFailed = false;
try {
  await page.locator("#hid").click();
} catch (error) {
  hiddenFailed = String(error.message).includes("timed out") || String(error.message).includes("actionable");
}
if (!hiddenFailed) throw new Error("hidden click must not be treated as actionable");
await browser.close();

import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(`<!DOCTYPE html><html><body>
<ul>
  <li><span class="ok">apple</span></li>
  <li>banana</li>
  <li><span class="ok">apricot</span></li>
</ul>
<input id="q" value="hello">
</body></html>`);
const banana = page.locator("li").filter({ hasText: "banana" });
if ((await banana.count()) !== 1) throw new Error("filter count");
if ((await banana.innerText()).trim() !== "banana") throw new Error("filter text");
const withOk = page.locator("li").filter({ has: page.locator("span.ok") });
if ((await withOk.count()) !== 2) throw new Error("filter has count " + (await withOk.count()));
const withoutOk = page.locator("li").filter({ hasNot: page.locator("span.ok") });
if ((await withoutOk.count()) !== 1) throw new Error("filter hasNot " + (await withoutOk.count()));
if ((await withoutOk.innerText()).trim() !== "banana") throw new Error("hasNot text");
let andClosed = false;
try {
  await page.locator("li").and(page.locator("li"));
} catch (error) {
  andClosed = String(error).includes("unsupported_playwright_operation: Locator.and");
}
if (!andClosed) throw new Error("Locator.and must fail closed");
let orClosed = false;
try {
  await page.locator("li").or(page.locator("li"));
} catch (error) {
  orClosed = String(error).includes("unsupported_playwright_operation: Locator.or");
}
if (!orClosed) throw new Error("Locator.or must fail closed");
await page.locator("#q").scrollIntoViewIfNeeded({ timeout: 5_000 });
await page.locator("#q").selectText({ timeout: 5_000 });
await page.mainFrame().hover("input");
if (!(await page.mainFrame().isVisible("input"))) throw new Error("visible");
await browser.close();

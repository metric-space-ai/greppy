import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(`<!DOCTYPE html><html><body>
<ul>
  <li>apple</li>
  <li>banana</li>
  <li>apricot</li>
</ul>
<input id="q" value="hello">
</body></html>`);
const banana = page.locator("li").filter({ hasText: "banana" });
if ((await banana.count()) !== 1) throw new Error("filter count");
if ((await banana.innerText()).trim() !== "banana") throw new Error("filter text");
await page.locator("#q").scrollIntoViewIfNeeded();
await page.locator("#q").selectText();
await page.mainFrame().hover("input");
if (!(await page.mainFrame().isVisible("input"))) throw new Error("visible");
await browser.close();

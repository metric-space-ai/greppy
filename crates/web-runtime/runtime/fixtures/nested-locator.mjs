import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(`<!DOCTYPE html><html><body>
<section id="box">
  <button>Go</button>
  <p>Hello</p>
  <input placeholder="inside">
</section>
<button>Other</button>
</body></html>`);
const box = page.locator("#box");
if ((await box.getByRole("button").count()) !== 1) throw new Error("scoped role");
if ((await box.getByText("Hello").count()) !== 1) throw new Error("scoped text");
if ((await box.locator("input").count()) !== 1) throw new Error("scoped css");
if ((await box.getByPlaceholder("inside").count()) !== 1) throw new Error("scoped placeholder");
await page.locator("#box button").click();
let workersUnsupported = false;
try {
  await page.workers();
} catch (error) {
  workersUnsupported = String(error).includes("unsupported_playwright_operation: Page.workers");
}
if (!workersUnsupported) throw new Error("page.workers must fail closed");
await page.mainFrame().waitForSelector("#box");
await browser.close();

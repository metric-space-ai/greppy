import { chromium } from "playwright";

async function expectUnsupported(label, fn) {
  try {
    const result = fn();
    if (result && typeof result.then === "function") {
      await result;
      throw new Error(label + " resolved instead of throwing");
    }
  } catch (error) {
    if (String(error.message).includes(label + " resolved")) throw error;
    if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
    return;
  }
  throw new Error(label + " did not throw");
}

const browser = await chromium.launch();
const context = await browser.newContext();
const pages = [];
context.on("page", (page) => {
  pages.push(page);
});
await expectUnsupported("context.on request", () => context.on("request", () => {}));
const page = await context.newPage();
if (pages.length !== 1 || pages[0] !== page) {
  throw new Error("BrowserContext page event " + pages.length);
}
await page.setContent(
  `<!DOCTYPE html><html><body>
    <button>Go</button>
    <p>Hello</p>
    <label>Name<input id="n"></label>
  </body></html>`,
);
if ((await page.getByText("Hello", { exact: true }).count()) !== 1) {
  throw new Error("getByText exact true");
}
if ((await page.getByRole("button", { name: "Go" }).count()) !== 1) {
  throw new Error("getByRole name");
}
await expectUnsupported("getByRole checked", () => page.getByRole("button", { checked: true }));
if ((await page.getByRole("button", { name: "Go", exact: true }).count()) !== 1) {
  throw new Error("getByRole exact true");
}
await expectUnsupported("getByRole exact false", () => page.getByRole("button", { name: "Go", exact: false }));
await expectUnsupported("getByRole includeHidden", () => page.getByRole("button", { includeHidden: true }));
await expectUnsupported("getByText exact false", () => page.getByText("Hel", { exact: false }));
await expectUnsupported("getByLabel exact false", () => page.getByLabel("Name", { exact: false }));
await expectUnsupported("click force", () => page.locator("button").click({ force: true }));
await expectUnsupported("click position", () => page.locator("button").click({ position: { x: 1, y: 1 } }));
await expectUnsupported("filter hasText regex", () => page.locator("p").filter({ hasText: /Hel/ }));
await expectUnsupported("elementHandle", () => page.locator("button").elementHandle());
page.setDefaultTimeout(2_000);
await page.evaluate(() => {
  window.__n = 0;
  setTimeout(() => {
    window.__n = 1;
  }, 50);
});
if (!(await page.locator("body").waitForFunction(() => window.__n === 1))) {
  throw new Error("Locator.waitForFunction");
}
let closed = 0;
context.on("close", () => {
  closed += 1;
});
await context.close();
if (closed !== 1) throw new Error("BrowserContext close event " + closed);
await browser.close();

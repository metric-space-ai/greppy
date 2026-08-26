import { chromium } from "playwright";

function expectUnsupported(label, fn) {
  let failed = false;
  try {
    const result = fn();
    if (result && typeof result.then === "function") {
      throw new Error(label + " returned a promise instead of throwing");
    }
  } catch (error) {
    if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
    failed = true;
  }
  if (!failed) throw new Error(label + " did not throw");
}

const browser = await chromium.launch();
const context = await browser.newContext();
const pages = [];
context.on("page", (page) => {
  pages.push(page);
});
expectUnsupported("context.on request", () => context.on("request", () => {}));
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
expectUnsupported("getByRole checked", () => page.getByRole("button", { checked: true }));
if ((await page.getByRole("button", { name: "Go", exact: true }).count()) !== 1) {
  throw new Error("getByRole exact true");
}
expectUnsupported("getByRole exact false", () => page.getByRole("button", { name: "Go", exact: false }));
expectUnsupported("getByRole includeHidden", () => page.getByRole("button", { includeHidden: true }));
expectUnsupported("getByText exact false", () => page.getByText("Hel", { exact: false }));
expectUnsupported("getByLabel exact false", () => page.getByLabel("Name", { exact: false }));
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

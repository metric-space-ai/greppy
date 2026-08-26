import { chromium, selectors, errors } from "playwright";

selectors.setTestIdAttribute("data-qa");
const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(
  `<!DOCTYPE html><html><body><div data-qa="hero">ok</div><button data-testid="old">Nope</button></body></html>`,
);
if ((await page.getByTestId("hero").count()) !== 1) {
  throw new Error("setTestIdAttribute data-qa");
}
if ((await page.getByTestId("old").count()) !== 0) {
  throw new Error("default data-testid should no longer match");
}
page.setDefaultTimeout(500);
let timedOut = false;
try {
  await page.locator("#missing").click();
} catch (error) {
  timedOut =
    error instanceof errors.TimeoutError ||
    error.name === "TimeoutError" ||
    String(error.message).includes("timed out");
  if (!timedOut) throw error;
  if (error.name !== "TimeoutError") {
    throw new Error("timeout should be TimeoutError, got " + error.name + " " + error.message);
  }
}
if (!timedOut) throw new Error("missing click should time out");
await browser.close();

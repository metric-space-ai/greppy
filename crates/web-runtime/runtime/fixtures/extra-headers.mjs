import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setExtraHTTPHeaders({ "x-greppy-test": "yes" });
await page.goto(fixtureUrl);
const text = (await page.locator("body").innerText()).trim();
if (text !== "HEADER_OK") {
  throw new Error("expected HEADER_OK from extra headers, got " + JSON.stringify(text));
}
await browser.close();

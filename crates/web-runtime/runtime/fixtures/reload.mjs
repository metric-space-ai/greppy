import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(fixtureUrl);
await page.evaluate(() => {
  document.title = "changed";
});
await page.reload();
const title = await page.title();
if (title === "changed") {
  throw new Error("reload did not restore document title");
}
await browser.close();

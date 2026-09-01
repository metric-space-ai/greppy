import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(
  "<!DOCTYPE html><html><body><button>A</button><button>B</button></body></html>",
);
let strict = false;
try {
  await page.locator("button").click();
} catch (error) {
  if (String(error.message).includes("strict mode")) {
    strict = true;
  } else {
    throw error;
  }
}
if (!strict) throw new Error("locator.click must fail closed on two buttons");
if ((await page.locator("button").count()) !== 2) throw new Error("count");
await page.locator("button").first().click();
await browser.close();
